import Foundation
import MLX
import MLXNN
import MLXRandom

/// Pre-computed reference-audio conditioning consumed by `IndexTTSv2.generate`.
///
/// In the full pipeline these come from the torch-free preprocessing stack
/// (W2V-BERT → semantic feats; semantic-codec → codes; CAMPPlus → style;
/// 22 kHz mel). Until that lands they can be supplied directly (e.g. from a
/// precomputed `.npz`/`.safetensors`) so the MLX generation chain is testable.
public struct SpeakerConditioning {
    /// (1, T, 1024) W2V-BERT semantic features (NLC).
    public var spkCondEmb: MLXArray
    /// (1, 192) CAMPPlus style embedding.
    public var style: MLXArray
    /// (1, Lp, 512) length-regulated semantic codes of the reference (== refMel length).
    public var promptCondition: MLXArray
    /// (1, 80, L) reference mel (CFM prompt). `L == Lp`.
    public var refMel: MLXArray

    public init(spkCondEmb: MLXArray, style: MLXArray, promptCondition: MLXArray, refMel: MLXArray) {
        self.spkCondEmb = spkCondEmb
        self.style = style
        self.promptCondition = promptCondition
        self.refMel = refMel
    }

    /// Load conditioning arrays from a `.safetensors` (or MLX `.npz`) bundle with
    /// keys `spk_cond_emb`, `style`, `prompt_condition`, `ref_mel`.
    public static func load(from url: URL) throws -> SpeakerConditioning {
        let w = try MLX.loadArrays(url: url)
        func get(_ k: String) throws -> MLXArray {
            guard let a = w[k] else { throw IndexTTS2Error.missingWeights("speaker: \(k)") }
            return a.asType(.float32)
        }
        return SpeakerConditioning(
            spkCondEmb: try get("spk_cond_emb"),
            style: try get("style"),
            promptCondition: try get("prompt_condition"),
            refMel: try get("ref_mel"))
    }
}

/// Knobs for `IndexTTSv2.generate`, defaults matching `generate_v2.py`.
public struct GenerationOptions {
    public var maxMelTokens = 1500
    public var maxTextTokensPerSegment = 120
    public var intervalSilenceMs = 200
    public var temperature: Float = 0.8
    public var topP: Float = 0.8
    public var topK = 30
    public var repetitionPenalty: Float = 10.0
    /// CFM Euler ODE steps. Lowered from the reference's 25 to 20: on Apple
    /// Silicon (fp16 CFM) the spectral change vs 25 steps is ~0.012 (mean
    /// log-spec corr 0.988), inaudible, for a small extra speedup.
    public var diffusionSteps = 20
    public var cfgRate: Float = 0.7
    public var segmentOverlapMs = 50
    public var speed: Float = 1.0
    public var seed: UInt64? = nil
    public var verbose = false
    /// When true, accumulate per-stage wall-clock timing into `StageTimer.shared`.
    /// Adds `eval` boundaries between stages, so leave off for production runs.
    public var profile = false
    /// Optional separate emotion-reference W2V features (1, T, 1024). When nil,
    /// emotion is taken from the speaker reference audio (the default).
    public var emotionEmb: MLXArray? = nil
    public init() {}
}

/// IndexTTS-2 inference pipeline (MLX). Mirrors `generate_v2.py::IndexTTSv2.generate`
/// for the already-ported generation stack: GPT v2 autoregression → S2Mel CFM →
/// BigVGAN v2. Reference-audio preprocessing is supplied via `SpeakerConditioning`.
public final class IndexTTSv2 {
    public let config: IndexTTS2Config
    let gpt: UnifiedVoiceV2
    let s2mel: S2Mel
    let bigvgan: BigVGANV2
    let vq2emb: VQ2Emb
    public let tokenizer: TextTokenizer
    public let sampleRate: Int

    /// `computeDType` sets the precision of the CFM/DiT estimator and BigVGAN
    /// vocoder (the two heaviest stages). `.bfloat16` ≈ 2× memory-bandwidth
    /// throughput with negligible quality loss; the GPT and CFM Euler loop stay
    /// fp32. Default `.float32` preserves the original numerics.
    public init(modelDir: URL, verbose: Bool = false, computeDType: DType = .float32) throws {
        self.config = try IndexTTS2Config.load(from: modelDir.appendingPathComponent("config.json"))
        self.sampleRate = config.sampleRate
        self.gpt = try UnifiedVoiceV2.fromPretrained(
            directory: modelDir, config: config, verbose: verbose)
        self.s2mel = try S2Mel.fromPretrained(directory: modelDir, verbose: verbose)
        self.bigvgan = BigVGANV2()
        try loadWeights(
            into: bigvgan, from: modelDir.appendingPathComponent("bigvgan.safetensors"),
            label: "bigvgan", verbose: verbose)
        self.vq2emb = try VQ2Emb.load(from: modelDir.appendingPathComponent("vq2emb.safetensors"))
        self.tokenizer = try TextTokenizer(
            modelPath: modelDir.appendingPathComponent("tokenizer.model"))

        if computeDType != .float32 {
            bigvgan.computeDType = computeDType
            castParameters(bigvgan, to: computeDType)
            s2mel.cfm.estimator.computeDType = computeDType
            castParameters(s2mel.cfm.estimator, to: computeDType)
            if verbose {
                FileHandle.standardError.write(
                    Data("✓ CFM/DiT + BigVGAN cast to \(computeDType)\n".utf8))
            }
        }
    }

    /// Synthesize speech for `text` using the reference `speaker` conditioning.
    /// Returns mono 22.05 kHz float samples in [-1, 1].
    public func generate(
        text: String, speaker: SpeakerConditioning, options: GenerationOptions = GenerationOptions()
    ) -> [Float] {
        if let seed = options.seed { MLXRandom.seed(seed) }
        let prof = options.profile

        // GPT conditioning: speaker (Conformer+Perceiver) + emotion (reference audio).
        let conditioning = timed(prof, "GPT cond") { () -> MLXArray in
            let spkNCL = speaker.spkCondEmb.transposed(0, 2, 1)  // (1, 1024, T)
            let speechCond = gpt.getConditioning(spkNCL)
            // Emotion from a separate reference if supplied, else the speaker reference.
            let emoNCL = (options.emotionEmb ?? speaker.spkCondEmb).transposed(0, 2, 1)
            let emoVec = gpt.getEmovec(emoNCL)
            let c = gpt.prepareConditioningLatents(speechCond, emoVec)
            eval(c)
            return c
        }

        // Tokenize + segment.
        let tokens = tokenizer.tokenize(text)
        let segments = tokenizer.splitSegments(
            tokens, maxTokensPerSegment: options.maxTextTokensPerSegment)
        if options.verbose {
            log("text tokens: \(tokens.count), segments: \(segments.count)")
        }

        let silenceCount = options.intervalSilenceMs > 0 && segments.count > 1
            ? sampleRate * options.intervalSilenceMs / 1000 : 0

        var allAudio: [[Float]] = []
        var usedSilence = false
        for (segIdx, segment) in segments.enumerated() {
            let ids = tokenizer.convertTokensToIds(segment)
            let textTokens = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])

            // GPT autoregressive mel-code generation.
            var melCodes = timed(prof, "GPT AR") {
                gpt.generateMelCodes(
                    conditioning: conditioning, textTokens: textTokens,
                    maxMelTokens: options.maxMelTokens, temperature: options.temperature,
                    topK: options.topK, topP: options.topP,
                    repetitionPenalty: options.repetitionPenalty, verbose: options.verbose)
            }
            if prof { StageTimer.shared.tally("mel_tokens", Double(melCodes.count)) }
            melCodes = compressSilence(melCodes)
            if options.verbose { log("segment \(segIdx + 1): \(melCodes.count) mel tokens") }
            if melCodes.isEmpty { continue }

            let segAudio = decodeSegment(
                melCodes: melCodes, conditioning: conditioning, textTokens: textTokens,
                speaker: speaker, options: options)
            allAudio.append(segAudio)

            if silenceCount > 0 && segIdx < segments.count - 1 {
                allAudio.append([Float](repeating: 0, count: silenceCount))
                usedSilence = true
            }
        }

        if allAudio.isEmpty { return [] }

        var audio: [Float]
        if allAudio.count == 1 {
            audio = allAudio[0]
        } else if options.segmentOverlapMs > 0 && !usedSilence {
            audio = crossfadeSegments(
                allAudio, sampleRate: sampleRate, overlapMs: options.segmentOverlapMs)
        } else {
            audio = allAudio.flatMap { $0 }
        }

        if options.speed != 1.0 {
            audio = timeStretchWSOLA(audio, rate: options.speed, sampleRate: sampleRate)
        }
        return audio
    }

    /// GPT latent → S2Mel (gpt_layer + vq2emb + length reg + CFM) → BigVGAN for one segment.
    private func decodeSegment(
        melCodes: [Int], conditioning: MLXArray, textTokens: MLXArray,
        speaker: SpeakerConditioning, options: GenerationOptions
    ) -> [Float] {
        let prof = options.profile
        let codesMx = MLXArray(melCodes.map { Int32($0) }).reshaped([1, melCodes.count])

        // GPT second pass → per-token latent, projected to semantic content.
        let catCondition = timed(prof, "GPT latent + LR") { () -> MLXArray in
            var latent = gpt.forwardLatent(
                conditioning: conditioning, textTokens: textTokens, melCodes: codesMx)
            latent = s2mel.gpt_layer(latent)  // (1, T, 1024)

            // vq2emb codes → content, add latent.
            var sInfer = vq2emb(codesMx).transposed(0, 2, 1)  // (1, T, 1024)
            sInfer = sInfer + latent

            // Length-regulate to mel length, prepend reference prompt condition.
            let targetLen = Int(Double(melCodes.count) * 1.72)
            let cond = s2mel.length_regulator(sInfer, targetLen: targetLen)  // (1, targetLen, 512)
            let cc = concatenated([speaker.promptCondition, cond], axis: 1)
            if prof { eval(cc) }
            return cc
        }

        // CFM diffusion → mel, trim the prompt region.
        let promptLen = speaker.refMel.dim(2)
        let melOut = timed(prof, "CFM") { () -> MLXArray in
            var m = s2mel.cfm.inference(
                mu: catCondition, prompt: speaker.refMel, style: speaker.style,
                nTimesteps: options.diffusionSteps, temperature: 1.0, cfgRate: options.cfgRate)
            m = m[0..., 0..., promptLen...]
            if prof { eval(m) }
            return m
        }

        // BigVGAN vocoder + peak-normalize / clip.
        return timed(prof, "BigVGAN") { () -> [Float] in
            let audioOut = bigvgan(melOut)  // (1, 1, samples)
            var seg = audioOut[0, 0]
            let peak = MLX.abs(seg).max().item(Float.self)
            if peak > 1.0 { seg = seg / MLXArray(max(peak, 1e-6)) }
            seg = MLX.clip(seg, min: MLXArray(Float(-0.99)), max: MLXArray(Float(0.99)))
            eval(seg)
            return seg.asArray(Float.self)
        }
    }

    private func log(_ s: String) {
        FileHandle.standardError.write(Data((s + "\n").utf8))
    }
}

/// Run `body`, recording elapsed time under `name` in `StageTimer.shared` when
/// `enabled`; otherwise run it directly with no profiling overhead.
@inline(__always)
func timed<T>(_ enabled: Bool, _ name: String, _ body: () -> T) -> T {
    enabled ? StageTimer.shared.measure(name, body) : body()
}
