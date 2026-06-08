import Foundation
import MLX
import MLXNN

/// UnifiedVoice v2 — the IndexTTS-2 GPT.
///
/// Extends the 1.5 GPT (speaker conditioning via Conformer + PerceiverResampler)
/// with emotion conditioning: a second Conformer + single-latent perceiver feed
/// `emovec_layer`/`emo_layer`, and a `speed_emb` adds two duration tokens. The
/// transformer backbone is the same GPT-2 stack, here loaded 8-bit quantized.
///
/// Port of `models/gpt_v2.py`. The speaker/emotion conditioning inputs are
/// W2V-BERT semantic features (1024-d), so both Conformers use `inputSize = 1024`.
public final class UnifiedVoiceV2: Module {
    let cfg: IndexTTS2Config.GPT

    // Speaker conditioning
    let conditioning_encoder: Conformer
    let perceiver_encoder: PerceiverResampler
    // Emotion conditioning
    let emo_conditioning_encoder: Conformer
    let emo_perceiver_encoder: PerceiverResampler
    let emo_layer: Linear
    let emovec_layer: Linear
    let speed_emb: Embedding
    // Token + position embeddings
    let text_embedding: Embedding
    let mel_embedding: Embedding
    let mel_pos_embedding: LearnedPositionEncoding
    let text_pos_embedding: LearnedPositionEncoding
    // Backbone + heads
    let gpt: GPT2Model
    let final_norm: LayerNorm
    let text_head: Linear
    let mel_head: Linear

    public init(_ c: IndexTTS2Config.GPT) {
        self.cfg = c
        let modelDim = c.model_dim

        var ca = ConformerArgs()
        ca.inputSize = 1024
        ca.outputSize = 512
        ca.numBlocks = 6
        ca.linearUnits = 2048
        ca.attentionHeads = 8
        ca.inputLayer = "conv2d2"
        ca.perceiverMult = 2
        self.conditioning_encoder = Conformer(ca)
        self.perceiver_encoder = PerceiverResampler(
            nDim: modelDim, nDimContext: 512, nLatents: c.condition_num_latent,
            nHeads: 8, nFFMult: 2)

        var ea = ConformerArgs()
        ea.inputSize = 1024
        ea.outputSize = 512
        ea.numBlocks = 4
        ea.linearUnits = 1024
        ea.attentionHeads = 4
        ea.inputLayer = "conv2d2"
        ea.perceiverMult = 2
        self.emo_conditioning_encoder = Conformer(ea)
        self.emo_perceiver_encoder = PerceiverResampler(
            nDim: 1024, nDimContext: 512, nLatents: 1, nHeads: 4, nFFMult: 2)

        self.emo_layer = Linear(modelDim, modelDim)
        self.emovec_layer = Linear(1024, modelDim)
        self.speed_emb = Embedding(embeddingCount: 2, dimensions: modelDim)

        self.text_embedding = Embedding(
            embeddingCount: c.number_text_tokens + 1, dimensions: modelDim)
        self.mel_embedding = Embedding(embeddingCount: c.number_mel_codes, dimensions: modelDim)
        self.mel_pos_embedding = LearnedPositionEncoding(
            seqLen: c.max_mel_tokens + 2 + 1, modelDim: modelDim)
        self.text_pos_embedding = LearnedPositionEncoding(
            seqLen: c.max_text_tokens + 2, modelDim: modelDim)

        self.gpt = GPT2Model(nEmbd: modelDim, nHead: c.heads, nLayer: c.layers)
        self.final_norm = LayerNorm(dimensions: modelDim)
        self.text_head = Linear(modelDim, c.number_text_tokens + 1)
        self.mel_head = Linear(modelDim, c.number_mel_codes)
        super.init()
    }

    // MARK: - Conditioning

    /// `speechCondNCL`: (B, 1024, T) semantic features. Returns (B, latents, dim).
    func getConditioning(_ speechCondNCL: MLXArray) -> MLXArray {
        let x = speechCondNCL.transposed(0, 2, 1)  // NCL -> NLC
        return perceiver_encoder(conditioning_encoder(x))
    }

    /// `speechCondNCL`: (B, 1024, T). Returns the emotion vector (B, dim).
    func getEmovec(_ speechCondNCL: MLXArray) -> MLXArray {
        let x = speechCondNCL.transposed(0, 2, 1)
        let conds = emo_perceiver_encoder(emo_conditioning_encoder(x))  // (B, 1, 1024)
        let raw = conds.reshaped([conds.dim(0), conds.dim(2)])           // (B, 1024)
        return emo_layer(emovec_layer(raw))                             // (B, dim)
    }

    /// Combine speaker conditioning (B, latents, dim) + emotion (B, dim) and append
    /// the two speed tokens, yielding (B, latents + 2, dim).
    func prepareConditioningLatents(_ speechCond: MLXArray, _ emoVec: MLXArray) -> MLXArray {
        let b = speechCond.dim(0)
        let condsWithEmo = speechCond + emoVec.expandedDimensions(axis: 1)
        let zeros = MLXArray.zeros([b], dtype: .int32)
        let ones = MLXArray.ones([b], dtype: .int32)
        let durationEmb = speed_emb(zeros).expandedDimensions(axis: 1)       // (B,1,dim)
        let durationEmbHalf = speed_emb(ones).expandedDimensions(axis: 1)    // (B,1,dim)
        return concatenated([condsWithEmo, durationEmbHalf, durationEmb], axis: 1)
    }

    /// Build [conditioning, text_emb] with text start/stop tokens.
    func prepareInputs(_ conditioning: MLXArray, _ textTokens: MLXArray) -> MLXArray {
        let b = textTokens.dim(0)
        let start = MLXArray.full([b, 1], values: MLXArray(Int32(cfg.start_text_token)))
        let stop = MLXArray.full([b, 1], values: MLXArray(Int32(cfg.stop_text_token)))
        let t = concatenated([start, textTokens, stop], axis: 1)
        let textEmb = text_embedding(t) + text_pos_embedding(t)
        return concatenated([conditioning, textEmb], axis: 1)
    }

    // MARK: - Autoregressive generation

    /// Run the AR loop and return the generated mel-code ids (excluding stop).
    func generateMelCodes(
        conditioning: MLXArray,
        textTokens: MLXArray,
        maxMelTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float,
        verbose: Bool = false
    ) -> [Int] {
        var inputEmb = prepareInputs(conditioning, textTokens)
        let melStart = MLXArray([Int32(cfg.start_mel_token)]).reshaped([1, 1])
        let melStartEmb = mel_embedding(melStart) + mel_pos_embedding(melStart, offset: 0)
        inputEmb = concatenated([inputEmb, melStartEmb], axis: 1)
        eval(inputEmb)

        let cache = (0 ..< cfg.layers).map { _ in KVCache() }
        var melCodes: [Int] = []

        var hidden = gpt(inputEmb, cache: cache)
        for i in 0 ..< maxMelTokens {
            let lastIdx = hidden.dim(1) - 1
            let last = final_norm(hidden[0..., lastIdx ..< (lastIdx + 1), 0...])  // (1,1,dim)
            let logits = mel_head(last)
            let tok = sampleMelToken(
                logits, temperature: temperature, topK: topK, topP: topP,
                repetitionPenalty: repetitionPenalty, generated: melCodes)
            if tok == cfg.stop_mel_token { break }
            melCodes.append(tok)

            // Reference (`gpt_v2.py`) feeds the token back at mel position
            // `len(mel_codes) + 1` (mel_start is position 0).
            let tokA = MLXArray([Int32(tok)]).reshaped([1, 1])
            let pos = melCodes.count + 1
            let emb = mel_embedding(tokA) + mel_pos_embedding(tokA, offset: pos)
            hidden = gpt(emb, cache: cache)
            eval(hidden)
            if verbose && (i + 1) % 100 == 0 {
                FileHandle.standardError.write(Data("  generated \(i + 1) mel tokens\n".utf8))
            }
        }
        return melCodes
    }

    /// Second forward pass: returns the per-mel-token GPT latent (B, melLen, dim)
    /// consumed by S2Mel. Port of `forward_latent`.
    func forwardLatent(
        conditioning: MLXArray, textTokens: MLXArray, melCodes: MLXArray
    ) -> MLXArray {
        let b = textTokens.dim(0)
        let melLen = melCodes.dim(1)

        let tStart = MLXArray.full([b, 1], values: MLXArray(Int32(cfg.start_text_token)))
        let tStop = MLXArray.full([b, 1], values: MLXArray(Int32(cfg.stop_text_token)))
        let t = concatenated([tStart, textTokens, tStop], axis: 1)
        let textEmb = text_embedding(t) + text_pos_embedding(t)

        let mStart = MLXArray.full([b, 1], values: MLXArray(Int32(cfg.start_mel_token)))
        let mStop = MLXArray.full([b, 1], values: MLXArray(Int32(cfg.stop_mel_token)))
        let m = concatenated([mStart, melCodes, mStop], axis: 1)
        let melEmb = mel_embedding(m) + mel_pos_embedding(m, offset: 0)

        let emb = concatenated([conditioning, textEmb, melEmb], axis: 1)
        let hidden = gpt(emb, cache: nil)

        let condLen = conditioning.dim(1)
        let enc = final_norm(hidden[0..., condLen..., 0...])
        let textLen = textEmb.dim(1)
        return enc[0..., textLen ..< (textLen + melLen), 0...]
    }

    // MARK: - Loading

    /// Build the GPT, optionally quantize the backbone, then load `gpt.safetensors`.
    public static func fromPretrained(
        directory: URL, config: IndexTTS2Config, verbose: Bool = false
    ) throws -> UnifiedVoiceV2 {
        let model = UnifiedVoiceV2(config.gpt)
        if let bits = config.quantize_bits {
            quantize(model: model.gpt, groupSize: 64, bits: bits)
        }
        try loadWeights(
            into: model, from: directory.appendingPathComponent("gpt.safetensors"),
            label: "gpt", verbose: verbose)
        return model
    }
}
