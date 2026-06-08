import Foundation
import MLX

// MARK: - Reference-audio preprocessing (torch-free)
//
// Ports `generate_v2.py::_process_reference_audio`: a reference .wav -> the four
// conditioning tensors consumed by `IndexTTSv2.generate`. Bundles the three
// preprocessing models (W2V-BERT, RepCodec semantic codec, CAMPPlus) plus the
// kaldi fbank and 22 kHz mel. The length-regulator step (which needs S2Mel) is
// applied by `IndexTTSv2.makeSpeaker`.

public final class ReferenceEncoder {
    let w2v: W2VSpeakerEncoder
    let repcodec: RepCodec
    let campplus: CAMPPlus

    /// `dir` holds the converted preprocessing weights:
    /// `w2vbert_mlx.safetensors`, `w2vbert_stats.safetensors`,
    /// `semantic_codec_mlx.safetensors`, `campplus_mlx.safetensors`.
    public init(dir: URL, verbose: Bool = false) throws {
        self.w2v = try W2VSpeakerEncoder(
            weights: dir.appendingPathComponent("w2vbert_mlx.safetensors"),
            stats: dir.appendingPathComponent("w2vbert_stats.safetensors"),
            verbose: verbose)
        self.repcodec = try RepCodec.fromPretrained(
            weights: dir.appendingPathComponent("semantic_codec_mlx.safetensors"), verbose: verbose)
        self.campplus = try CAMPPlus.fromPretrained(
            weights: dir.appendingPathComponent("campplus_mlx.safetensors"), verbose: verbose)
    }

    /// Raw conditioning from a reference waveform (before length regulation).
    /// - spkCondEmb: (1, T, 1024)  W2V-BERT hidden_states[17], normalized
    /// - style:      (1, 192)      CAMPPlus
    /// - sRef:       (1, T, 1024)  RepCodec continuous quantized embeddings
    /// - refMel:     (1, 80, Lp)   22.05 kHz reference mel
    public struct Raw {
        public var spkCondEmb: MLXArray
        public var style: MLXArray
        public var sRef: MLXArray
        public var refMel: MLXArray
    }

    public func encode(audioURL: URL, verbose: Bool = false) throws -> Raw {
        let audio16 = try AudioIO.loadAudio(audioURL, sampleRate: KaldiFbank.sr)   // 16k
        let audio22 = try AudioIO.loadAudio(audioURL, sampleRate: IndexTTSMel.sr)  // 22.05k

        // W2V-BERT semantic features -> RepCodec continuous codes.
        let spkCondEmb = w2v(audio16)                          // (1, T, 1024)
        let (sRef, _) = repcodec.quantize(spkCondEmb)          // (1, T, 1024)

        // 22 kHz reference mel for the CFM prompt.
        let refMel = IndexTTSMel.refMel(audio22)              // (1, 80, Lp)

        // CAMPPlus style: kaldi fbank (per-utterance mean subtraction) -> (1, 192).
        let fb = KaldiFbank.fbank(audio16)                    // (m, 80)
        let fbMS = fb - fb.mean(axis: 0, keepDims: true)
        let style = campplus(fbMS.expandedDimensions(axis: 0))  // (1, 192)

        eval(spkCondEmb, sRef, refMel, style)
        if verbose {
            log("ref: spkCondEmb \(spkCondEmb.shape) sRef \(sRef.shape) refMel \(refMel.shape) style \(style.shape)")
        }
        return Raw(spkCondEmb: spkCondEmb, style: style, sRef: sRef, refMel: refMel)
    }

    /// W2V-BERT features (1, T, 1024) for a separate emotion reference .wav.
    public func encodeEmotion(audioURL: URL) throws -> MLXArray {
        let audio16 = try AudioIO.loadAudio(audioURL, sampleRate: KaldiFbank.sr)
        let emb = w2v(audio16)
        eval(emb)
        return emb
    }

    private func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
}

extension IndexTTSv2 {
    /// Build `SpeakerConditioning` from a reference .wav, running the full
    /// torch-free preprocessing stack + S2Mel length regulator.
    public func makeSpeaker(audioURL: URL, using ref: ReferenceEncoder, verbose: Bool = false)
        throws -> SpeakerConditioning
    {
        let raw = try ref.encode(audioURL: audioURL, verbose: verbose)
        let lp = raw.refMel.dim(2)
        // prompt_condition = length_regulator(S_ref, ylens=ref_mel_len)  (1, Lp, 512)
        let promptCondition = s2mel.length_regulator(raw.sRef, targetLen: lp)
        eval(promptCondition)
        return SpeakerConditioning(
            spkCondEmb: raw.spkCondEmb, style: raw.style,
            promptCondition: promptCondition, refMel: raw.refMel)
    }
}
