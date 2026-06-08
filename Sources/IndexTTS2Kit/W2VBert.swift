import Foundation
import MLX
import MLXNN

// MARK: - W2V-BERT 2.0 (facebook/w2v-bert-2.0) -> spk_cond_emb
//
// Torch-free port of the SeamlessM4T feature extractor + Wav2Vec2BertModel
// conformer encoder, producing `spk_cond_emb` = hidden_states[17] normalized by
// `wav2vec2bert_stats` (mean/std). Only 17 of 24 layers are needed.
//
// All ops run in NLC (B, T, C). No attention mask / padding (single utterance),
// so the conv/attention masking paths are omitted. `position_embeddings_type` is
// "relative_key" (Shaw-style distance embedding added to attention scores).

private let w2vHidden = 1024
private let w2vHeads = 16
private let w2vHead = 64
private let w2vInter = 4096
private let w2vLeftMax = 64
private let w2vRightMax = 8
private let w2vEps: Float = 1e-5

func swish(_ x: MLXArray) -> MLXArray { x * MLX.sigmoid(x) }

/// SeamlessM4T feature extractor: 80 log-mel (scaled, povey, kaldi mel),
/// per-mel-bin normalization, then stride-2 frame stacking -> (1, T, 160).
public enum W2VFeatureExtractor {
    public static func extract(_ wave: MLXArray, stride: Int = 2) -> MLXArray {
        // (m, 80) seamless fbank. HF scales the wave by 2^15 (power x 2^30), which
        // only adds a constant 30*ln2 to every log-mel value — removed by the
        // per-bin normalization below. We skip the scale (keeping values small for
        // float32 precision) and divide the mel_floor by 2^30 to stay identical.
        let floor: Float = 1.192092955078125e-07 / Float(1 << 30)
        var f = KaldiFbank.fbank(wave, scale: 1.0, melFloor: floor)
        // per-mel-bin normalization over time (ddof=1).
        let m = f.dim(0)
        let mean = f.mean(axis: 0, keepDims: true)
        let diff = f - mean
        let varU = (diff * diff).sum(axis: 0, keepDims: true) / Float(m - 1)
        f = diff / MLX.sqrt(varU + MLXArray(Float(1e-7)))
        // trim to even, stack pairs -> (m/2, 160)
        let rem = m % stride
        if rem != 0 { f = f[0 ..< (m - rem), 0...] }
        let t = (m - rem) / stride
        f = f.reshaped([t, 80 * stride])
        return f.expandedDimensions(axis: 0)  // (1, t, 160)
    }
}

final class W2VFeedForward: Module {
    let intermediate_dense: Linear
    let output_dense: Linear
    override init() {
        self.intermediate_dense = Linear(w2vHidden, w2vInter)
        self.output_dense = Linear(w2vInter, w2vHidden)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        output_dense(swish(intermediate_dense(x)))
    }
}

final class W2VConvModule: Module {
    let layer_norm: LayerNorm
    let pointwise_conv1: Conv1d        // 1024 -> 2048, k1, no bias
    let depthwise_conv: Conv1d         // 1024 -> 1024, k31, groups 1024, no bias
    let depthwise_layer_norm: LayerNorm
    let pointwise_conv2: Conv1d        // 1024 -> 1024, k1, no bias
    let kernel = 31

    override init() {
        self.layer_norm = LayerNorm(dimensions: w2vHidden, eps: w2vEps)
        self.pointwise_conv1 = Conv1d(inputChannels: w2vHidden, outputChannels: 2 * w2vHidden, kernelSize: 1, bias: false)
        self.depthwise_conv = Conv1d(inputChannels: w2vHidden, outputChannels: w2vHidden, kernelSize: 31,
                                     groups: w2vHidden, bias: false)
        self.depthwise_layer_norm = LayerNorm(dimensions: w2vHidden, eps: w2vEps)
        self.pointwise_conv2 = Conv1d(inputChannels: w2vHidden, outputChannels: w2vHidden, kernelSize: 1, bias: false)
        super.init()
    }
    /// `x`: (B, T, 1024).
    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = layer_norm(x0)
        x = pointwise_conv1(x)                       // (B,T,2048)
        let a = x[0..., 0..., 0 ..< w2vHidden]
        let b = x[0..., 0..., w2vHidden ..< (2 * w2vHidden)]
        x = a * MLX.sigmoid(b)                        // GLU -> (B,T,1024)
        // causal left pad (kernel-1) along time.
        x = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((kernel - 1, 0)), IntOrPair((0, 0))])
        x = depthwise_conv(x)                         // (B,T,1024)
        x = depthwise_layer_norm(x)
        x = swish(x)
        x = pointwise_conv2(x)
        return x
    }
}

final class W2VAttention: Module {
    let linear_q: Linear
    let linear_k: Linear
    let linear_v: Linear
    let linear_out: Linear
    let distance_embedding: Embedding

    override init() {
        self.linear_q = Linear(w2vHidden, w2vHidden)
        self.linear_k = Linear(w2vHidden, w2vHidden)
        self.linear_v = Linear(w2vHidden, w2vHidden)
        self.linear_out = Linear(w2vHidden, w2vHidden)
        self.distance_embedding = Embedding(embeddingCount: w2vLeftMax + w2vRightMax + 1, dimensions: w2vHead)
        super.init()
    }
    /// `x`: (B, T, 1024).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        func heads(_ l: Linear) -> MLXArray {
            l(x).reshaped([B, T, w2vHeads, w2vHead]).transposed(0, 2, 1, 3)  // (B,H,T,D)
        }
        let q = heads(linear_q), k = heads(linear_k), v = heads(linear_v)
        let scale = 1.0 / sqrt(Float(w2vHead))
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale       // (B,H,T,T)

        // relative_key: distance embedding term.
        let l = MLXArray(Array(0 ..< Int32(T))).reshaped([T, 1])
        let r = MLXArray(Array(0 ..< Int32(T))).reshaped([1, T])
        var dist = r - l
        dist = MLX.clip(dist, min: MLXArray(Int32(-w2vLeftMax)), max: MLXArray(Int32(w2vRightMax)))
        dist = dist + Int32(w2vLeftMax)
        let pos = distance_embedding(dist)                            // (T,T,D)
        let rel = einsum("bhld,lrd->bhlr", q, pos)                    // (B,H,T,T)
        scores = scores + rel * scale

        let probs = MLX.softmax(scores, axis: -1)
        var out = matmul(probs, v)                                    // (B,H,T,D)
        out = out.transposed(0, 2, 1, 3).reshaped([B, T, w2vHeads * w2vHead])
        return linear_out(out)
    }
}

final class W2VEncoderLayer: Module {
    let ffn1_layer_norm: LayerNorm
    let ffn1: W2VFeedForward
    let self_attn_layer_norm: LayerNorm
    let self_attn: W2VAttention
    let conv_module: W2VConvModule
    let ffn2_layer_norm: LayerNorm
    let ffn2: W2VFeedForward
    let final_layer_norm: LayerNorm

    override init() {
        self.ffn1_layer_norm = LayerNorm(dimensions: w2vHidden, eps: w2vEps)
        self.ffn1 = W2VFeedForward()
        self.self_attn_layer_norm = LayerNorm(dimensions: w2vHidden, eps: w2vEps)
        self.self_attn = W2VAttention()
        self.conv_module = W2VConvModule()
        self.ffn2_layer_norm = LayerNorm(dimensions: w2vHidden, eps: w2vEps)
        self.ffn2 = W2VFeedForward()
        self.final_layer_norm = LayerNorm(dimensions: w2vHidden, eps: w2vEps)
        super.init()
    }
    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = x0
        x = ffn1(ffn1_layer_norm(x)) * 0.5 + x
        x = self_attn(self_attn_layer_norm(x)) + x
        x = conv_module(x) + x
        x = ffn2(ffn2_layer_norm(x)) * 0.5 + x
        return final_layer_norm(x)
    }
}

final class W2VEncoder: Module {
    let layers: [W2VEncoderLayer]
    init(numLayers: Int) {
        self.layers = (0 ..< numLayers).map { _ in W2VEncoderLayer() }
        super.init()
    }
}

final class W2VFeatureProjection: Module {
    let layer_norm: LayerNorm
    let projection: Linear
    override init() {
        self.layer_norm = LayerNorm(dimensions: 160, eps: w2vEps)
        self.projection = Linear(160, w2vHidden)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { projection(layer_norm(x)) }
}

public final class W2VBert: Module {
    let feature_projection: W2VFeatureProjection
    let encoder: W2VEncoder

    /// `numLayers` = number of conformer layers to run; hidden_states[17] (the
    /// IndexTTS feature) is the output after running layers 0..16, so 17.
    public init(numLayers: Int = 17) {
        self.feature_projection = W2VFeatureProjection()
        self.encoder = W2VEncoder(numLayers: numLayers)
        super.init()
    }

    /// `inputFeatures`: (B, T, 160). Returns hidden_states[17] (B, T, 1024).
    public func callAsFunction(_ inputFeatures: MLXArray) -> MLXArray {
        var x = feature_projection(inputFeatures)
        for layer in encoder.layers { x = layer(x) }
        return x
    }

    public static func fromPretrained(weights url: URL, verbose: Bool = false) throws -> W2VBert {
        let m = W2VBert()
        try loadWeights(into: m, from: url, label: "w2vbert", verbose: verbose)
        eval(m)
        return m
    }
}

/// Full W2V-BERT speaker-feature extractor: wave (16k) -> spk_cond_emb (1, T, 1024).
public final class W2VSpeakerEncoder {
    let model: W2VBert
    let mean: MLXArray   // (1024,)
    let std: MLXArray    // (1024,)

    public init(weights: URL, stats: URL, verbose: Bool = false) throws {
        self.model = try W2VBert.fromPretrained(weights: weights, verbose: verbose)
        let s = try loadArrays(url: stats)
        guard let mean = s["mean"], let varr = s["var"] else {
            throw IndexTTS2Error.missingWeights("w2vbert_stats mean/var")
        }
        self.mean = mean
        self.std = MLX.sqrt(varr)
    }

    /// `wave`: (samples,) 16 kHz. Returns spk_cond_emb (1, T, 1024).
    public func callAsFunction(_ wave: MLXArray) -> MLXArray {
        let feat = W2VFeatureExtractor.extract(wave)
        let h = model(feat)
        return (h - mean) / std
    }
}
