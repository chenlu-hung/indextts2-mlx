import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Diffusion Transformer (DiT) — CFM estimator
//
// Port of models/s2mel/dit.py. hidden_dim 512, 8 heads, depth 13, RoPE
// attention, AdaptiveLayerNorm (RMSNorm + projected scale/shift), U-ViT skip
// connections, long skip, and a WaveNet final layer. Non-causal full attention
// at inference (single utterance, no padding) so attention masks are omitted.

private let kHidden = 512
private let kHeads = 8
private let kHeadDim = 64
private let kDepth = 13
private let kInChannels = 80
private let kStyleDim = 192
private let kIntermediate = 1536
private let kBlockSize = 16384

/// Sinusoidal timestep embedder. `freqs` is stored (present in checkpoint).
final class TimestepEmbedder: Module {
    var freqs: MLXArray
    let linear1: Linear
    let linear2: Linear
    let scale: Float = 1000

    init(hiddenSize: Int, freqDim: Int = 256) {
        let half = freqDim / 2
        let maxPeriod = 10000.0
        let f = (0 ..< half).map { Float(exp(-log(maxPeriod) * Double($0) / Double(half))) }
        self.freqs = MLXArray(f)
        self.linear1 = Linear(freqDim, hiddenSize)
        self.linear2 = Linear(hiddenSize, hiddenSize)
        super.init()
    }

    /// `t`: (B,). Returns (B, hiddenSize).
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let args = (t * scale).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        let emb = concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
        return linear2(silu(linear1(emb)))
    }
}

/// Adaptive LayerNorm: RMSNorm then projected per-channel scale/shift from `c`.
final class AdaptiveLayerNorm: Module {
    let dModel: Int
    let project_layer: Linear
    let norm: RMSNorm

    init(_ dModel: Int, eps: Float = 1e-5) {
        self.dModel = dModel
        self.project_layer = Linear(dModel, 2 * dModel)
        self.norm = RMSNorm(dimensions: dModel, eps: eps)
        super.init()
    }

    /// `x`: (B, L, d). `emb`: (B, 1, d) or nil (→ plain norm).
    func callAsFunction(_ x: MLXArray, _ emb: MLXArray?) -> MLXArray {
        guard let emb else { return norm(x) }
        let proj = project_layer(emb)               // (B,1,2d)
        let weight = proj[0..., 0..., 0 ..< dModel]
        let bias = proj[0..., 0..., dModel...]
        return weight * norm(x) + bias
    }
}

// MARK: RoPE

func makeFreqsCis(headDim: Int, maxSeq: Int, base: Float = 10000) -> MLXArray {
    let half = headDim / 2
    let inv = (0 ..< half).map { Float(1.0 / pow(Double(base), Double(2 * $0) / Double(headDim))) }
    let invFreq = MLXArray(inv)                                   // (half,)
    let t = MLXArray(Array(0 ..< Int32(maxSeq))).asType(.float32)  // (maxSeq,)
    let freqs = outer(t, invFreq)                                 // (maxSeq, half)
    return stacked([MLX.cos(freqs), MLX.sin(freqs)], axis: -1)    // (maxSeq, half, 2)
}

/// `x`: (B, seq, heads, headDim). `freqsCis`: (seq, headDim/2, 2).
func applyRotary(_ x: MLXArray, _ freqsCis: MLXArray) -> MLXArray {
    let s = x.shape
    let xr = x.reshaped(Array(s[0..<(s.count - 1)]) + [-1, 2])   // (...,half,2)
    let cos = freqsCis[0..., 0..., 0].expandedDimensions(axes: [0, 2])  // (1,seq,1,half)
    let sin = freqsCis[0..., 0..., 1].expandedDimensions(axes: [0, 2])
    let xReal = xr[.ellipsis, 0]
    let xImag = xr[.ellipsis, 1]
    let outReal = xReal * cos - xImag * sin
    let outImag = xImag * cos + xReal * sin
    return stacked([outReal, outImag], axis: -1).reshaped(s)
}

final class DiTAttention: Module {
    let wqkv: Linear
    let wo: Linear
    let scale: Float

    override init() {
        let total = (kHeads + 2 * kHeads) * kHeadDim
        self.wqkv = Linear(kHidden, total, bias: false)
        self.wo = Linear(kHeadDim * kHeads, kHidden, bias: false)
        self.scale = 1.0 / Float(kHeadDim).squareRoot()
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ freqsCis: MLXArray) -> MLXArray {
        let b = x.dim(0), l = x.dim(1)
        let kv = kHeads * kHeadDim
        let qkv = wqkv(x)
        var q = qkv[0..., 0..., 0 ..< kv].reshaped([b, l, kHeads, kHeadDim])
        var k = qkv[0..., 0..., kv ..< (2 * kv)].reshaped([b, l, kHeads, kHeadDim])
        let v = qkv[0..., 0..., (2 * kv)...].reshaped([b, l, kHeads, kHeadDim])
        q = applyRotary(q, freqsCis)
        k = applyRotary(k, freqsCis)
        let qh = q.transposed(0, 2, 1, 3)
        let kh = k.transposed(0, 2, 1, 3)
        let vh = v.transposed(0, 2, 1, 3)
        var o = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh, scale: scale, mask: nil)
        o = o.transposed(0, 2, 1, 3).reshaped([b, l, -1])
        return wo(o)
    }
}

final class DiTFeedForward: Module {
    let w1: Linear
    let w2: Linear
    let w3: Linear
    override init() {
        self.w1 = Linear(kHidden, kIntermediate, bias: false)
        self.w2 = Linear(kIntermediate, kHidden, bias: false)
        self.w3 = Linear(kHidden, kIntermediate, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

final class DiTBlock: Module {
    let attention: DiTAttention
    let feed_forward: DiTFeedForward
    let attention_norm: AdaptiveLayerNorm
    let ffn_norm: AdaptiveLayerNorm
    let skip_in_linear: Linear

    override init() {
        self.attention = DiTAttention()
        self.feed_forward = DiTFeedForward()
        self.attention_norm = AdaptiveLayerNorm(kHidden)
        self.ffn_norm = AdaptiveLayerNorm(kHidden)
        self.skip_in_linear = Linear(kHidden * 2, kHidden)
        super.init()
    }

    func callAsFunction(_ x0: MLXArray, _ c: MLXArray, _ freqsCis: MLXArray, _ skipInX: MLXArray?)
        -> MLXArray
    {
        var x = x0
        if let skipInX {
            x = skip_in_linear(concatenated([x, skipInX], axis: -1))
        }
        let h = x + attention(attention_norm(x, c), freqsCis)
        return h + feed_forward(ffn_norm(h, c))
    }
}

final class DiTTransformer: Module {
    let layers: [DiTBlock]
    let norm: AdaptiveLayerNorm
    var freqs_cis: MLXArray  // computed buffer (not in checkpoint)
    let emitSkip: Set<Int>
    let receiveSkip: Set<Int>

    override init() {
        self.layers = (0 ..< kDepth).map { _ in DiTBlock() }
        self.norm = AdaptiveLayerNorm(kHidden)
        self.freqs_cis = makeFreqsCis(headDim: kHeadDim, maxSeq: kBlockSize)
        self.emitSkip = Set((0 ..< kDepth).filter { $0 < kDepth / 2 })
        self.receiveSkip = Set((0 ..< kDepth).filter { $0 > kDepth / 2 })
        super.init()
    }

    /// `x`: (B, L, d). `c`: (B, 1, d).
    func callAsFunction(_ x0: MLXArray, _ c: MLXArray, _ inputPos: MLXArray) -> MLXArray {
        var x = x0
        let fc = freqs_cis[inputPos]
        var skipStack: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            let skipInX = receiveSkip.contains(i) ? skipStack.removeLast() : nil
            x = layer(x, c, fc, skipInX)
            if emitSkip.contains(i) { skipStack.append(x) }
        }
        return norm(x, c)
    }
}

/// SiLU + Linear modulation (matches `adaLN_modulation.layers.1`).
final class AdaLNModulation: Module {
    let layers: [Module]
    init(_ hiddenSize: Int) {
        self.layers = [SiLU(), Linear(hiddenSize, 2 * hiddenSize)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (layers[1] as! Linear)(silu(x))
    }
}

final class FinalLayer: Module {
    let hiddenSize: Int
    let linear: Linear
    let adaLN_modulation: AdaLNModulation

    init(_ hiddenSize: Int, outChannels: Int) {
        self.hiddenSize = hiddenSize
        self.linear = Linear(hiddenSize, outChannels)
        self.adaLN_modulation = AdaLNModulation(hiddenSize)
        super.init()
    }

    private func layerNorm(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let v = x.variance(axis: -1, keepDims: true)
        return (x - mean) / MLX.sqrt(v + 1e-6)
    }

    /// `x`: (B, L, hidden). `c`: (B, hidden).
    func callAsFunction(_ x: MLXArray, _ c: MLXArray) -> MLXArray {
        let mod = adaLN_modulation(c)                       // (B, 2*hidden)
        let shift = mod[0..., 0 ..< hiddenSize].expandedDimensions(axis: 1)
        let scale = mod[0..., hiddenSize...].expandedDimensions(axis: 1)
        return linear(layerNorm(x) * (scale + 1) + shift)
    }
}

final class DiT: Module {
    /// Internal compute precision. Set to `.bfloat16` (via `castParameters`) for
    /// ~2× speedup; inputs are cast in and the velocity cast back to fp32 on exit
    /// so the CFM Euler loop stays fp32.
    var computeDType: DType = .float32

    let x_embedder: Linear
    let cond_projection: Linear
    let cond_embedder: Embedding           // discrete content (unused at inference)
    let content_mask_embedder: Embedding   // unused at inference
    let t_embedder: TimestepEmbedder
    let t_embedder2: TimestepEmbedder
    let cond_x_merge_linear: Linear
    let skip_linear: Linear
    let transformer: DiTTransformer
    let conv1: Linear
    let conv2: Conv1d
    let wavenet: WN
    let final_layer: FinalLayer
    let res_projection: Linear

    override init() {
        self.x_embedder = Linear(kInChannels, kHidden)
        self.cond_projection = Linear(kHidden, kHidden)  // content_dim 512 -> hidden 512
        self.cond_embedder = Embedding(embeddingCount: 1024, dimensions: kHidden)
        self.content_mask_embedder = Embedding(embeddingCount: 1, dimensions: kHidden)
        self.t_embedder = TimestepEmbedder(hiddenSize: kHidden)
        self.t_embedder2 = TimestepEmbedder(hiddenSize: kHidden)
        let mergeDim = kHidden + kInChannels * 2 + kStyleDim  // 512 + 160 + 192 = 864
        self.cond_x_merge_linear = Linear(mergeDim, kHidden)
        self.skip_linear = Linear(kHidden + kInChannels, kHidden)  // 592 -> 512
        self.transformer = DiTTransformer()
        self.conv1 = Linear(kHidden, kHidden)
        self.conv2 = Conv1d(inputChannels: kHidden, outputChannels: kInChannels, kernelSize: 1)
        self.wavenet = WN(hiddenChannels: kHidden, kernelSize: 5, nLayers: 8, ginChannels: kHidden)
        self.final_layer = FinalLayer(kHidden, outChannels: kHidden)
        self.res_projection = Linear(kHidden, kHidden)
        super.init()
    }

    /// `x`, `promptX`: (B, 80, T) NCL. `t`: (B,). `style`: (B, 192). `cond`: (B, T, 512).
    /// Returns (B, 80, T) NCL.
    func callAsFunction(
        _ x0: MLXArray, _ promptX0: MLXArray, _ t0: MLXArray, _ style0: MLXArray, _ cond00: MLXArray
    ) -> MLXArray {
        let dt = computeDType
        let x = x0.asType(dt)
        let promptX = promptX0.asType(dt)
        let t = t0.asType(dt)
        let style = style0.asType(dt)
        let cond0 = cond00.asType(dt)
        let b = x.dim(0), t_len = x.dim(2)
        let t1 = t_embedder(t)                       // (B, 512)
        let cond = cond_projection(cond0)            // (B, T, 512)
        let xt = x.transposed(0, 2, 1)               // (B, T, 80)
        let promptXt = promptX.transposed(0, 2, 1)   // (B, T, 80)

        var xIn = concatenated([xt, promptXt, cond], axis: -1)  // (B, T, 672)
        let styleExp = MLX.broadcast(
            style.expandedDimensions(axis: 1), to: [b, t_len, kStyleDim])
        xIn = concatenated([xIn, styleExp], axis: -1)           // (B, T, 864)
        xIn = cond_x_merge_linear(xIn)                          // (B, T, 512)

        let inputPos = MLXArray(Array(0 ..< Int32(t_len)))
        var xRes = transformer(xIn, t1.expandedDimensions(axis: 1), inputPos)  // (B, T, 512)

        // long skip connection
        xRes = skip_linear(concatenated([xRes, xt], axis: -1))  // (B, T, 512)

        // WaveNet final layer
        var xOut = conv1(xRes)                       // (B, T, 512) NLC
        xOut = xOut.transposed(0, 2, 1)              // (B, 512, T) NCL
        let t2 = t_embedder2(t)                      // (B, 512)
        let xMaskWN = MLXArray.ones([b, 1, t_len], dtype: dt)
        var wnOut = wavenet(xOut, xMaskWN, g: t2.expandedDimensions(axis: 2))  // (B,512,T) NCL
        wnOut = wnOut.transposed(0, 2, 1)            // (B, T, 512) NLC
        xOut = wnOut + res_projection(xRes)          // (B, T, 512)
        xOut = final_layer(xOut, t1)                 // (B, T, 512)
        xOut = conv2(xOut)                           // (B, T, 80)
        return xOut.transposed(0, 2, 1).asType(.float32)  // (B, 80, T) NCL
    }
}
