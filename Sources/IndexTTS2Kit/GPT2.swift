import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

/// KV cache with chunked pre-allocation.
///
/// The backing buffers grow in fixed `step`-sized chunks, so a full
/// concatenation happens at most once every `step` tokens; in between, new keys
/// and values are written in place. This avoids the O(n²) per-step concat of a
/// naive cache. Mirrors the `mlx_lm` KVCache strategy.
final class KVCache {
    var keys: MLXArray?
    var values: MLXArray?
    var offset = 0
    let step = 256

    func updateAndFetch(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset
        let needed = prev + k.dim(2)

        if keys == nil || needed > keys!.dim(2) {
            let b = k.dim(0)
            let nKV = k.dim(1)
            let kHead = k.dim(3)
            let vHead = v.dim(3)
            let nSteps = (step + k.dim(2) - 1) / step
            let newK = MLXArray.zeros([b, nKV, nSteps * step, kHead], dtype: k.dtype)
            let newV = MLXArray.zeros([b, nKV, nSteps * step, vHead], dtype: v.dtype)

            if var kk = keys, var vv = values {
                // trim any unused (pre-allocated but unwritten) tail before growing
                if prev % step != 0 {
                    kk = kk[0 ..< prev, axis: 2]
                    vv = vv[0 ..< prev, axis: 2]
                }
                keys = concatenated([kk, newK], axis: 2)
                values = concatenated([vv, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
        }

        offset += k.dim(2)
        keys![prev ..< offset, axis: 2] = k
        values![prev ..< offset, axis: 2] = v
        return (keys![0 ..< offset, axis: 2], values![0 ..< offset, axis: 2])
    }
}

/// GPT-2 self attention with fused QKV projection.
final class GPT2Attention: Module {
    let nHead: Int
    let scale: Float
    // @ModuleInfo so `quantize`'s `update(modules:)` can swap in QuantizedLinear.
    @ModuleInfo var c_attn: Linear
    @ModuleInfo var c_proj: Linear

    init(nEmbd: Int, nHead: Int) {
        self.nHead = nHead
        let headDim = nEmbd / nHead
        self.scale = pow(Float(headDim), -0.5)
        self._c_attn.wrappedValue = Linear(nEmbd, 3 * nEmbd, bias: true)
        self._c_proj.wrappedValue = Linear(nEmbd, nEmbd, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let b = x.dim(0)
        let l = x.dim(1)
        let qkv = c_attn(x)
        let parts = split(qkv, parts: 3, axis: -1)
        var q = parts[0].reshaped([b, l, nHead, -1]).transposed(0, 2, 1, 3)
        var k = parts[1].reshaped([b, l, nHead, -1]).transposed(0, 2, 1, 3)
        var v = parts[2].reshaped([b, l, nHead, -1]).transposed(0, 2, 1, 3)

        if let cache {
            (k, v) = cache.updateAndFetch(k, v)
        }

        var o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        o = o.transposed(0, 2, 1, 3).reshaped([b, l, -1])
        return c_proj(o)
    }
}

final class GPT2MLP: Module {
    @ModuleInfo var c_fc: Linear
    @ModuleInfo var c_proj: Linear

    init(nEmbd: Int) {
        self._c_fc.wrappedValue = Linear(nEmbd, 4 * nEmbd)
        self._c_proj.wrappedValue = Linear(4 * nEmbd, nEmbd)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        c_proj(geluApproximate(c_fc(x)))
    }
}

final class GPT2Block: Module {
    let ln_1: LayerNorm
    let attn: GPT2Attention
    let ln_2: LayerNorm
    let mlp: GPT2MLP

    init(nEmbd: Int, nHead: Int, eps: Float) {
        self.ln_1 = LayerNorm(dimensions: nEmbd, eps: eps)
        self.attn = GPT2Attention(nEmbd: nEmbd, nHead: nHead)
        self.ln_2 = LayerNorm(dimensions: nEmbd, eps: eps)
        self.mlp = GPT2MLP(nEmbd: nEmbd)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let h = x + attn(ln_1(x), mask: mask, cache: cache)
        return h + mlp(ln_2(h))
    }
}

/// GPT-2 transformer stack. Inputs are embeddings directly (wte/wpe are identity
/// in IndexTTS — positions are added by the outer model).
final class GPT2Model: Module {
    let h: [GPT2Block]
    let ln_f: LayerNorm

    init(nEmbd: Int, nHead: Int, nLayer: Int, eps: Float = 1e-5) {
        self.h = (0 ..< nLayer).map { _ in GPT2Block(nEmbd: nEmbd, nHead: nHead, eps: eps) }
        self.ln_f = LayerNorm(dimensions: nEmbd, eps: eps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = inputs
        let l = hidden.dim(1)
        let mask: MLXArray? = l > 1 ? causalMask(l, dtype: hidden.dtype) : nil
        for (i, layer) in h.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[i])
        }
        return ln_f(hidden)
    }
}

/// Temperature + top-k sampling over a single logits vector. Returns the token id.
func sampleTopK(_ logits: MLXArray, temp: Float = 0.8, topK: Int = 30) -> Int {
    let vocab = logits.size
    let flat = logits.reshaped([vocab]) * (1.0 / temp)
    let ascending = sorted(flat, axis: -1)
    let threshold = ascending[vocab - topK]
    let masked = MLX.which(flat .>= threshold, flat, MLXArray(-Float.infinity))
    let token = categorical(masked)
    return token.item(Int.self)
}
