import Foundation
import MLX
import MLXFast
import MLXNN

/// Standard multi-head attention (used by the perceiver resampler).
final class MultiHeadAttention: Module {
    let nHead: Int
    let headDim: Int
    let scale: Float

    let linear_q: Linear
    let linear_k: Linear
    let linear_v: Linear
    let linear_out: Linear

    init(nHead: Int, nFeat: Int, bias: Bool = true, headDim: Int? = nil) {
        self.nHead = nHead
        self.headDim = headDim ?? (nFeat / nHead)
        self.scale = pow(Float(self.headDim), -0.5)
        let inner = self.headDim * nHead
        self.linear_q = Linear(nFeat, inner, bias: bias)
        self.linear_k = Linear(nFeat, inner, bias: bias)
        self.linear_v = Linear(nFeat, inner, bias: bias)
        self.linear_out = Linear(inner, nFeat, bias: bias)
        super.init()
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray? = nil)
        -> MLXArray
    {
        let qp = linear_q(q)
        let kp = linear_k(k)
        let vp = linear_v(v)

        let batch = qp.dim(0)
        let qSeq = qp.dim(1)
        let kSeq = kp.dim(1)

        let qh = qp.reshaped([batch, qSeq, nHead, headDim]).transposed(0, 2, 1, 3)
        let kh = kp.reshaped([batch, kSeq, nHead, headDim]).transposed(0, 2, 1, 3)
        let vh = vp.reshaped([batch, kSeq, nHead, headDim]).transposed(0, 2, 1, 3)

        var o = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh, scale: scale, mask: mask)
        o = o.transposed(0, 2, 1, 3).reshaped([batch, qSeq, -1])
        return linear_out(o)
    }
}

/// Relative-position multi-head attention (Conformer self-attention).
final class RelPositionMultiHeadAttention: Module {
    let nHead: Int
    let headDim: Int
    let scale: Float

    let linear_q: Linear
    let linear_k: Linear
    let linear_v: Linear
    let linear_out: Linear
    let linear_pos: Linear

    var pos_bias_u: MLXArray
    var pos_bias_v: MLXArray

    init(nHead: Int, nFeat: Int, bias: Bool = true) {
        self.nHead = nHead
        self.headDim = nFeat / nHead
        self.scale = pow(Float(self.headDim), -0.5)
        self.linear_q = Linear(nFeat, nFeat, bias: bias)
        self.linear_k = Linear(nFeat, nFeat, bias: bias)
        self.linear_v = Linear(nFeat, nFeat, bias: bias)
        self.linear_out = Linear(nFeat, nFeat, bias: bias)
        self.linear_pos = Linear(nFeat, nFeat, bias: false)
        self.pos_bias_u = MLXArray.zeros([nHead, headDim])
        self.pos_bias_v = MLXArray.zeros([nHead, headDim])
        super.init()
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, posEmb: MLXArray)
        -> MLXArray
    {
        let qp = linear_q(q)
        let kp = linear_k(k)
        let vp = linear_v(v)
        let p = linear_pos(posEmb)

        let batch = qp.dim(0)
        let qSeq = qp.dim(1)
        let kSeq = kp.dim(1)
        let posLen = p.dim(1)

        let qr = qp.reshaped([batch, qSeq, nHead, headDim])
        let qu = (qr + pos_bias_u).transposed(0, 2, 1, 3)
        let qv = (qr + pos_bias_v).transposed(0, 2, 1, 3)

        let kh = kp.reshaped([batch, kSeq, nHead, headDim]).transposed(0, 2, 1, 3)
        let vh = vp.reshaped([batch, kSeq, nHead, headDim]).transposed(0, 2, 1, 3)
        let ph = p.reshaped([batch, posLen, nHead, headDim]).transposed(0, 2, 1, 3)

        // relative-position bias used as the additive attention mask
        var matrixBD = matmul(qv, ph.swappedAxes(-2, -1))
        matrixBD = matrixBD * scale

        var o = MLXFast.scaledDotProductAttention(
            queries: qu, keys: kh, values: vh, scale: scale, mask: matrixBD)
        o = o.transposed(0, 2, 1, 3).reshaped([batch, qSeq, -1])
        return linear_out(o)
    }
}

/// Sinusoidal relative positional encoding (a non-trained buffer).
final class RelPositionalEncoding: Module {
    let dModel: Int
    var maxLen: Int
    let scaleInput: Float

    // buffer (excluded from checkpoint via sanitize dropping "pos_enc")
    var pe: MLXArray

    init(dModel: Int, maxLen: Int = 2048, scaleInput: Bool = true) {
        precondition(dModel % 2 == 0)
        self.dModel = dModel
        self.maxLen = maxLen
        self.scaleInput = scaleInput ? Float(sqrt(Double(dModel))) : 1.0
        self.pe = RelPositionalEncoding.makePE(dModel: dModel, maxLen: maxLen)
        super.init()
    }

    static func makePE(dModel: Int, maxLen: Int) -> MLXArray {
        var vals = [Float](repeating: 0, count: maxLen * dModel)
        for pos in 0 ..< maxLen {
            for i in stride(from: 0, to: dModel, by: 2) {
                let divTerm = exp(Double(i) * -(log(10000.0) / Double(dModel)))
                vals[pos * dModel + i] = Float(sin(Double(pos) * divTerm))
                if i + 1 < dModel {
                    vals[pos * dModel + i + 1] = Float(cos(Double(pos) * divTerm))
                }
            }
        }
        return MLXArray(vals, [1, maxLen, dModel])
    }

    /// Returns `(x * scale, pos_emb)`.
    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        let inputLen = x.dim(1) + offset
        if inputLen > maxLen {
            maxLen = inputLen + 1
            pe = RelPositionalEncoding.makePE(dModel: dModel, maxLen: maxLen)
        }
        let scaled = x * scaleInput
        let posEmb = pe[0..., offset ..< (offset + x.dim(1)), 0...].asType(x.dtype)
        return (scaled, posEmb)
    }
}

/// Learned absolute position encoding (mel/text positions in the GPT path).
final class LearnedPositionEncoding: Module {
    let emb: Embedding

    init(seqLen: Int, modelDim: Int) {
        self.emb = Embedding(embeddingCount: seqLen, dimensions: modelDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let idx = MLXArray(Array(Int32(offset) ..< Int32(offset + x.dim(1))))
        return emb(idx)
    }
}
