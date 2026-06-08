import Foundation
import MLX
import MLXNN

// MARK: - RepCodec semantic codec (amphion/MaskGCT)  -> S_ref
//
// Torch-free port of the `quantize()` path of `repcodec_model.py` (encoder +
// FactorizedVQ). Consumes W2V-BERT semantic features (B, T, 1024) and produces
// the continuous quantized embedding `S_ref` (B, T, 1024) fed to the S2Mel
// length regulator. Decoder is unused and not loaded.
//
// All ops run in NLC (B, T, C) — MLX Conv1d native — matching PyTorch math.

/// ConvNeXt block (Vocos variant), NLC.
final class ConvNeXtBlock: Module {
    let dwconv: Conv1d           // depthwise k7
    let norm: LayerNorm
    let pwconv1: Linear
    let pwconv2: Linear
    var gamma: MLXArray

    init(dim: Int, intermediate: Int) {
        self.dwconv = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7,
                             padding: 3, groups: dim, bias: true)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6)
        self.pwconv1 = Linear(dim, intermediate)
        self.pwconv2 = Linear(intermediate, dim)
        self.gamma = MLXArray.ones([dim])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = dwconv(x)            // (B,T,dim)
        h = norm(h)
        h = pwconv1(h)
        h = gelu(h)
        h = pwconv2(h)
        h = gamma * h
        return x + h
    }
}

/// Vocos backbone: embed conv + LN + 12 ConvNeXt blocks + final LN. NLC in/out.
final class VocosBackbone: Module {
    let embed: Conv1d
    let norm: LayerNorm
    let convnext: [ConvNeXtBlock]
    let final_layer_norm: LayerNorm

    init(inputChannels: Int, dim: Int, intermediate: Int, numLayers: Int) {
        self.embed = Conv1d(inputChannels: inputChannels, outputChannels: dim, kernelSize: 7, padding: 3, bias: true)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6)
        self.convnext = (0 ..< numLayers).map { _ in ConvNeXtBlock(dim: dim, intermediate: intermediate) }
        self.final_layer_norm = LayerNorm(dimensions: dim, eps: 1e-6)
        super.init()
    }
    /// `x`: (B, T, inputChannels). Returns (B, T, dim).
    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = embed(x0)
        x = norm(x)
        for blk in convnext { x = blk(x) }
        return final_layer_norm(x)
    }
}

/// FactorizedVectorQuantize — `decode_latents` (cosine-nearest) + projections.
final class FactorizedVQ: Module {
    let in_project: Conv1d       // 1x1: hidden -> codebook_dim
    let out_project: Conv1d      // 1x1: codebook_dim -> hidden
    let codebook: Embedding

    init(hidden: Int, codebookSize: Int, codebookDim: Int) {
        self.in_project = Conv1d(inputChannels: hidden, outputChannels: codebookDim, kernelSize: 1, bias: true)
        self.out_project = Conv1d(inputChannels: codebookDim, outputChannels: hidden, kernelSize: 1, bias: true)
        self.codebook = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
        super.init()
    }

    static func l2normalize(_ x: MLXArray, eps: Float = 1e-12) -> MLXArray {
        let n = MLX.sqrt((x * x).sum(axis: -1, keepDims: true))
        return x / MLX.maximum(n, MLXArray(eps))
    }

    /// `z`: (B, T, hidden). Returns (S_ref (B,T,hidden), indices (B,T)).
    func quantize(_ z: MLXArray) -> (MLXArray, MLXArray) {
        let B = z.dim(0), T = z.dim(1)
        let zE = in_project(z)                       // (B,T,codebook_dim)
        let D = zE.dim(2)
        let enc = FactorizedVQ.l2normalize(zE.reshaped([B * T, D]))
        let cb = codebook.weight                     // (codebookSize, D)
        let cbn = FactorizedVQ.l2normalize(cb)
        let sims = matmul(enc, cbn.transposed())     // (B*T, codebookSize)
        let idx = argMax(sims, axis: 1)              // (B*T,)
        let zq = cb[idx].reshaped([B, T, D])         // raw codebook lookup
        let sRef = out_project(zq)                   // (B,T,hidden)
        return (sRef, idx.reshaped([B, T]))
    }
}

final class ResidualVQ: Module {
    let quantizers: [FactorizedVQ]
    init(hidden: Int, numQuantizers: Int, codebookSize: Int, codebookDim: Int) {
        self.quantizers = (0 ..< numQuantizers).map { _ in
            FactorizedVQ(hidden: hidden, codebookSize: codebookSize, codebookDim: codebookDim)
        }
        super.init()
    }
}

public final class RepCodec: Module {
    let encoder: [Module]        // [VocosBackbone, Linear(vocos_dim -> hidden)]
    let quantizer: ResidualVQ

    public init(hidden: Int = 1024, vocosDim: Int = 384, vocosIntermediate: Int = 2048,
                vocosLayers: Int = 12, codebookSize: Int = 8192, codebookDim: Int = 8,
                numQuantizers: Int = 1) {
        self.encoder = [
            VocosBackbone(inputChannels: hidden, dim: vocosDim,
                          intermediate: vocosIntermediate, numLayers: vocosLayers),
            Linear(vocosDim, hidden),
        ]
        self.quantizer = ResidualVQ(hidden: hidden, numQuantizers: numQuantizers,
                                    codebookSize: codebookSize, codebookDim: codebookDim)
        super.init()
    }

    /// `x`: W2V-BERT features (B, T, 1024). Returns (S_ref (B,T,1024), indices (B,T)).
    public func quantize(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var z = (encoder[0] as! VocosBackbone)(x)
        z = (encoder[1] as! Linear)(z)               // (B,T,hidden)
        return quantizer.quantizers[0].quantize(z)
    }

    public static func fromPretrained(weights url: URL, verbose: Bool = false) throws -> RepCodec {
        let m = RepCodec()
        try loadWeights(into: m, from: url, label: "semantic_codec", verbose: verbose)
        eval(m)
        return m
    }
}
