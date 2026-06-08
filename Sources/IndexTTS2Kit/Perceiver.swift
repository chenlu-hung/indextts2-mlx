import Foundation
import MLX
import MLXNN

/// Gated-GELU feed forward used by the perceiver resampler.
final class PerceiverFeedForward: Module {
    let w_1: Linear
    let w_2: Linear

    init(dim: Int, dFF: Int, useBias: Bool = true) {
        self.w_1 = Linear(dim, dFF * 2, bias: useBias)
        self.w_2 = Linear(dFF, dim, bias: useBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(w_1(x), parts: 2, axis: -1)
        let value = parts[0]
        let gate = parts[1]
        return w_2(gelu(gate) * value)
    }
}

/// Perceiver resampler: cross-attends learned latents to the conformer output.
final class PerceiverResampler: Module {
    let proj_context: UnaryLayer
    var latents: MLXArray
    let layers: [[Module]]
    let norm: RMSNorm

    init(
        nDim: Int, nDepth: Int = 2, nDimContext: Int? = nil, nLatents: Int = 32,
        nDimHead: Int = 64, nHeads: Int = 8, nFFMult: Int = 4
    ) {
        let ctx = nDimContext ?? nDim
        self.proj_context = ctx != nDim ? Linear(ctx, nDim) : Identity()
        self.latents = MLXArray.zeros([nLatents, nDim])
        let dFF = (nDim * nFFMult * 2) / 3
        self.layers = (0 ..< nDepth).map { _ -> [Module] in
            [
                MultiHeadAttention(nHead: nHeads, nFeat: nDim, bias: false, headDim: nDimHead),
                PerceiverFeedForward(dim: nDim, dFF: dFF),
            ]
        }
        self.norm = RMSNorm(dimensions: nDim)
        super.init()
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        let b = x0.dim(0)
        var latentsB = broadcast(latents, to: [b, latents.dim(0), latents.dim(1)])
        let x = proj_context(x0)
        for pair in layers {
            let attn = pair[0] as! MultiHeadAttention
            let ff = pair[1] as! PerceiverFeedForward
            let kv = concatenated([x, latentsB], axis: -2)
            latentsB = latentsB + attn(latentsB, kv, kv)
            latentsB = latentsB + ff(latentsB)
        }
        return norm(latentsB)
    }
}
