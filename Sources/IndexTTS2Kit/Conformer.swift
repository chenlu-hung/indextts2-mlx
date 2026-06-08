import Foundation
import MLX
import MLXNN

struct ConformerArgs {
    var inputSize: Int = 100
    var outputSize: Int = 512
    var numBlocks: Int = 6
    var linearUnits: Int = 2048
    var attentionHeads: Int = 8
    var inputLayer: String = "conv2d2"
    var cnnModuleKernel: Int = 15
    var posEmbMaxLen: Int = 2048
    var useBias: Bool = true
    var xscaling: Bool = true
    var perceiverMult: Int = 2
}

/// Position-wise feed forward (SiLU) used inside Conformer blocks.
final class ConformerFeedForward: Module {
    let w_1: Linear
    let activation: SiLU
    let w_2: Linear

    init(dim: Int, dFF: Int, useBias: Bool = true) {
        self.w_1 = Linear(dim, dFF, bias: useBias)
        self.activation = SiLU()
        self.w_2 = Linear(dFF, dim, bias: useBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w_2(activation(w_1(x))) }
}

/// Conformer convolution module (pointwise -> GLU -> depthwise -> norm -> act -> pointwise).
final class ConformerConvolution: Module {
    let pointwise_conv1: Conv1d
    let depthwise_conv: Conv1d
    let norm: LayerNorm
    let activation: SiLU
    let pointwise_conv2: Conv1d

    init(_ args: ConformerArgs) {
        precondition((args.cnnModuleKernel - 1) % 2 == 0)
        let c = args.outputSize
        self.pointwise_conv1 = Conv1d(
            inputChannels: c, outputChannels: c * 2, kernelSize: 1, stride: 1, padding: 0,
            bias: args.useBias)
        self.depthwise_conv = Conv1d(
            inputChannels: c, outputChannels: c, kernelSize: args.cnnModuleKernel, stride: 1,
            padding: (args.cnnModuleKernel - 1) / 2, groups: c, bias: args.useBias)
        self.norm = LayerNorm(dimensions: c)
        self.activation = SiLU()
        self.pointwise_conv2 = Conv1d(
            inputChannels: c, outputChannels: c, kernelSize: 1, stride: 1, padding: 0,
            bias: args.useBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = pointwise_conv1(x)
        h = glu(h, axis: -1)
        h = depthwise_conv(h)
        h = norm(h)
        h = activation(h)
        h = pointwise_conv2(h)
        return h
    }
}

final class ConformerBlock: Module {
    let norm_mha: LayerNorm
    let self_attn: RelPositionMultiHeadAttention
    let norm_conv: LayerNorm
    let conv_module: ConformerConvolution
    let norm_ff: LayerNorm
    let feed_forward: ConformerFeedForward
    let norm_final: LayerNorm

    init(_ args: ConformerArgs) {
        self.norm_mha = LayerNorm(dimensions: args.outputSize)
        self.self_attn = RelPositionMultiHeadAttention(
            nHead: args.attentionHeads, nFeat: args.outputSize, bias: args.useBias)
        self.norm_conv = LayerNorm(dimensions: args.outputSize)
        self.conv_module = ConformerConvolution(args)
        self.norm_ff = LayerNorm(dimensions: args.outputSize)
        self.feed_forward = ConformerFeedForward(
            dim: args.outputSize, dFF: args.linearUnits, useBias: args.useBias)
        self.norm_final = LayerNorm(dimensions: args.outputSize)
        super.init()
    }

    func callAsFunction(_ x0: MLXArray, posEmb: MLXArray) -> MLXArray {
        var x = x0
        let xNorm = norm_mha(x)
        x = x + self_attn(xNorm, xNorm, xNorm, posEmb: posEmb)
        x = x + conv_module(norm_conv(x))
        x = x + feed_forward(norm_ff(x))
        return norm_final(x)
    }
}

/// Conv2d subsampling front-end (input_layer="conv2d2" => single 3x3 stride-2 conv).
///
/// The IndexTTS-2 checkpoint stores `embed.conv` / `embed.out` as single modules
/// (flat keys `embed.conv.weight`, `embed.out.weight`), so they are modelled as a
/// single `Conv2d` + a single `Linear` (the ReLU is parameter-free).
final class Conv2dSubsampling: Module {
    let conv: Conv2d
    let out: Linear
    let outFreq: Int

    init(_ args: ConformerArgs) {
        // v2 only uses conv2d2 (a single 3x3 stride-2 conv).
        self.conv = Conv2d(
            inputChannels: 1, outputChannels: args.outputSize,
            kernelSize: IntOrPair(3), stride: IntOrPair(2))
        let freq = (args.inputSize - 3 + 2) / 2
        self.outFreq = freq
        self.out = Linear(args.outputSize * freq, args.outputSize)
        super.init()
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = expandedDimensions(x0, axis: -1)  // (B, T, F, 1)
        x = relu(conv(x))                          // (B, T', W', C)
        // (B, T', W', C) -> (B, T', C, W') -> (B, T', C*W')
        x = x.swappedAxes(2, 3).reshaped([x.dim(0), x.dim(1), -1])
        return out(x)
    }
}

final class Conformer: Module {
    let pos_enc: RelPositionalEncoding
    let embed: Conv2dSubsampling
    let encoders: [ConformerBlock]
    let after_norm: LayerNorm

    init(_ args: ConformerArgs) {
        self.pos_enc = RelPositionalEncoding(
            dModel: args.outputSize, maxLen: args.posEmbMaxLen, scaleInput: args.xscaling)
        self.embed = Conv2dSubsampling(args)
        self.encoders = (0 ..< args.numBlocks).map { _ in ConformerBlock(args) }
        self.after_norm = LayerNorm(dimensions: args.outputSize, eps: 1e-5)
        super.init()
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = embed(x0)
        let (scaled, posEmb) = pos_enc(x, offset: 0)
        x = scaled
        for layer in encoders {
            x = layer(x, posEmb: posEmb)
        }
        return after_norm(x)
    }
}
