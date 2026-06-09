import Foundation
import MLX
import MLXNN

// MARK: - BigVGAN v2 (IndexTTS-2)
//
// Pure mel-to-audio vocoder (nvidia/bigvgan_v2_22khz_80band_256x). Unlike the
// 1.5 vocoder there is **no** speaker encoder / conditioning injection: input is
// an 80-band mel spectrogram, output is the 22.05 kHz waveform.
//
// Differences from the 1.5 `BigVGANConditioning`:
//   * weights are pre-folded weight-norm (single `.weight`), so plain `Conv1d`
//     / `ConvTransposed1d` are used instead of `WNConv1d`.
//   * 80 mel channels in, no `speaker_encoder` / `cond_layer` / `conds`.
//   * `use_tanh_at_final = false`, `use_bias_at_final = false` (matches the
//     converted checkpoint: `conv_post` has no bias, output is clipped).
//
// Everything runs in NLC (batch, length, channels) layout internally and the
// Snake/Activation1d anti-aliasing infra is shared with `Activations.swift`.

public struct BigVGANV2Config {
    public init() {}
    var numMels: Int = 80
    var upsampleRates: [Int] = [4, 4, 2, 2, 2, 2]
    var upsampleKernelSizes: [Int] = [8, 8, 4, 4, 4, 4]
    var upsampleInitialChannel: Int = 1536
    var resblockKernelSizes: [Int] = [3, 7, 11]
    var resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    var activation: String = "snakebeta"
    var snakeLogscale: Bool = true
    var useTanhAtFinal: Bool = false
    var useBiasAtFinal: Bool = false
}

/// Anti-aliased multi-periodicity block (type 1), folded-weight variant.
/// `convs1` (dilated) + `convs2` (dilation 1), one Activation1d before each conv.
/// Operates in NLC layout.
final class AMPBlock1V2: Module {
    let convs1: [Conv1d]
    let convs2: [Conv1d]
    let activations: [Activation1d]

    init(channels: Int, kernelSize: Int, dilations: [Int], activation: String, alphaLogscale: Bool) {
        self.convs1 = dilations.map { d in
            Conv1d(
                inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                padding: getPadding(kernelSize, d), dilation: d, bias: true)
        }
        self.convs2 = dilations.map { _ in
            Conv1d(
                inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                padding: getPadding(kernelSize, 1), dilation: 1, bias: true)
        }
        self.activations = (0 ..< dilations.count * 2).map { _ in
            Activation1d(
                act: activation == "snake"
                    ? Snake(channels, alphaLogscale: alphaLogscale)
                    : SnakeBeta(channels, alphaLogscale: alphaLogscale))
        }
        super.init()
    }

    /// `x`: (batch, length, channels) NLC.
    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = x0
        for i in 0 ..< convs1.count {
            let a1 = activations[2 * i]
            let a2 = activations[2 * i + 1]
            x = x + convs2[i](a2(convs1[i](a1(x))))
        }
        return x
    }
}

func getPadding(_ kernelSize: Int, _ dilation: Int = 1) -> Int {
    (kernelSize * dilation - dilation) / 2
}

public final class BigVGANV2: Module {
    let numKernels: Int
    let numUpsamples: Int
    let useTanhAtFinal: Bool
    /// Internal compute precision. Set to `.bfloat16` (via `castParameters`) for
    /// ~2× speedup; inputs are cast in and the waveform cast back to fp32 on exit.
    var computeDType: DType = .float32

    var conv_pre: Conv1d
    let ups: [ConvTransposed1d]
    let resblocks: [AMPBlock1V2]
    let activation_post: Activation1d
    let conv_post: Conv1d

    public init(_ c: BigVGANV2Config = BigVGANV2Config()) {
        self.numKernels = c.resblockKernelSizes.count
        self.numUpsamples = c.upsampleRates.count
        self.useTanhAtFinal = c.useTanhAtFinal

        self.conv_pre = Conv1d(
            inputChannels: c.numMels, outputChannels: c.upsampleInitialChannel,
            kernelSize: 7, padding: 3, bias: true)

        var upsList: [ConvTransposed1d] = []
        var ch = c.upsampleInitialChannel
        for (rate, kernel) in zip(c.upsampleRates, c.upsampleKernelSizes) {
            let outCh = ch / 2
            upsList.append(
                ConvTransposed1d(
                    inputChannels: ch, outputChannels: outCh, kernelSize: kernel,
                    stride: rate, padding: (kernel - rate) / 2, bias: true))
            ch = outCh
        }
        self.ups = upsList

        var rbList: [AMPBlock1V2] = []
        ch = c.upsampleInitialChannel
        for _ in 0 ..< numUpsamples {
            ch = ch / 2
            for (k, d) in zip(c.resblockKernelSizes, c.resblockDilationSizes) {
                rbList.append(
                    AMPBlock1V2(
                        channels: ch, kernelSize: k, dilations: d,
                        activation: c.activation, alphaLogscale: c.snakeLogscale))
            }
        }
        self.resblocks = rbList

        let postCh = ch
        let postAct: Module =
            c.activation == "snake"
            ? Snake(postCh, alphaLogscale: c.snakeLogscale)
            : SnakeBeta(postCh, alphaLogscale: c.snakeLogscale)
        self.activation_post = Activation1d(act: postAct)
        self.conv_post = Conv1d(
            inputChannels: postCh, outputChannels: 1, kernelSize: 7, padding: 3,
            bias: c.useBiasAtFinal)
        super.init()
    }

    /// `mel`: (batch, numMels, time) NCL. Returns (batch, 1, samples) NCL.
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = mel.transposed(0, 2, 1).asType(computeDType)  // NCL -> NLC
        x = conv_pre(x)

        for i in 0 ..< numUpsamples {
            x = ups[i](x)
            var xs = resblocks[i * numKernels](x)
            for j in 1 ..< numKernels {
                xs = xs + resblocks[i * numKernels + j](x)
            }
            x = xs / Float(numKernels)
        }

        x = activation_post(x)
        x = conv_post(x)
        x = useTanhAtFinal ? MLX.tanh(x) : MLX.clip(x, min: -1.0, max: 1.0)
        return x.transposed(0, 2, 1).asType(.float32)  // NLC -> NCL
    }
}
