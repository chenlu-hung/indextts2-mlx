import Foundation
import MLX
import MLXNN

// MARK: - S2Mel length regulator (InterpolateRegulator)
//
// Upsamples the semantic content (continuous, 1024-d) to the target mel length
// via nearest-neighbour interpolation, then a conv/groupnorm/Mish stack.
// Port of `models/s2mel/length_regulator.py`. Everything is continuous at
// inference (`is_discrete = false`), so `embedding` is loaded but unused.

/// GroupNorm operating in NCL layout, normalising over (channels_per_group, length).
final class LRGroupNorm: Module {
    let numGroups: Int
    var weight: MLXArray
    var bias: MLXArray
    let eps: Float

    init(_ numGroups: Int, _ numChannels: Int, eps: Float = 1e-5) {
        self.numGroups = max(numGroups, 1)
        self.weight = MLXArray.ones([numChannels])
        self.bias = MLXArray.zeros([numChannels])
        self.eps = eps
        super.init()
    }

    /// `x`: (B, C, L) NCL.
    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        let b = x0.dim(0), c = x0.dim(1), l = x0.dim(2)
        var x = x0.reshaped([b, numGroups, c / numGroups, l])
        let mean = x.mean(axes: [2, 3], keepDims: true)
        let variance = x.variance(axes: [2, 3], keepDims: true)
        x = (x - mean) / MLX.sqrt(variance + eps)
        x = x.reshaped([b, c, l])
        return x * weight.reshaped([1, c, 1]) + bias.reshaped([1, c, 1])
    }
}

func mish(_ x: MLXArray) -> MLXArray { x * MLX.tanh(MLX.log(1 + MLX.exp(x))) }

/// Nearest-neighbour interpolation along the length axis. `x`: (B, C, L) NCL.
func interpolateNearest(_ x: MLXArray, _ targetLength: Int) -> MLXArray {
    let length = x.dim(2)
    let scale = Float(length) / Float(targetLength)
    var idx = (MLXArray(Array(0 ..< Int32(targetLength))).asType(.float32) * scale).asType(.int32)
    idx = MLX.clip(idx, min: MLXArray(Int32(0)), max: MLXArray(Int32(length - 1)))
    return x[0..., 0..., idx]
}

final class InterpolateRegulator: Module {
    let channels: Int
    let content_in_proj: Linear
    let embedding: Embedding       // present in checkpoint, unused at inference
    var mask_token: MLXArray
    let model: [Module]            // [Conv, GN, Mish] * 4 + [Conv1x1]
    let numBlocks: Int

    init(
        channels: Int = 512, inChannels: Int = 1024, codebookSize: Int = 2048,
        samplingRatios: [Int] = [1, 1, 1, 1], groups: Int = 1
    ) {
        self.channels = channels
        self.numBlocks = samplingRatios.count
        self.content_in_proj = Linear(inChannels, channels)
        self.embedding = Embedding(embeddingCount: codebookSize, dimensions: channels)
        self.mask_token = MLXArray.zeros([1, channels])

        var m: [Module] = []
        for _ in samplingRatios {
            m.append(Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1))
            m.append(LRGroupNorm(groups, channels))
            m.append(MishLayer())
        }
        m.append(Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1))
        self.model = m
        super.init()
    }

    /// `x`: (B, T, inChannels) continuous content. `targetLen`: scalar target length.
    /// Returns (B, targetLen, channels).
    func callAsFunction(_ x0: MLXArray, targetLen: Int) -> MLXArray {
        var x = content_in_proj(x0)          // (B, T, channels) NLC
        x = x.transposed(0, 2, 1)            // NCL
        x = interpolateNearest(x, targetLen) // (B, C, targetLen)

        for layer in model {
            switch layer {
            case let conv as Conv1d:
                x = conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)  // NCL->NLC->conv->NCL
            case let gn as LRGroupNorm:
                x = gn(x)
            case is MishLayer:
                x = mish(x)
            default:
                break
            }
        }
        return x.transposed(0, 2, 1)         // NCL -> NLC
    }
}

/// Parameter-free Mish wrapper so it can live in the `model` module list.
final class MishLayer: Module {}
