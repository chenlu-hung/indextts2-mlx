import Foundation
import MLX
import MLXNN

// MARK: - Weight-normalized convolutions

func normalizeWeight(_ x: MLXArray, exceptDim: Int = 0) -> MLXArray {
    let axes = (0 ..< x.ndim).filter { $0 != exceptDim }
    return MLX.sqrt(MLX.sum(x * x, axes: axes, keepDims: true))
}

final class WNConv1d: Module {
    var weight_g: MLXArray
    var weight_v: MLXArray
    var bias: MLXArray?
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int

    init(
        _ inCh: Int, _ outCh: Int, _ kernelSize: Int, _ stride: Int = 1, _ padding: Int = 0,
        dilation: Int = 1, groups: Int = 1, bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        let scale = Float(sqrt(1.0 / Double(inCh * kernelSize)))
        let w = MLXRandom.uniform(low: -scale, high: scale, [outCh, kernelSize, inCh / groups])
        self.weight_g = normalizeWeight(w, exceptDim: 0)
        self.weight_v = w / (self.weight_g + 1e-12)
        self.bias = bias ? MLXArray.zeros([outCh]) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let weight = weight_g * weight_v / normalizeWeight(weight_v, exceptDim: 0)
        var y = conv1d(x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
        if let bias { y = y + bias }
        return y
    }
}

final class WNConvTranspose1d: Module {
    var weight_g: MLXArray
    var weight_v: MLXArray
    var bias: MLXArray?
    let stride: Int
    let padding: Int
    let dilation: Int
    let outputPadding: Int

    init(
        _ inCh: Int, _ outCh: Int, _ kernelSize: Int, _ stride: Int = 1, _ padding: Int = 0,
        dilation: Int = 1, outputPadding: Int = 0, bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.outputPadding = outputPadding
        let scale = Float(sqrt(1.0 / Double(inCh * kernelSize)))
        let w = MLXRandom.uniform(low: -scale, high: scale, [outCh, kernelSize, inCh])
        self.weight_g = normalizeWeight(w, exceptDim: 2)
        self.weight_v = w / (self.weight_g + 1e-12)
        self.bias = bias ? MLXArray.zeros([outCh]) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let weight = weight_g * weight_v / normalizeWeight(weight_v, exceptDim: 2)
        var y = convTransposed1d(
            x, weight, stride: stride, padding: padding, dilation: dilation,
            outputPadding: outputPadding)
        if let bias { y = y + bias }
        return y
    }
}

// MARK: - Snake activations

final class Snake: Module {
    var alpha: MLXArray
    let alphaLogscale: Bool

    init(_ inFeatures: Int, alphaLogscale: Bool = false) {
        self.alphaLogscale = alphaLogscale
        self.alpha = (alphaLogscale ? MLXArray.zeros([inFeatures]) : MLXArray.ones([inFeatures]))
        super.init()
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        // channels-middle layout (1, C, 1)
        var a = alpha.reshaped([1, alpha.dim(0), 1])
        if alphaLogscale { a = MLX.exp(a) }
        return x0 + (1.0 / (a + 1e-9)) * MLX.pow(MLX.sin(x0 * a), 2)
    }
}

final class SnakeBeta: Module {
    var alpha: MLXArray
    var beta: MLXArray
    let alphaLogscale: Bool

    init(_ inFeatures: Int, alphaLogscale: Bool = false) {
        self.alphaLogscale = alphaLogscale
        self.alpha = (alphaLogscale ? MLXArray.zeros([inFeatures]) : MLXArray.ones([inFeatures]))
        self.beta = (alphaLogscale ? MLXArray.zeros([inFeatures]) : MLXArray.ones([inFeatures]))
        super.init()
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        // channels-last layout (1, 1, C)
        var a = alpha.reshaped([1, 1, alpha.dim(0)])
        var b = beta.reshaped([1, 1, beta.dim(0)])
        if alphaLogscale {
            a = MLX.exp(a)
            b = MLX.exp(b)
        }
        return x0 + (1.0 / (b + 1e-9)) * MLX.pow(MLX.sin(x0 * a), 2)
    }
}

// MARK: - Anti-aliased resampling (kaiser-windowed sinc)

private func besselI0(_ x: Double) -> Double {
    var sum = 1.0
    var term = 1.0
    let y = x * x / 4.0
    var k = 1.0
    while true {
        term *= y / (k * k)
        sum += term
        if term < 1e-12 * sum { break }
        k += 1
    }
    return sum
}

private func kaiserWindow(_ m: Int, beta: Double) -> [Double] {
    if m == 1 { return [1.0] }
    let alpha = Double(m - 1) / 2.0
    let denom = besselI0(beta)
    return (0 ..< m).map { n -> Double in
        let r = (Double(n) - alpha) / alpha
        return besselI0(beta * sqrt(max(0.0, 1.0 - r * r))) / denom
    }
}

func kaiserSincFilter1d(cutoff: Double, halfWidth: Double, kernelSize: Int) -> MLXArray {
    let even = kernelSize % 2 == 0
    let halfSize = kernelSize / 2

    let deltaF = 4.0 * halfWidth
    let a = 2.285 * (Double(halfSize) - 1.0) * Double.pi * deltaF + 7.95
    let beta: Double
    if a > 50.0 {
        beta = 0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        beta = 0.5842 * pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0)
    } else {
        beta = 0.0
    }
    let window = kaiserWindow(kernelSize, beta: beta)

    var time = [Double](repeating: 0, count: kernelSize)
    if even {
        for i in 0 ..< kernelSize { time[i] = Double(-halfSize + i) + 0.5 }
    } else {
        for i in 0 ..< kernelSize { time[i] = Double(i - halfSize) }
    }

    var filt = [Float](repeating: 0, count: kernelSize)
    if cutoff > 0 {
        func sinc(_ x: Double) -> Double { x == 0 ? 1.0 : sin(Double.pi * x) / (Double.pi * x) }
        var vals = [Double](repeating: 0, count: kernelSize)
        var total = 0.0
        for i in 0 ..< kernelSize {
            vals[i] = 2.0 * cutoff * window[i] * sinc(2.0 * cutoff * time[i])
            total += vals[i]
        }
        for i in 0 ..< kernelSize { filt[i] = Float(vals[i] / total) }
    }
    return MLXArray(filt, [1, kernelSize, 1])
}

final class LowPassFilter1d: Module {
    let stride: Int
    let padLeft: Int
    let padRight: Int
    var filter: MLXArray

    init(cutoff: Double = 0.5, halfWidth: Double = 0.6, stride: Int = 1, kernelSize: Int = 12) {
        let even = kernelSize % 2 == 0
        self.stride = stride
        self.padLeft = kernelSize / 2 - (even ? 1 : 0)
        self.padRight = kernelSize / 2
        self.filter = kaiserSincFilter1d(cutoff: cutoff, halfWidth: halfWidth, kernelSize: kernelSize)
        super.init()
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        let c = x0.dim(2)
        let x = padded(x0, widths: [IntOrPair(0), IntOrPair((padLeft, padRight)), IntOrPair(0)], mode: .edge)
        let w = broadcast(filter, to: [c, filter.dim(1), 1])
        return conv1d(x, w, stride: stride, groups: c)
    }
}

final class UpSample1d: Module {
    let ratio: Int
    let stride: Int
    let pad: Int
    let padLeft: Int
    let padRight: Int
    var filter: MLXArray

    init(ratio: Int = 2, kernelSize: Int? = nil) {
        self.ratio = ratio
        let ks = kernelSize ?? (Int(6 * ratio / 2) * 2)
        self.stride = ratio
        self.pad = ks / ratio - 1
        self.padLeft = pad * stride + (ks - stride) / 2
        self.padRight = pad * stride + (ks - stride + 1) / 2
        self.filter = kaiserSincFilter1d(
            cutoff: 0.5 / Double(ratio), halfWidth: 0.6 / Double(ratio), kernelSize: ks)
        super.init()
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        let c = x0.dim(2)
        let x = padded(x0, widths: [IntOrPair(0), IntOrPair((pad, pad)), IntOrPair(0)], mode: .edge)
        let w = broadcast(filter, to: [c, filter.dim(1), 1])
        var y = Float(ratio) * convTransposed1d(x, w, stride: stride, groups: c)
        y = y[0..., padLeft ..< (y.dim(1) - padRight), 0...]
        return y
    }
}

final class DownSample1d: Module {
    let lowpass: LowPassFilter1d

    init(ratio: Int = 2, kernelSize: Int? = nil) {
        let ks = kernelSize ?? (Int(6 * ratio / 2) * 2)
        self.lowpass = LowPassFilter1d(
            cutoff: 0.5 / Double(ratio), halfWidth: 0.6 / Double(ratio), stride: ratio, kernelSize: ks)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { lowpass(x) }
}

final class Activation1d: Module {
    let act: Module
    let upsample: UpSample1d
    let downsample: DownSample1d

    init(act: Module, upRatio: Int = 2, downRatio: Int = 2, upKernel: Int = 12, downKernel: Int = 12)
    {
        self.act = act
        self.upsample = UpSample1d(ratio: upRatio, kernelSize: upKernel)
        self.downsample = DownSample1d(ratio: downRatio, kernelSize: downKernel)
        super.init()
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = upsample(x0)
        x = (act as! UnaryLayerLike).forward(x)
        x = downsample(x)
        return x
    }
}

/// Lets Snake / SnakeBeta be invoked through a common interface.
protocol UnaryLayerLike {
    func forward(_ x: MLXArray) -> MLXArray
}
extension Snake: UnaryLayerLike { func forward(_ x: MLXArray) -> MLXArray { callAsFunction(x) } }
extension SnakeBeta: UnaryLayerLike { func forward(_ x: MLXArray) -> MLXArray { callAsFunction(x) } }

// MARK: - AMP residual blocks

final class AMPBlock1: Module {
    let convs1: [WNConv1d]
    let convs2: [WNConv1d]
    let activations: [Activation1d]

    init(channels: Int, snakeLogscale: Bool, activation: String, kernelSize: Int = 3,
        dilation: [Int] = [1, 3, 5])
    {
        self.convs1 = dilation.map {
            WNConv1d(channels, channels, kernelSize, 1, ((kernelSize - 1) * $0) / 2, dilation: $0)
        }
        self.convs2 = dilation.map { _ in
            WNConv1d(channels, channels, kernelSize, 1, (kernelSize - 1) / 2, dilation: 1)
        }
        self.activations = (0 ..< dilation.count * 2).map { _ in
            Activation1d(
                act: activation == "snake"
                    ? Snake(channels, alphaLogscale: snakeLogscale)
                    : SnakeBeta(channels, alphaLogscale: snakeLogscale))
        }
        super.init()
    }

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

