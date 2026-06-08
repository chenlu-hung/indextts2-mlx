import Foundation
import MLX
import MLXNN

// MARK: - CAMPPlus speaker encoder (D-TDNN)
//
// Torch-free port of `indextts/s2mel/modules/campplus/{DTDNN,layers}.py`.
// Consumes kaldi-fbank features (B, T, 80) and produces the 192-d `style`
// embedding used for emotion/speaker conditioning. Weights from funasr/campplus
// converted to MLX layout by `scripts/convert_campplus.py`.
//
// Conv2d runs in NHWC (MLX native); Conv1d stacks run in NCL with a transpose
// around each MLX `Conv1d` (which is NLC-native), mirroring PyTorch's NCL flow.

// MARK: BatchNorm (inference: fixed running stats)

/// BatchNorm over the last axis (NHWC channel dim). `affine` toggles weight/bias.
final class CampBatchNorm2d: Module {
    var weight: MLXArray
    var bias: MLXArray
    var running_mean: MLXArray
    var running_var: MLXArray
    let eps: Float

    init(_ c: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([c]); self.bias = MLXArray.zeros([c])
        self.running_mean = MLXArray.zeros([c]); self.running_var = MLXArray.ones([c])
        self.eps = eps
        super.init()
    }
    /// `x`: (B, H, W, C).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = x.dim(3)
        let xn = (x - running_mean.reshaped([1, 1, 1, c])) / MLX.sqrt(running_var.reshaped([1, 1, 1, c]) + eps)
        return xn * weight.reshaped([1, 1, 1, c]) + bias.reshaped([1, 1, 1, c])
    }
}

/// BatchNorm over the channel axis of NCL data. `affine=false` drops weight/bias
/// (the final `dense` layer uses `batchnorm_`, running stats only).
final class CampBatchNorm1d: Module {
    let affine: Bool
    var weight: MLXArray?
    var bias: MLXArray?
    var running_mean: MLXArray
    var running_var: MLXArray
    let eps: Float

    init(_ c: Int, affine: Bool = true, eps: Float = 1e-5) {
        self.affine = affine
        self.weight = affine ? MLXArray.ones([c]) : nil
        self.bias = affine ? MLXArray.zeros([c]) : nil
        self.running_mean = MLXArray.zeros([c]); self.running_var = MLXArray.ones([c])
        self.eps = eps
        super.init()
    }
    /// `x`: (B, C, L).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = x.dim(1)
        var xn = (x - running_mean.reshaped([1, c, 1])) / MLX.sqrt(running_var.reshaped([1, c, 1]) + eps)
        if affine { xn = xn * weight!.reshaped([1, c, 1]) + bias!.reshaped([1, c, 1]) }
        return xn
    }
}

/// `get_nonlinear('batchnorm-relu')`: a BatchNorm1d (key `batchnorm`) + ReLU.
final class BNReLU: Module {
    let batchnorm: CampBatchNorm1d
    init(_ c: Int) { self.batchnorm = CampBatchNorm1d(c); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { MLX.maximum(batchnorm(x), 0) }
}

// MARK: NCL Conv1d helper

/// MLX `Conv1d` subclass that takes/returns NCL (B, C, L) like PyTorch. Subclassing
/// (rather than wrapping) keeps the `weight`/`bias` checkpoint keys flat.
final class NCLConv1d: Conv1d {
    init(_ cin: Int, _ cout: Int, kernel: Int, stride: Int = 1, padding: Int = 0,
         dilation: Int = 1, bias: Bool = false) {
        super.init(inputChannels: cin, outputChannels: cout, kernelSize: kernel,
                   stride: stride, padding: padding, dilation: dilation, bias: bias)
    }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(x.transposed(0, 2, 1)).transposed(0, 2, 1)
    }
}

// MARK: FCM head (2D front-end)

final class CampResBlock: Module {
    let conv1: Conv2d
    let bn1: CampBatchNorm2d
    let conv2: Conv2d
    let bn2: CampBatchNorm2d
    let shortcut: [Module]   // [] or [Conv2d, CampBatchNorm2d]
    let stride: Int

    init(inPlanes: Int, planes: Int, stride: Int) {
        self.stride = stride
        self.conv1 = Conv2d(inputChannels: inPlanes, outputChannels: planes, kernelSize: 3,
                            stride: IntOrPair((stride, 1)), padding: 1, bias: false)
        self.bn1 = CampBatchNorm2d(planes)
        self.conv2 = Conv2d(inputChannels: planes, outputChannels: planes, kernelSize: 3,
                            stride: 1, padding: 1, bias: false)
        self.bn2 = CampBatchNorm2d(planes)
        if stride != 1 || inPlanes != planes {
            self.shortcut = [
                Conv2d(inputChannels: inPlanes, outputChannels: planes, kernelSize: 1,
                       stride: IntOrPair((stride, 1)), bias: false),
                CampBatchNorm2d(planes),
            ]
        } else {
            self.shortcut = []
        }
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = MLX.maximum(bn1(conv1(x)), 0)
        out = bn2(conv2(out))
        var sc = x
        if !shortcut.isEmpty {
            sc = (shortcut[0] as! Conv2d)(x)
            sc = (shortcut[1] as! CampBatchNorm2d)(sc)
        }
        return MLX.maximum(out + sc, 0)
    }
}

final class CampFCM: Module {
    let conv1: Conv2d
    let bn1: CampBatchNorm2d
    let layer1: [CampResBlock]
    let layer2: [CampResBlock]
    let conv2: Conv2d
    let bn2: CampBatchNorm2d
    let outChannels: Int

    init(mChannels: Int = 32, featDim: Int = 80) {
        self.conv1 = Conv2d(inputChannels: 1, outputChannels: mChannels, kernelSize: 3,
                            stride: 1, padding: 1, bias: false)
        self.bn1 = CampBatchNorm2d(mChannels)
        self.layer1 = [CampResBlock(inPlanes: mChannels, planes: mChannels, stride: 2),
                       CampResBlock(inPlanes: mChannels, planes: mChannels, stride: 1)]
        self.layer2 = [CampResBlock(inPlanes: mChannels, planes: mChannels, stride: 2),
                       CampResBlock(inPlanes: mChannels, planes: mChannels, stride: 1)]
        self.conv2 = Conv2d(inputChannels: mChannels, outputChannels: mChannels, kernelSize: 3,
                            stride: IntOrPair((2, 1)), padding: 1, bias: false)
        self.bn2 = CampBatchNorm2d(mChannels)
        self.outChannels = mChannels * (featDim / 8)
        super.init()
    }

    /// `feat`: (B, T, F=80). Returns (B, outChannels=320, T) NCL.
    func callAsFunction(_ feat: MLXArray) -> MLXArray {
        // (B,T,F) -> permute (B,F,T) -> NHWC (B, H=F, W=T, C=1)
        var x = feat.transposed(0, 2, 1).expandedDimensions(axis: 3)
        x = MLX.maximum(bn1(conv1(x)), 0)
        for b in layer1 { x = b(x) }
        for b in layer2 { x = b(x) }
        x = MLX.maximum(bn2(conv2(x)), 0)
        // NHWC (B, H, W=T, C) -> NCHW (B, C, H, T) -> (B, C*H, T)
        let B = x.dim(0), H = x.dim(1), T = x.dim(2), C = x.dim(3)
        x = x.transposed(0, 3, 1, 2).reshaped([B, C * H, T])
        return x
    }
}

// MARK: xvector (D-TDNN backbone)

final class CampTDNNLayer: Module {
    let linear: NCLConv1d
    let nonlinear: BNReLU
    init(_ cin: Int, _ cout: Int, kernel: Int, stride: Int, dilation: Int) {
        let padding = (kernel - 1) / 2 * dilation
        self.linear = NCLConv1d(cin, cout, kernel: kernel, stride: stride, padding: padding,
                                dilation: dilation, bias: false)
        self.nonlinear = BNReLU(cout)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { nonlinear(linear(x)) }
}

final class CampCAMLayer: Module {
    let linear_local: NCLConv1d
    let linear1: NCLConv1d
    let linear2: NCLConv1d
    let segLen = 100

    init(bnChannels: Int, outChannels: Int, kernel: Int, padding: Int, dilation: Int, reduction: Int = 2) {
        self.linear_local = NCLConv1d(bnChannels, outChannels, kernel: kernel, padding: padding,
                                      dilation: dilation, bias: false)
        self.linear1 = NCLConv1d(bnChannels, bnChannels / reduction, kernel: 1, bias: true)
        self.linear2 = NCLConv1d(bnChannels / reduction, outChannels, kernel: 1, bias: true)
        super.init()
    }
    /// `x`: (B, C, L) NCL.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = linear_local(x)
        let context = x.mean(axis: -1, keepDims: true) + segPooling(x)  // (B, C, 1)
        let c = MLX.maximum(linear1(context), 0)
        let m = MLX.sigmoid(linear2(c))
        return y * m
    }
    /// avg_pool1d(kernel=stride=segLen, ceil_mode=True) then expand back to L.
    func segPooling(_ x: MLXArray) -> MLXArray {
        let L = x.dim(2)
        let nseg = (L + segLen - 1) / segLen
        var means: [MLXArray] = []
        for s in 0 ..< nseg {
            let lo = s * segLen, hi = min((s + 1) * segLen, L)
            means.append(x[0..., 0..., lo ..< hi].mean(axis: -1, keepDims: true))  // (B,C,1)
        }
        let seg = concatenated(means, axis: 2)          // (B, C, nseg)
        // repeat each segment segLen times along L (np.repeat), then trim to L.
        let rep = repeated(seg, count: segLen, axis: 2)  // (B, C, nseg*segLen)
        return rep[0..., 0..., 0 ..< L]
    }
}

final class CampDenseTDNNLayer: Module {
    let nonlinear1: BNReLU
    let linear1: NCLConv1d
    let nonlinear2: BNReLU
    let cam_layer: CampCAMLayer

    init(inChannels: Int, outChannels: Int, bnChannels: Int, kernel: Int, dilation: Int) {
        let padding = (kernel - 1) / 2 * dilation
        self.nonlinear1 = BNReLU(inChannels)
        self.linear1 = NCLConv1d(inChannels, bnChannels, kernel: 1, bias: false)
        self.nonlinear2 = BNReLU(bnChannels)
        self.cam_layer = CampCAMLayer(bnChannels: bnChannels, outChannels: outChannels,
                                      kernel: kernel, padding: padding, dilation: dilation)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = nonlinear1(x)
        y = linear1(y)
        y = nonlinear2(y)
        return cam_layer(y)
    }
}

final class CampDenseBlock: Module {
    let layers: [CampDenseTDNNLayer]
    init(numLayers: Int, inChannels: Int, outChannels: Int, bnChannels: Int, kernel: Int, dilation: Int) {
        var ls: [CampDenseTDNNLayer] = []
        for i in 0 ..< numLayers {
            ls.append(CampDenseTDNNLayer(inChannels: inChannels + i * outChannels,
                                         outChannels: outChannels, bnChannels: bnChannels,
                                         kernel: kernel, dilation: dilation))
        }
        self.layers = ls
        super.init()
    }
    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = x0
        for layer in layers {
            x = concatenated([x, layer(x)], axis: 1)
        }
        return x
    }
}

final class CampTransitLayer: Module {
    let nonlinear: BNReLU
    let linear: NCLConv1d
    init(_ cin: Int, _ cout: Int) {
        self.nonlinear = BNReLU(cin)
        self.linear = NCLConv1d(cin, cout, kernel: 1, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(nonlinear(x)) }
}

/// `get_nonlinear('batchnorm_')`: BatchNorm1d (affine=false, key `batchnorm`), no ReLU.
final class BNOnly: Module {
    let batchnorm: CampBatchNorm1d
    init(_ c: Int) { self.batchnorm = CampBatchNorm1d(c, affine: false); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { batchnorm(x) }
}

/// Final DenseLayer with `batchnorm_` (affine=false), no ReLU.
final class CampDenseLayer: Module {
    let linear: NCLConv1d
    let nonlinear: BNOnly
    init(_ cin: Int, _ cout: Int) {
        self.linear = NCLConv1d(cin, cout, kernel: 1, bias: false)
        self.nonlinear = BNOnly(cout)
        super.init()
    }
    /// `x`: (B, 2C) -> (B, cout).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = linear(x.expandedDimensions(axis: 2))  // (B, cout, 1)
        return nonlinear(h)[0..., 0..., 0]
    }
}

final class CampXVector: Module {
    let tdnn: CampTDNNLayer
    let block1: CampDenseBlock
    let transit1: CampTransitLayer
    let block2: CampDenseBlock
    let transit2: CampTransitLayer
    let block3: CampDenseBlock
    let transit3: CampTransitLayer
    @ModuleInfo(key: "out_nonlinear") var out_nonlinear: BNReLU
    let dense: CampDenseLayer

    init(inChannels: Int, embeddingSize: Int = 512, growthRate: Int = 32,
         bnSize: Int = 4, initChannels: Int = 128) {
        self.tdnn = CampTDNNLayer(inChannels, initChannels, kernel: 5, stride: 2, dilation: 1)
        var ch = initChannels
        let specs = [(12, 3, 1), (24, 3, 2), (16, 3, 2)]
        var blocks: [CampDenseBlock] = []
        var transits: [CampTransitLayer] = []
        for (numLayers, kernel, dilation) in specs {
            blocks.append(CampDenseBlock(numLayers: numLayers, inChannels: ch, outChannels: growthRate,
                                         bnChannels: bnSize * growthRate, kernel: kernel, dilation: dilation))
            ch = ch + numLayers * growthRate
            transits.append(CampTransitLayer(ch, ch / 2))
            ch /= 2
        }
        self.block1 = blocks[0]; self.transit1 = transits[0]
        self.block2 = blocks[1]; self.transit2 = transits[1]
        self.block3 = blocks[2]; self.transit3 = transits[2]
        self._out_nonlinear.wrappedValue = BNReLU(ch)
        self.dense = CampDenseLayer(ch * 2, embeddingSize)
        super.init()
    }

    /// `x`: (B, C, T) NCL. Returns (B, 192).
    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = tdnn(x0)
        x = transit1(block1(x))
        x = transit2(block2(x))
        x = transit3(block3(x))
        x = out_nonlinear(x)
        // statistics pooling: mean + unbiased std over time -> (B, 2C)
        let L = x.dim(2)
        let mean = x.mean(axis: 2)                                   // (B, C)
        let diff = x - mean.expandedDimensions(axis: 2)
        let varU = (diff * diff).sum(axis: 2) / Float(L - 1)
        let std = MLX.sqrt(varU)
        let stats = concatenated([mean, std], axis: 1)              // (B, 2C)
        return dense(stats)
    }
}

public final class CAMPPlus: Module {
    let head: CampFCM
    let xvector: CampXVector

    public override init() {
        self.head = CampFCM(featDim: 80)
        // This funasr checkpoint produces a 192-d embedding (`style`).
        self.xvector = CampXVector(inChannels: head.outChannels, embeddingSize: 192)
        super.init()
    }

    /// `feat`: (B, T, 80) kaldi-fbank features (mean-subtracted). Returns `style` (B, 192).
    public func callAsFunction(_ feat: MLXArray) -> MLXArray {
        xvector(head(feat))
    }

    public static func fromPretrained(weights url: URL, verbose: Bool = false) throws -> CAMPPlus {
        let m = CAMPPlus()
        try loadWeights(into: m, from: url, label: "campplus", verbose: verbose)
        eval(m)
        return m
    }
}
