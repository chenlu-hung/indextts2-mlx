import Foundation
import MLX
import MLXNN

// MARK: - WaveNet final layer for the DiT (port of models/s2mel/wavenet.py)
//
// All dilations are 1 (dilation_rate = 1), so SConv1d reduces to a Conv1d with
// symmetric reflect padding. Operates in NCL (batch, channels, length).

/// Conv1d with symmetric reflect padding. Stored as `conv` to match the
/// checkpoint key `...<name>.conv.weight` (the converter collapses the original
/// nested `.conv.conv.` into `.conv.`).
final class SConv1d: Module {
    let conv: Conv1d
    let kernelSize: Int

    init(_ inCh: Int, _ outCh: Int, _ kernelSize: Int) {
        self.conv = Conv1d(
            inputChannels: inCh, outputChannels: outCh, kernelSize: kernelSize,
            padding: 0, bias: true)
        self.kernelSize = kernelSize
        super.init()
    }

    /// `x`: (B, C, L) NCL.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)  // NLC
        let pad = (kernelSize - 1) / 2
        if pad > 0 { h = reflectPad(h, axis: 1, pad: pad) }
        h = conv(h)
        return h.transposed(0, 2, 1)   // NCL
    }
}

private func fusedAddTanhSigmoid(_ a: MLXArray, _ b: MLXArray, _ n: Int) -> MLXArray {
    let inAct = a + b
    let t = MLX.tanh(inAct[0..., 0 ..< n, 0...])
    let s = MLX.sigmoid(inAct[0..., n ..< (2 * n), 0...])
    return t * s
}

final class WN: Module {
    let hiddenChannels: Int
    let nLayers: Int
    let cond_layer: SConv1d
    let in_layers: [SConv1d]
    let res_skip_layers: [SConv1d]

    init(hiddenChannels: Int = 512, kernelSize: Int = 5, nLayers: Int = 8, ginChannels: Int = 512) {
        self.hiddenChannels = hiddenChannels
        self.nLayers = nLayers
        self.cond_layer = SConv1d(ginChannels, 2 * hiddenChannels * nLayers, 1)
        self.in_layers = (0 ..< nLayers).map { _ in
            SConv1d(hiddenChannels, 2 * hiddenChannels, kernelSize)
        }
        self.res_skip_layers = (0 ..< nLayers).map { i in
            let resSkipCh = i < nLayers - 1 ? 2 * hiddenChannels : hiddenChannels
            return SConv1d(hiddenChannels, resSkipCh, 1)
        }
        super.init()
    }

    /// `x`: (B, hidden, L) NCL. `g`: (B, gin, 1) conditioning. `xMask`: (B, 1, L).
    func callAsFunction(_ x0: MLXArray, _ xMask: MLXArray, g: MLXArray) -> MLXArray {
        var x = x0
        var output = MLXArray.zeros(like: x)
        let gc = cond_layer(g)  // (B, 2*hidden*nLayers, 1)

        for i in 0 ..< nLayers {
            let xIn = in_layers[i](x)  // (B, 2*hidden, L)
            let off = i * 2 * hiddenChannels
            let gl = gc[0..., off ..< (off + 2 * hiddenChannels), 0...]
            let acts = fusedAddTanhSigmoid(xIn, gl, hiddenChannels)  // (B, hidden, L)
            let resSkip = res_skip_layers[i](acts)
            if i < nLayers - 1 {
                let resActs = resSkip[0..., 0 ..< hiddenChannels, 0...]
                x = (x + resActs) * xMask
                output = output + resSkip[0..., hiddenChannels..., 0...]
            } else {
                output = output + resSkip
            }
        }
        return output * xMask
    }
}
