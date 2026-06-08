import Foundation
import MLX
import MLXNN
import MLXRandom

/// Conditional Flow Matching — fixed-step Euler ODE solver with classifier-free
/// guidance, using the DiT as the velocity estimator. Port of models/s2mel/cfm.py.
final class CFM: Module {
    let estimator: DiT
    let inChannels = 80

    override init() {
        self.estimator = DiT()
        super.init()
    }

    /// `mu`: (B, T, 512) content conditioning. `prompt`: (B, 80, promptLen) ref mel.
    /// `style`: (B, 192). Returns generated mel (B, 80, T).
    func inference(
        mu: MLXArray, prompt: MLXArray, style: MLXArray,
        nTimesteps: Int, temperature: Float = 1.0, cfgRate: Float = 0.7
    ) -> MLXArray {
        let b = mu.dim(0), t = mu.dim(1)
        let z = MLXRandom.normal([b, inChannels, t]) * temperature
        let tSpan = (0 ... nTimesteps).map { Float($0) / Float(nTimesteps) }
        return solveEuler(z, prompt: prompt, mu: mu, style: style, tSpan: tSpan, cfgRate: cfgRate)
    }

    private func zeroPromptRegion(_ x: MLXArray, _ promptLen: Int) -> MLXArray {
        let b = x.dim(0), c = x.dim(1)
        return concatenated(
            [MLXArray.zeros([b, c, promptLen]), x[0..., 0..., promptLen...]], axis: 2)
    }

    private func solveEuler(
        _ x0: MLXArray, prompt: MLXArray, mu: MLXArray, style: MLXArray,
        tSpan: [Float], cfgRate: Float
    ) -> MLXArray {
        let b = x0.dim(0), c = x0.dim(1), T = x0.dim(2)
        let promptLen = prompt.dim(2)

        // prompt_x: ref mel in [0, promptLen), zeros after.
        let promptX = concatenated(
            [prompt[0..., 0..., 0 ..< promptLen], MLXArray.zeros([b, c, T - promptLen])], axis: 2)
        var x = zeroPromptRegion(x0, promptLen)

        for step in 1 ..< tSpan.count {
            let t = tSpan[step - 1]
            let dt = tSpan[step] - tSpan[step - 1]
            var dphiDt: MLXArray
            if cfgRate > 0 {
                let sPromptX = concatenated([promptX, MLXArray.zeros(like: promptX)], axis: 0)
                let sStyle = concatenated([style, MLXArray.zeros(like: style)], axis: 0)
                let sMu = concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)
                let sX = concatenated([x, x], axis: 0)
                let sT = MLXArray([t, t])
                let out = estimator(sX, sPromptX, sT, sStyle, sMu)
                let parts = split(out, parts: 2, axis: 0)
                dphiDt = parts[0] * (1.0 + cfgRate) - parts[1] * cfgRate
            } else {
                dphiDt = estimator(x, promptX, MLXArray([t]), style, mu)
            }
            x = x + dphiDt * dt
            x = zeroPromptRegion(x, promptLen)
            eval(x)
        }
        return x
    }
}
