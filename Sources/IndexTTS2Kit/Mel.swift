import Foundation
import MLX
import MLXFFT

/// 22.05 kHz / 80-band reference mel for the S2Mel CFM prompt (`ref_mel`).
///
/// Exact port of `generate_v2.py::_init_mel_config.mel_spectrogram`:
///   * reflect pad by `(n_fft - hop) / 2`
///   * STFT (`center=False`, Hann window)
///   * magnitude `sqrt(re² + im² + 1e-9)`
///   * librosa **Slaney**-normalized mel basis (htk=False, norm="slaney")
///   * `log(clamp(·, min=1e-5))`
///
/// Differs from `logMelSpectrogram` (which uses HTK mels + n_fft/2 centering),
/// so it is kept separate.
public enum IndexTTSMel {
    public static let sr = 22050
    public static let nFft = 1024
    public static let hop = 256
    public static let win = 1024
    public static let nMels = 80
    public static let fmin = 0.0
    public static let fmax = Double(sr) / 2.0  // librosa default when fmax=None

    /// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm="slaney").
    /// Shape `(nMels, nFft/2 + 1)`. Computed in Double on CPU for fidelity.
    static func slaneyMelBasis() -> MLXArray {
        let nFreqs = nFft / 2 + 1

        // Slaney mel scale.
        let fSp = 200.0 / 3.0
        let minLogHz = 1000.0
        let minLogMel = (minLogHz - 0.0) / fSp  // = 15
        let logstep = log(6.4) / 27.0
        func hzToMel(_ f: Double) -> Double {
            f >= minLogHz ? minLogMel + log(f / minLogHz) / logstep : f / fSp
        }
        func melToHz(_ m: Double) -> Double {
            m >= minLogMel ? minLogHz * exp(logstep * (m - minLogMel)) : fSp * m
        }

        // mel band edges in Hz.
        let mMin = hzToMel(fmin)
        let mMax = hzToMel(fmax)
        let melF = (0 ..< (nMels + 2)).map { i -> Double in
            melToHz(mMin + (mMax - mMin) * Double(i) / Double(nMels + 1))
        }
        // fft bin frequencies.
        let fftFreqs = (0 ..< nFreqs).map { Double($0) * Double(sr) / Double(nFft) }

        let fdiff = (0 ..< (melF.count - 1)).map { melF[$0 + 1] - melF[$0] }
        var fb = [Float](repeating: 0, count: nMels * nFreqs)
        for m in 0 ..< nMels {
            // Slaney area normalization.
            let enorm = 2.0 / (melF[m + 2] - melF[m])
            for f in 0 ..< nFreqs {
                let lower = (fftFreqs[f] - melF[m]) / fdiff[m]
                let upper = (melF[m + 2] - fftFreqs[f]) / fdiff[m + 1]
                let v = max(0.0, min(lower, upper))
                fb[m * nFreqs + f] = Float(v * enorm)
            }
        }
        return MLXArray(fb, [nMels, nFreqs])
    }

    /// Periodic (DFT-even) Hann window — matches `torch.hann_window` (periodic=True),
    /// which the reference uses. Note this differs from the symmetric `hannWindow`.
    static func periodicHann(_ size: Int) -> MLXArray {
        let vals = (0 ..< size).map { n -> Float in
            Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(n) / Double(size)))
        }
        return MLXArray(vals)
    }

    /// `audio`: (samples,) or (1, samples) at 22.05 kHz. Returns `(1, 80, frames)`.
    public static func refMel(_ audio0: MLXArray) -> MLXArray {
        var audio = audio0
        if audio.ndim == 2 { audio = audio.reshaped([-1]) }

        // Reflect pad by (n_fft - hop)/2, frame, window.
        let pad = (nFft - hop) / 2
        let xp = reflectPad(audio, axis: 0, pad: pad)
        let numFrames = 1 + (xp.dim(0) - nFft) / hop
        precondition(numFrames > 0, "Input too short for mel STFT")
        let frames = asStrided(xp, [numFrames, nFft], strides: [hop, 1])
        let spec = MLXFFT.rfft(frames * periodicHann(win), axis: -1)  // (frames, nFreqs)

        // magnitude sqrt(|.|^2 + 1e-9), then mel + log.
        let mag = MLX.sqrt(MLX.abs(spec) * MLX.abs(spec) + MLXArray(Float(1e-9)))
        let basis = slaneyMelBasis()                       // (nMels, nFreqs)
        let mel = matmul(mag, basis.transposed())          // (frames, nMels)
        let logMel = MLX.log(MLX.maximum(mel, MLXArray(Float(1e-5))))
        return logMel.transposed(1, 0).expandedDimensions(axis: 0)  // (1, nMels, frames)
    }
}
