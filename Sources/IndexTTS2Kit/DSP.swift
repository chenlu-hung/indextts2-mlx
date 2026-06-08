import Foundation
import MLX
import MLXFFT

/// Symmetric Hann window (matches mlx_audio `hanning(size, periodic=False)`).
func hannWindow(_ size: Int) -> MLXArray {
    let denom = Double(size - 1)
    let vals = (0 ..< size).map { n -> Float in
        Float(0.5 * (1.0 - cos(2.0 * Double.pi * Double(n) / denom)))
    }
    return MLXArray(vals)
}

/// Short-time Fourier transform with reflect center padding.
/// Returns complex array of shape `(num_frames, n_fft/2 + 1)`.
func stft(_ x: MLXArray, nFft: Int = 1024, hopLength: Int = 256, winLength: Int = 1024) -> MLXArray
{
    var w = hannWindow(winLength)
    if winLength < nFft {
        w = concatenated([w, MLXArray.zeros([nFft - winLength])], axis: 0)
    }

    // center: reflect pad by n_fft // 2
    let xp = reflectPad(x, axis: 0, pad: nFft / 2)

    let numFrames = 1 + (xp.dim(0) - nFft) / hopLength
    precondition(numFrames > 0, "Input too short for STFT")

    let frames = asStrided(xp, [numFrames, nFft], strides: [hopLength, 1])
    return MLXFFT.rfft(frames * w, axis: -1)
}

/// HTK triangular mel filterbank, shape `(n_mels, n_fft/2 + 1)`.
/// Computed in Double precision on the CPU to match the reference closely.
func melFilters(sampleRate: Int = 24000, nFft: Int = 1024, nMels: Int = 100) -> MLXArray {
    let nFreqs = nFft / 2 + 1
    let fMin = 0.0
    let fMax = Double(sampleRate) / 2.0

    func hzToMel(_ f: Double) -> Double { 2595.0 * log10(1.0 + f / 700.0) }
    func melToHz(_ m: Double) -> Double { 700.0 * (pow(10.0, m / 2595.0) - 1.0) }

    // linspace(0, sampleRate/2, nFreqs)
    let allFreqs = (0 ..< nFreqs).map { i -> Double in
        Double(sampleRate / 2) * Double(i) / Double(nFreqs - 1)
    }
    // linspace(m_min, m_max, nMels + 2)
    let mMin = hzToMel(fMin)
    let mMax = hzToMel(fMax)
    let mPts = (0 ..< (nMels + 2)).map { i -> Double in
        mMin + (mMax - mMin) * Double(i) / Double(nMels + 1)
    }
    let fPts = mPts.map(melToHz)
    let fDiff = (0 ..< (fPts.count - 1)).map { fPts[$0 + 1] - fPts[$0] }

    // filterbank[m, f]
    var fb = [Float](repeating: 0, count: nMels * nFreqs)
    for f in 0 ..< nFreqs {
        for m in 0 ..< nMels {
            // slopes uses f_pts indices m, m+1, m+2
            let down = -(fPts[m] - allFreqs[f]) / fDiff[m]
            let up = (fPts[m + 2] - allFreqs[f]) / fDiff[m + 1]
            let v = max(0.0, min(down, up))
            fb[m * nFreqs + f] = Float(v)
        }
    }
    return MLXArray(fb, [nMels, nFreqs])
}

/// Log-mel spectrogram, shape `(1, num_frames, n_mels)`.
public func logMelSpectrogram(
    _ audio: MLXArray, sampleRate: Int = 24000, nMels: Int = 100, nFft: Int = 1024,
    hopLength: Int = 256
) -> MLXArray {
    let freqs = stft(audio, nFft: nFft, hopLength: hopLength, winLength: nFft)
    let magnitudes = MLX.abs(freqs)  // (frames, nFreqs)
    let filters = melFilters(sampleRate: sampleRate, nFft: nFft, nMels: nMels)  // (nMels, nFreqs)
    let melSpec = matmul(magnitudes, filters.transposed())  // (frames, nMels)
    let logSpec = MLX.log(MLX.maximum(melSpec, MLXArray(Float(1e-5))))
    return expandedDimensions(logSpec, axis: 0)
}
