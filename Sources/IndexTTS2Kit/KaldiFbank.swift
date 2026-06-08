import Foundation
import MLX
import MLXFFT

/// Torch-free port of `torchaudio.compliance.kaldi.fbank` for the CAMPPlus
/// speaker encoder, with the exact defaults used in `generate_v2.py`:
///   `num_mel_bins=80, dither=0, sample_frequency=16000`
/// (frame_length 25ms, frame_shift 10ms, povey window, preemph 0.97,
/// remove_dc_offset, use_power, use_log_fbank, snip_edges, low_freq 20).
///
/// Returns `(numFrames, 80)` log-mel-energy features (NOT mean-subtracted —
/// the caller does per-utterance mean subtraction).
public enum KaldiFbank {
    public static let sr = 16000
    static let numMelBins = 80
    static let frameLength = 25.0   // ms
    static let frameShift = 10.0    // ms
    static let preemph: Float = 0.97
    static let lowFreq = 20.0
    static let highFreq = 0.0       // -> nyquist

    static func melScale(_ f: Double) -> Double { 1127.0 * log(1.0 + f / 700.0) }

    /// Povey window: symmetric Hann (`periodic=false`) raised to the 0.85 power.
    static func poveyWindow(_ n: Int) -> MLXArray {
        let vals = (0 ..< n).map { i -> Float in
            let h = 0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / Double(n - 1))
            return Float(pow(h, 0.85))
        }
        return MLXArray(vals)
    }

    /// Kaldi triangular mel filterbank, shape `(numBins, numFftBins)` where
    /// `numFftBins = paddedWindow/2`. Computed in Double on CPU for fidelity.
    static func melBanks(numBins: Int, paddedWindow: Int, sampleFreq: Double,
                         lowFreq: Double, highFreq: Double) -> MLXArray {
        let numFftBins = paddedWindow / 2
        let nyquist = 0.5 * sampleFreq
        var hi = highFreq
        if hi <= 0.0 { hi += nyquist }
        let fftBinWidth = sampleFreq / Double(paddedWindow)
        let melLow = melScale(lowFreq)
        let melHigh = melScale(hi)
        let melDelta = (melHigh - melLow) / Double(numBins + 1)
        var fb = [Float](repeating: 0, count: numBins * numFftBins)
        for b in 0 ..< numBins {
            let leftMel = melLow + Double(b) * melDelta
            let centerMel = melLow + Double(b + 1) * melDelta
            let rightMel = melLow + Double(b + 2) * melDelta
            for f in 0 ..< numFftBins {
                let mel = melScale(fftBinWidth * Double(f))
                let up = (mel - leftMel) / (centerMel - leftMel)
                let down = (rightMel - mel) / (rightMel - centerMel)
                let v = max(0.0, min(up, down))
                fb[b * numFftBins + f] = Float(v)
            }
        }
        return MLXArray(fb, [numBins, numFftBins])
    }

    /// `wave`: (samples,) at 16 kHz, float in [-1, 1]. Returns `(frames, 80)`.
    /// `scale` pre-multiplies the waveform (SeamlessM4T uses 2^15); `melFloor`
    /// is the pre-log clamp (kaldi/CAMPPlus uses float eps, SeamlessM4T 1.19e-7).
    public static func fbank(_ wave0: MLXArray, scale: Float = 1.0,
                             melFloor: Float = Float.ulpOfOne) -> MLXArray {
        var wave = wave0
        if wave.ndim == 2 { wave = wave.reshaped([-1]) }
        if scale != 1.0 { wave = wave * scale }

        let winShift = Int(Double(sr) * frameShift * 0.001)   // 160
        let winSize = Int(Double(sr) * frameLength * 0.001)    // 400
        var padded = 1
        while padded < winSize { padded <<= 1 }                // 512

        let n = wave.dim(0)
        precondition(n >= winSize, "audio too short for fbank")
        let m = 1 + (n - winSize) / winShift

        // Compute in float64: low-energy mel bins lose relative precision in the
        // float32 FFT + 257-term mel matmul, which the W2V-BERT per-bin variance
        // normalization then amplifies. Double precision keeps parity with the
        // reference. float64 isn't supported on the Metal GPU, so run on the CPU
        // device, returning an evaluated float32 result. (Single utterance — cheap.)
        return Device.withDefaultDevice(.cpu) {
            // Frame via strided view: (m, winSize), in float64.
            let frames = asStrided(wave.asType(.float64), [m, winSize], strides: [winShift, 1])

            // Remove DC offset (per-frame mean).
            var x = frames - frames.mean(axis: 1, keepDims: true)

            // Preemphasis with prev = max(0, j-1).
            let prev = concatenated([x[0..., 0 ..< 1], x[0..., 0 ..< (winSize - 1)]], axis: 1)
            x = x - preemph * prev

            // Window, then zero-pad to padded window size.
            x = x * poveyWindow(winSize).asType(.float64)
            if padded > winSize {
                x = concatenated([x, MLXArray.zeros([m, padded - winSize], dtype: .float64)], axis: 1)
            }

            // Power spectrum, mel, log.
            let spec = MLXFFT.rfft(x, axis: -1)              // (m, padded/2+1)
            let power = MLX.abs(spec) * MLX.abs(spec)        // |.|^2
            var mb = melBanks(numBins: numMelBins, paddedWindow: padded,
                              sampleFreq: Double(sr), lowFreq: lowFreq, highFreq: highFreq).asType(.float64)
            mb = concatenated([mb, MLXArray.zeros([numMelBins, 1], dtype: .float64)], axis: 1)  // (80, padded/2+1)
            let melE = matmul(power, mb.transposed())        // (m, 80)
            let logMel = MLX.log(MLX.maximum(melE, MLXArray(Double(melFloor)))).asType(.float32)
            eval(logMel)
            return logMel
        }
    }
}
