import Foundation

// Post-processing helpers ported from `generate.py` (operate on plain Swift
// buffers since they run once per utterance and don't need MLX).

/// Compress runs of the silence mel-code so long pauses don't blow up. Mirrors
/// PyTorch IndexTTS `remove_long_silence`: only acts when the total silence count
/// exceeds `maxConsecutive`, then caps each run at `keep`.
public func compressSilence(
    _ melCodes: [Int], silentToken: Int = 52, maxConsecutive: Int = 30, keep: Int = 10
) -> [Int] {
    let count = melCodes.reduce(0) { $0 + ($1 == silentToken ? 1 : 0) }
    if count <= maxConsecutive { return melCodes }

    var result: [Int] = []
    result.reserveCapacity(melCodes.count)
    var consecutive = 0
    for code in melCodes {
        if code != silentToken {
            result.append(code)
            consecutive = 0
        } else if consecutive < keep {
            result.append(code)
            consecutive += 1
        }
    }
    return result
}

/// Linear crossfade between consecutive audio segments to hide boundary clicks.
/// Port of `crossfade_segments`.
public func crossfadeSegments(
    _ segments: [[Float]], sampleRate: Int, overlapMs: Int = 50
) -> [Float] {
    if segments.isEmpty { return [] }
    if segments.count == 1 { return segments[0] }
    if overlapMs <= 0 { return segments.flatMap { $0 } }

    let overlap = overlapMs * sampleRate / 1000
    var result = segments[0]

    for k in 1 ..< segments.count {
        let cur = segments[k]
        if result.count < overlap || cur.count < overlap {
            result.append(contentsOf: cur)
            continue
        }
        let base = result.count - overlap
        for n in 0 ..< overlap {
            let fadeOut = 1.0 - Float(n) / Float(overlap - 1)
            let fadeIn = Float(n) / Float(overlap - 1)
            result[base + n] = result[base + n] * fadeOut + cur[n] * fadeIn
        }
        result.append(contentsOf: cur[overlap...])
    }
    return result
}

/// WSOLA (Waveform Similarity Overlap-Add) time stretch — `rate > 1` speeds up,
/// `rate < 1` slows down, no pitch change. Port of `time_stretch_wsola`.
public func timeStretchWSOLA(
    _ audio: [Float], rate: Float, frameMs: Int = 30, sampleRate: Int = 22050
) -> [Float] {
    if abs(rate - 1.0) < 1e-3 { return audio }

    let frameLen = sampleRate * frameMs / 1000
    let half = frameLen / 2
    let synHop = half
    let anaHop = Int(Float(synHop) * rate)
    let search = half / 2

    let n = audio.count
    let outLen = Int(Float(n) / rate) + frameLen
    var output = [Float](repeating: 0, count: outLen)
    var norm = [Float](repeating: 0, count: outLen)

    // Hann window.
    var window = [Float](repeating: 0, count: frameLen)
    for i in 0 ..< frameLen {
        window[i] = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(i) / Float(frameLen - 1))
    }

    var inPos = 0
    var outPos = 0
    while inPos + frameLen < n && outPos + frameLen < outLen {
        var bestOffset = 0
        if outPos != 0 {
            let lo = max(0, inPos - search)
            let hi = min(n - frameLen, inPos + search)
            if lo < hi {
                var bestCorr = -Float.greatestFiniteMagnitude
                for offset in lo ..< hi {
                    var corr: Float = 0
                    for m in 0 ..< half {
                        corr += output[outPos + m] * audio[offset + m]
                    }
                    if corr > bestCorr {
                        bestCorr = corr
                        bestOffset = offset - inPos
                    }
                }
            }
        }

        var srcPos = inPos + bestOffset
        if srcPos < 0 { srcPos = 0 }
        if srcPos + frameLen > n { break }

        for m in 0 ..< frameLen {
            output[outPos + m] += audio[srcPos + m] * window[m]
            norm[outPos + m] += window[m]
        }
        inPos += anaHop
        outPos += synHop
    }

    var lastNonzero = outPos
    for i in 0 ..< outLen where norm[i] > 1e-8 {
        output[i] /= norm[i]
        lastNonzero = i + 1
    }
    return Array(output[0 ..< lastNonzero])
}
