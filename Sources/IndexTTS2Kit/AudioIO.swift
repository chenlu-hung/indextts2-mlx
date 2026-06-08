import AVFoundation
import Foundation
import MLX

public enum AudioIO {

    /// Load an audio file, downmix to mono, resample to `sampleRate`.
    public static func loadAudio(_ url: URL, sampleRate: Int) throws -> MLXArray {
        let file = try AVAudioFile(forReading: url)
        let inFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard frameCount > 0,
            let inBuf = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: frameCount)
        else {
            throw NSError(
                domain: "AudioIO", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Empty or unreadable audio: \(url.path)"])
        }
        try file.read(into: inBuf)

        guard
            let outFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
                interleaved: false),
            let converter = AVAudioConverter(from: inFormat, to: outFormat)
        else {
            throw NSError(
                domain: "AudioIO", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Cannot build audio converter"])
        }

        let ratio = Double(sampleRate) / inFormat.sampleRate
        let outCap = AVAudioFrameCount(Double(frameCount) * ratio) + 4096
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCap) else {
            throw NSError(domain: "AudioIO", code: 3, userInfo: nil)
        }

        var provided = false
        var convError: NSError?
        let status = converter.convert(to: outBuf, error: &convError) { _, outStatus in
            if provided {
                outStatus.pointee = .noDataNow
                return nil
            }
            provided = true
            outStatus.pointee = .haveData
            return inBuf
        }
        if status == .error, let convError { throw convError }

        let n = Int(outBuf.frameLength)
        guard n > 0, let ptr = outBuf.floatChannelData?[0] else {
            throw NSError(
                domain: "AudioIO", code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Resample produced no samples"])
        }
        let samples = Array(UnsafeBufferPointer(start: ptr, count: n))
        return MLXArray(samples)
    }

    /// Write mono Float samples to a 16-bit PCM WAV file.
    public static func writeWAV(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = UInt32(samples.count) * UInt32(bitsPerSample / 8)

        var data = Data()
        func appendStr(_ s: String) { data.append(contentsOf: Array(s.utf8)) }
        func appendU32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func appendU16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }

        appendStr("RIFF")
        appendU32(36 + dataSize)
        appendStr("WAVE")
        appendStr("fmt ")
        appendU32(16)
        appendU16(1)  // PCM
        appendU16(numChannels)
        appendU32(UInt32(sampleRate))
        appendU32(byteRate)
        appendU16(blockAlign)
        appendU16(bitsPerSample)
        appendStr("data")
        appendU32(dataSize)

        data.reserveCapacity(data.count + samples.count * 2)
        for s in samples {
            let clamped = max(-1.0, min(1.0, s))
            let i = Int16(clamped * 32767.0)
            var x = i.littleEndian
            withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
        }
        try data.write(to: url)
    }
}
