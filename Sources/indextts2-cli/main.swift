import Foundation
import MLX
import MLXRandom
import IndexTTS2Kit

// IndexTTS-2 MLX-Swift CLI.
//
// NOTE: this is an early scaffold. Full text->speech generation is being ported
// incrementally (GPT v2, S2Mel, torch-free preprocessing). Right now `--smoke`
// verifies that the already-ported modules (vq2emb, BigVGAN v2) load and run.

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    if let i = a.firstIndex(of: name), i + 1 < a.count { return a[i + 1] }
    return nil
}
func flag(_ name: String) -> Bool { CommandLine.arguments.contains(name) }

let modelDir = arg("--model") ?? "models/mlx-indextts2-standard-8bit"
let modelURL = URL(fileURLWithPath: modelDir)

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

if flag("--smoke") || CommandLine.arguments.count == 1 {
    err("IndexTTS-2 MLX-Swift — smoke test")
    err("model dir: \(modelDir)")

    // vq2emb
    let vq = try VQ2Emb.load(from: modelURL.appendingPathComponent("vq2emb.safetensors"))
    let codes = MLXArray((0 ..< 16).map { Int32($0) }).reshaped([1, 16])
    let emb = vq(codes)
    eval(emb)
    err("vq2emb: codes (1,16) -> emb \(emb.shape)  (expect [1, 1024, 16])")

    // BigVGAN v2
    let bigvgan = BigVGANV2()
    try loadWeights(
        into: bigvgan,
        from: modelURL.appendingPathComponent("bigvgan.safetensors"),
        label: "bigvgan", verbose: true)
    let mel = MLXRandom.normal([1, 80, 20])
    let audio = bigvgan(mel)
    eval(audio)
    err("bigvgan: mel (1,80,20) -> audio \(audio.shape)  (expect [1, 1, \(20 * 256)])")

    // GPT v2 (load + key mapping check; quantized backbone)
    let config = try IndexTTS2Config.load(from: modelURL.appendingPathComponent("config.json"))
    err("config: dim=\(config.gpt.model_dim) layers=\(config.gpt.layers) quantize=\(config.quantize_bits.map(String.init) ?? "none")")
    let gpt = try UnifiedVoiceV2.fromPretrained(directory: modelURL, config: config, verbose: true)
    _ = gpt
    err("gpt v2 loaded")

    // S2Mel (gpt_layer + length_regulator + CFM/DiT)
    let s2mel = try S2Mel.fromPretrained(directory: modelURL, verbose: true)
    _ = s2mel
    err("s2mel loaded")

    err("✓ smoke test complete")
} else if let text = arg("--text"), let refPath = arg("--ref") {
    // Real text-to-speech: reference .wav -> torch-free conditioning -> synthesis.
    let verbose = !flag("--quiet")
    let outPath = arg("--out") ?? "out.wav"
    let preprocDir = arg("--preproc-dir") ?? "models/preprocessing"

    err("IndexTTS-2 MLX-Swift — synthesis")
    let tts = try IndexTTSv2(modelDir: modelURL, verbose: verbose)
    let refEnc = try ReferenceEncoder(dir: URL(fileURLWithPath: preprocDir), verbose: verbose)
    err("loading reference \(refPath) …")
    let speaker = try tts.makeSpeaker(
        audioURL: URL(fileURLWithPath: refPath), using: refEnc, verbose: verbose)

    var opts = GenerationOptions()
    opts.verbose = verbose
    if let v = arg("--steps").flatMap(Int.init) { opts.diffusionSteps = v }
    if let v = arg("--seed").flatMap(UInt64.init) { opts.seed = v }
    if let v = arg("--cfg").flatMap(Float.init) { opts.cfgRate = v }
    if let v = arg("--temperature").flatMap(Float.init) { opts.temperature = v }
    if let v = arg("--top-p").flatMap(Float.init) { opts.topP = v }
    if let v = arg("--top-k").flatMap(Int.init) { opts.topK = v }
    if let v = arg("--speed").flatMap(Float.init) { opts.speed = v }
    if let v = arg("--max-mel-tokens").flatMap(Int.init) { opts.maxMelTokens = v }
    if let emoRef = arg("--emo-ref") {
        // Separate emotion reference: its W2V-BERT features drive the emotion vector.
        opts.emotionEmb = try refEnc.encodeEmotion(audioURL: URL(fileURLWithPath: emoRef))
        err("emotion reference: \(emoRef)")
    }

    let audio = tts.generate(text: text, speaker: speaker, options: opts)
    guard !audio.isEmpty else { err("⚠️  no audio generated"); exit(1) }
    try AudioIO.writeWAV(audio, sampleRate: tts.sampleRate, to: URL(fileURLWithPath: outPath))
    err("✓ wrote \(outPath) (\(String(format: "%.2f", Double(audio.count) / Double(tts.sampleRate)))s)")
} else if let srtPath = arg("--srt"), let refPath = arg("--ref") {
    // SRT batch synthesis: generate one .wav per subtitle entry.
    // Output files go into --out <dir> named <srt-stem>_001.wav, _002.wav, …
    let verbose = !flag("--quiet")
    let outDir = arg("--out") ?? "."
    let preprocDir = arg("--preproc-dir") ?? "models/preprocessing"

    // Parse SRT: blank-line-separated blocks, skip timestamp line.
    let srtContent = try String(contentsOfFile: srtPath, encoding: .utf8)
    struct SRTEntry { let index: Int; let text: String }
    var entries: [SRTEntry] = []
    for block in srtContent.components(separatedBy: "\n\n") {
        let lines = block.trimmingCharacters(in: .whitespacesAndNewlines)
                        .components(separatedBy: "\n")
        guard lines.count >= 3,
              let idx = Int(lines[0].trimmingCharacters(in: .whitespaces)),
              lines[1].contains("-->") else { continue }
        let text = lines[2...].joined(separator: " ")
                              .trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty { entries.append(SRTEntry(index: idx, text: text)) }
    }
    guard !entries.isEmpty else { err("no SRT entries parsed from \(srtPath)"); exit(1) }

    let srtStem = URL(fileURLWithPath: srtPath).deletingPathExtension().lastPathComponent
    try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)

    err("IndexTTS-2 MLX-Swift — SRT batch (\(entries.count) segments)")
    let tts = try IndexTTSv2(modelDir: modelURL, verbose: verbose)
    let refEnc = try ReferenceEncoder(dir: URL(fileURLWithPath: preprocDir), verbose: verbose)
    err("loading reference \(refPath) …")
    let speaker = try tts.makeSpeaker(
        audioURL: URL(fileURLWithPath: refPath), using: refEnc, verbose: verbose)

    var opts = GenerationOptions()
    opts.verbose = verbose
    if let v = arg("--steps").flatMap(Int.init) { opts.diffusionSteps = v }
    if let v = arg("--seed").flatMap(UInt64.init) { opts.seed = v }
    if let v = arg("--cfg").flatMap(Float.init) { opts.cfgRate = v }
    if let v = arg("--temperature").flatMap(Float.init) { opts.temperature = v }
    if let v = arg("--top-p").flatMap(Float.init) { opts.topP = v }
    if let v = arg("--top-k").flatMap(Int.init) { opts.topK = v }
    if let v = arg("--speed").flatMap(Float.init) { opts.speed = v }
    if let v = arg("--max-mel-tokens").flatMap(Int.init) { opts.maxMelTokens = v }
    if let emoRef = arg("--emo-ref") {
        opts.emotionEmb = try refEnc.encodeEmotion(audioURL: URL(fileURLWithPath: emoRef))
    }

    for entry in entries {
        let outFile = URL(fileURLWithPath: outDir)
            .appendingPathComponent(String(format: "%@_%03d.wav", srtStem, entry.index))
        err("[\(entry.index)/\(entries.count)] \(entry.text.prefix(60))")
        var segOpts = opts
        // Use deterministic seed per segment so reruns are reproducible.
        if let base = opts.seed { segOpts.seed = base &+ UInt64(entry.index) }
        let audio = tts.generate(text: entry.text, speaker: speaker, options: segOpts)
        if audio.isEmpty {
            err("  ⚠️  no audio for segment \(entry.index), skipping")
            continue
        }
        try AudioIO.writeWAV(audio, sampleRate: tts.sampleRate, to: outFile)
        err("  ✓ \(outFile.lastPathComponent) (\(String(format: "%.2f", Double(audio.count) / Double(tts.sampleRate)))s)")
    }
    err("✓ SRT batch complete → \(outDir)")
} else if flag("--gen-smoke") {
    // End-to-end generation smoke: run the full MLX chain (GPT AR -> S2Mel CFM ->
    // BigVGAN) on synthetic reference conditioning. Output is not meaningful audio
    // (random speaker), but it exercises every stage for shape/runtime parity.
    err("IndexTTS-2 MLX-Swift — generation smoke test")
    let tts = try IndexTTSv2(modelDir: modelURL, verbose: true)
    err("pipeline loaded (sr=\(tts.sampleRate))")

    let refLen = 80
    let speaker = SpeakerConditioning(
        spkCondEmb: MLXRandom.normal([1, 100, 1024]),
        style: MLXRandom.normal([1, 192]),
        promptCondition: MLXRandom.normal([1, refLen, 512]),
        refMel: MLXRandom.normal([1, 80, refLen]))

    var opts = GenerationOptions()
    opts.maxMelTokens = 30        // cap the AR loop so the smoke is fast
    opts.diffusionSteps = 4
    opts.verbose = true
    opts.seed = 0

    let audio = tts.generate(text: "Hello world.", speaker: speaker, options: opts)
    err("generated \(audio.count) samples (\(String(format: "%.3f", Double(audio.count) / Double(tts.sampleRate)))s)")
    if let out = arg("--out") {
        try AudioIO.writeWAV(audio, sampleRate: tts.sampleRate, to: URL(fileURLWithPath: out))
        err("wrote \(out)")
    }
    err("✓ generation smoke complete")
} else if flag("--campplus-test") {
    // Parity check for CAMPPlus + kaldi fbank: deterministic synthetic 16k wave
    // (matches scripts/ref_campplus.py synth_wave), dump fbank + style as raw f32.
    let n = 16000
    let sine = (0 ..< n).map { i -> Float in
        let t = Double(i) / 16000.0
        return Float(0.6 * sin(2 * .pi * 220 * t) + 0.3 * sin(2 * .pi * 440 * t) + 0.1 * sin(2 * .pi * 90 * t))
    }
    let wave = MLXArray(sine)
    let feat = KaldiFbank.fbank(wave)                 // (frames, 80)
    eval(feat)
    err("fbank shape \(feat.shape) mean=\(feat.mean().item(Float.self)) min=\(feat.min().item(Float.self)) max=\(feat.max().item(Float.self))")

    let featMS = feat - feat.mean(axis: 0, keepDims: true)
    let cpWeights = arg("--campplus") ?? "models/preprocessing/campplus_mlx.safetensors"
    let cp = try CAMPPlus.fromPretrained(weights: URL(fileURLWithPath: cpWeights), verbose: true)
    let style = cp(featMS.expandedDimensions(axis: 0))  // (1, 192)
    eval(style)
    err("style shape \(style.shape) mean=\(style.mean().item(Float.self)) min=\(style.min().item(Float.self)) max=\(style.max().item(Float.self))")

    let dir = arg("--out") ?? "/tmp/campswift"
    try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    func dump(_ a: MLXArray, _ name: String) throws {
        let vals = a.asArray(Float.self)
        var data = Data(capacity: vals.count * 4)
        for v in vals { var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        try data.write(to: URL(fileURLWithPath: dir).appendingPathComponent(name))
    }
    try dump(feat, "fbank.bin")
    try dump(style, "style.bin")
    err("wrote \(dir)/fbank.bin, style.bin")
} else if flag("--repcodec-test") {
    // Parity check for RepCodec quantize(): load input.bin (1, T, 1024) produced
    // by scripts/ref_repcodec.py, dump S_ref + indices for diffing.
    let dir = arg("--out") ?? "/tmp/repref"
    let T = Int(arg("--frames") ?? "50")!
    let inURL = URL(fileURLWithPath: dir).appendingPathComponent("input.bin")
    let raw = try Data(contentsOf: inURL)
    let floats = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let x = MLXArray(floats, [1, T, 1024])

    let weights = arg("--repcodec") ?? "models/preprocessing/semantic_codec_mlx.safetensors"
    let rc = try RepCodec.fromPretrained(weights: URL(fileURLWithPath: weights), verbose: true)
    let (sRef, idx) = rc.quantize(x)
    eval(sRef, idx)
    err("S_ref shape \(sRef.shape) mean=\(sRef.mean().item(Float.self)) min=\(sRef.min().item(Float.self)) max=\(sRef.max().item(Float.self))")
    let idxVals = idx.reshaped([-1]).asArray(Int32.self)
    err("indices[:10] \(Array(idxVals.prefix(10)))")

    func dumpF(_ a: MLXArray, _ name: String) throws {
        let vals = a.asArray(Float.self)
        var data = Data(capacity: vals.count * 4)
        for v in vals { var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        try data.write(to: URL(fileURLWithPath: dir).appendingPathComponent(name))
    }
    try dumpF(sRef, "sref_swift.bin")
    var idata = Data()
    for v in idxVals { var x = v; withUnsafeBytes(of: &x) { idata.append(contentsOf: $0) } }
    try idata.write(to: URL(fileURLWithPath: dir).appendingPathComponent("indices_swift.bin"))
    err("wrote sref_swift.bin, indices_swift.bin")
} else if flag("--w2vbert-test") {
    // Parity check for W2V-BERT spk_cond_emb: same synthetic 16k wave as the
    // numpy reference (scripts/ref_w2vbert.py), dump feat160 + spk.
    let dirEarly = arg("--out") ?? "/tmp/w2vswift"
    let waveURL = URL(fileURLWithPath: dirEarly).appendingPathComponent("wave.bin")
    let wave: MLXArray
    if let wdata = try? Data(contentsOf: waveURL) {
        let wf = wdata.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        wave = MLXArray(wf)
        err("loaded wave.bin (\(wf.count) samples)")
    } else {
        let n = 16000
        let sine = (0 ..< n).map { i -> Float in
            let t = Double(i) / 16000.0
            return Float(0.6 * sin(2 * .pi * 220 * t) + 0.3 * sin(2 * .pi * 440 * t) + 0.1 * sin(2 * .pi * 90 * t))
        }
        wave = MLXArray(sine)
    }
    let feat = W2VFeatureExtractor.extract(wave)
    eval(feat)
    err("feat160 shape \(feat.shape) mean=\(feat.mean().item(Float.self)) std=\(MLX.sqrt(feat.variance()).item(Float.self))")

    let enc = try W2VSpeakerEncoder(
        weights: URL(fileURLWithPath: arg("--w2vbert") ?? "models/preprocessing/w2vbert_mlx.safetensors"),
        stats: URL(fileURLWithPath: arg("--w2vstats") ?? "models/preprocessing/w2vbert_stats.safetensors"),
        verbose: true)
    let spk = enc(wave)
    eval(spk)
    err("spk shape \(spk.shape) mean=\(spk.mean().item(Float.self)) min=\(spk.min().item(Float.self)) max=\(spk.max().item(Float.self))")

    let dir = arg("--out") ?? "/tmp/w2vswift"
    try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    func dump(_ a: MLXArray, _ name: String) throws {
        let vals = a.asArray(Float.self)
        var data = Data(capacity: vals.count * 4)
        for v in vals { var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        try data.write(to: URL(fileURLWithPath: dir).appendingPathComponent(name))
    }
    try dump(feat, "feat160.bin")
    try dump(spk, "spk.bin")
    err("wrote \(dir)/feat160.bin, spk.bin")
} else if flag("--mel-dump") {
    // Numerical-parity check for the 22kHz reference mel: deterministic sine in,
    // raw float32 (80 x frames, row-major) out to --out for diffing vs librosa.
    let sr = IndexTTSMel.sr
    let n = 11025
    let sine = (0 ..< n).map { Float(sin(2.0 * Double.pi * 220.0 * Double($0) / Double(sr))) }
    let mel = IndexTTSMel.refMel(MLXArray(sine))
    eval(mel)
    err("mel shape \(mel.shape)")
    let flat = mel.reshaped([-1])
    err("mean=\(flat.mean().item(Float.self)) min=\(flat.min().item(Float.self)) max=\(flat.max().item(Float.self))")
    let outPath = arg("--out") ?? "/tmp/swift_mel.bin"
    let vals = mel.asArray(Float.self)
    var data = Data(capacity: vals.count * 4)
    for v in vals { var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
    try data.write(to: URL(fileURLWithPath: outPath))
    err("wrote \(outPath) (\(vals.count) floats)")
} else {
    err("usage: indextts2 --model <dir> --ref <ref.wav> --text \"...\" --out out.wav")
    err("       indextts2 --model <dir> --ref <ref.wav> --srt input.srt --out <dir>")
    err("  [--preproc-dir models/preprocessing] [--steps 25] [--cfg 0.7] [--seed N]")
    err("  [--temperature 0.8] [--top-p 0.8] [--top-k 30] [--speed 1.0] [--max-mel-tokens 1500]")
    err("diagnostics: --smoke | --gen-smoke [--out out.wav] | --mel-dump |")
    err("  --campplus-test | --repcodec-test | --w2vbert-test")
}
