# Module: `Sources/indextts2-cli`

## Summary
`main.swift` is the CLI entry point that parses command-line arguments (model directory, reference audio path, text input, optional SRT batch file) and drives `IndexTTS2Kit`'s Pipeline for end-to-end speech synthesis. It also provides an `--smoke` path that loads and exercises individual modules (VQ2Emb, BigVGANV2) to verify weight loading without running a full synthesis. Synthesized audio is written to disk as a WAV file.

<!-- projectmap:auto:start (generated — do not edit by hand) -->
## Files (1)
- `Sources/indextts2-cli/main.swift`

## Public symbols (7)
- `function arg` — Sources/indextts2-cli/main.swift:12
- `function flag` — Sources/indextts2-cli/main.swift:17
- `function err` — Sources/indextts2-cli/main.swift:22
- `struct SRTEntry` — Sources/indextts2-cli/main.swift:101
- `function dump` — Sources/indextts2-cli/main.swift:205
- `function dumpF` — Sources/indextts2-cli/main.swift:232
- `function dump` — Sources/indextts2-cli/main.swift:275

## Dependencies (imports)
- `Foundation`
- `IndexTTS2Kit`
- `MLX`
- `MLXRandom`
<!-- projectmap:auto:end -->
