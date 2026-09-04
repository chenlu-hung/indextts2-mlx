# Module: `Sources/indextts2-cli`

## Summary
`main.swift` is the CLI entry point that parses command-line arguments (model directory, reference audio path, text input, optional SRT batch file) and drives `IndexTTS2Kit`'s Pipeline for end-to-end speech synthesis. `--precision fp16|fp32|bf16` (fp16 default) and `--gpt-precision` (fp32 default) resolve to compute dtypes via `computeDType` / `gptDType`, and `--profile` prints `StageTimer` per-stage timings on both the `--text` and `--srt` paths. The SRT batch branch calls `prepareConditioning` once before its loop and reuses that tensor for every cue; `--smoke` instead loads and exercises individual modules (VQ2Emb, BigVGANV2) to verify weight loading without a full synthesis, and synthesized audio is written to disk as WAV.

<!-- projectmap:auto:start (generated — do not edit by hand) -->
## Files (1)
- `Sources/indextts2-cli/main.swift`

## Public symbols (9)
- `function arg` — Sources/indextts2-cli/main.swift:12
- `function flag` — Sources/indextts2-cli/main.swift:17
- `function computeDType` — Sources/indextts2-cli/main.swift:23
- `function gptDType` — Sources/indextts2-cli/main.swift:35
- `function err` — Sources/indextts2-cli/main.swift:46
- `struct SRTEntry` — Sources/indextts2-cli/main.swift:130
- `function dump` — Sources/indextts2-cli/main.swift:249
- `function dumpF` — Sources/indextts2-cli/main.swift:276
- `function dump` — Sources/indextts2-cli/main.swift:319

## Dependencies (imports)
- `Foundation`
- `IndexTTS2Kit`
- `MLX`
- `MLXRandom`
<!-- projectmap:auto:end -->
