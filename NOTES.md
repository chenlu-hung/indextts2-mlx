# IndexTTS-2 MLX-Swift — porting notes (WIP)

Native Swift / MLX port of **IndexTTS-2** (zero-shot voice cloning + emotion +
duration control), torch-free, modeled on the v1.5 project `../index-tts1.5-mlx`.

## Pipeline

```
text ─┐
      ▼
GPT v2 (UnifiedVoiceV2)  ── text + speaker-cond + emotion-cond → semantic mel-codes
      ▼ (codes)
vq2emb (codes→1024 emb) + GPT latent (forward_latent)
      ▼
S2Mel:  gpt_layer (1280→1024)  →  length_regulator (interp to mel len)
        →  CFM Euler ODE w/ DiT estimator (+ CFG)  → 80-band mel
      ▼
BigVGAN v2 (nvidia 22kHz 80-band 256x) → 22.05 kHz waveform
```

Speaker/emotion conditioning come from the **reference audio** via (torch-free TODO):
W2V-BERT 2.0 semantic features → RepCodec quantize (S_ref) ; CAMPPlus style (192-d) ;
22k/80-mel (CFM prompt). See `generate_v2.py::_process_reference_audio`.

## Status

| Component | File | State |
|---|---|---|
| Package scaffold + build | `Package.swift`, `build.sh` | ✅ builds (xcodebuild) |
| Reusable v1.5 modules | `Conformer/Perceiver/Attention/GPT2/DSP/Tokenizer/Normalize/AudioIO/Helpers` | ✅ copied/adapted |
| vq2emb | `VQ2Emb.swift` | ✅ loads + runs |
| BigVGAN v2 | `BigVGANV2.swift` (+ `Activations.swift`) | ✅ 449 keys map, runs |
| GPT v2 (quantized) | `GPTV2.swift`, `Sampling.swift` | ✅ 859 keys map, **forward verified e2e** |
| S2Mel (DiT/CFM/wavenet/length_reg) | `S2Mel/DiT/CFM/WaveNet/LengthRegulator.swift` | ✅ 264 keys map, **forward verified e2e** |
| generate pipeline (`generate()`) | `Pipeline.swift` | ✅ task 6 — runs e2e on synthetic cond (`--gen-smoke`) |
| tokenizer segments + helpers | `TextTokenizer.swift`, `GenerateHelpers.swift` | ✅ split_segments, silence/crossfade/wsola |
| emotion (`--emo-ref`) | `Pipeline.swift` | ✅ base (speaker) + separate emotion-reference audio; emo_matrix feat2.pt path TODO |
| 22k/80-mel `ref_mel` | `Mel.swift` | ✅ **verified vs librosa** (max abs diff 0.0077) |
| W2V-BERT 2.0 (torch-free) | `W2VBert.swift` | ✅ task 7 — feat-extractor + 17 conformer layers; **verified vs numpy** (broadband corr 1.0000, spk max abs 2e-3) |
| RepCodec semantic codec | `RepCodec.swift` | ✅ task 8 — encoder + FVQ quantize; **verified** (S_ref max abs 1e-6, indices exact) |
| CAMPPlus + kaldi fbank | `CAMPPlus.swift`, `KaldiFbank.swift` | ✅ task 9 — D-TDNN + kaldi fbank; **verified** (fbank 7e-4, style 4e-4) |
| length_regulator (continuous path is what prompt_condition uses) | `LengthRegulator.swift` | ✅ S_ref is continuous quantized emb → existing content_in_proj path works (no discrete path needed) |
| wire .wav → conditioning | `ReferenceEncoder.swift` | ✅ task 10 — `IndexTTSv2.makeSpeaker(audioURL:using:)` |
| real CLI (`--ref`/`--text`/`--out`/`--emo-ref`) | `main.swift` | ✅ task 11 — runs e2e; README added. SRT still TODO |

**Verify current state:**
`./build.sh Debug && ./.build/xcode/Build/Products/Debug/indextts2 --model models/mlx-indextts2-standard-8bit --gen-smoke --out /tmp/g.wav`
(also `--smoke` load-check, `--mel-dump` mel parity. Mel ref check: `uv run --with "numpy<2" --with librosa python /tmp/ref_mel.py`.)

## Build configuration — use Release for any real batch

`build.sh` defaults to **Debug** (`CONFIG="${1:-Debug}"`), which is what you want while porting.
For anything that synthesises more than a handful of segments, build Release instead:

```bash
./build.sh Release
# → .build/xcode/Build/Products/Release/indextts2
```

Both configs coexist under `.build/xcode/Build/Products/`, so keep Debug around for iteration
and point batch jobs at the Release binary explicitly.

**Measured difference** (this machine, 2026-08-13; 397-cue Chinese lecture deck driven by
`~/.claude/skills/lecture-video-generator/scripts/synthesize_tts.py`, model
`mlx-indextts2-standard-8bit`, zero-shot `--ref`):

| build | per cue | 397 cues |
|---|---|---|
| Debug | ~2 min | ~14 h |
| Release | ~55 s | ~6 h |

≈ **2.3× faster**. Not the 10× you might expect — most of the work is inside MLX's Metal kernels,
which are prebuilt either way — but 8 hours of wall clock on one deck is worth the one-off build.
Debug was the only config documented here for a long time; that cost a full extra pass before
anyone noticed, hence this section.

Downstream callers take the binary path as an argument, e.g.:

```bash
python3 synthesize_tts.py <topic_dir> --ref voice/ref.wav \
    --indextts2-dir "$INDEXTTS2_DIR" \
    --indextts2-bin "$INDEXTTS2_DIR/.build/xcode/Build/Products/Release/indextts2"
```

Interrupted batches are resumable: per-cue wavs land in `<topic_dir>/.tts_segments/`, and
`--skip-synth` reuses them instead of regenerating. **Switching build config invalidates nothing**
(the wavs are just audio), so it is safe to kill a Debug run, build Release, and restart — but
delete `.tts_segments/` if you want the whole deck synthesised by one build for consistency.

**`generate()` is the entry point** (`Pipeline.swift::IndexTTSv2.generate(text:speaker:options:)`). It consumes a
`SpeakerConditioning{spkCondEmb (1,T,1024), style (1,192), promptCondition (1,Lp,512), refMel (1,80,Lp)}`.
The only thing between here and real `.wav` synthesis is producing that struct from audio (tasks 7-10).
`SpeakerConditioning.load(from:)` reads those keys from a safetensors/npz bundle if one is precomputed elsewhere.

## Key facts gathered (so the next session is fast)

- Model dir `models/mlx-indextts2-standard-8bit` (HF `vanch007/mlx-indextts2-standard-8bit`).
  GPT 8-bit quantized (group_size 64), s2mel/bigvgan/vq2emb fp32. `config.json` has `quantize_bits: 8`.
- **Reference impls:** Python MLX `/tmp/mlx-indextts-ref` (re-clone: `git clone --depth 1 https://github.com/solar2ain/mlx-indextts`). Port generation from `mlx_indextts/generate_v2.py` + `models/{gpt_v2,bigvgan_v2,s2mel/*}.py`. Preprocessing torch source: `mlx_indextts/indextts/utils/maskgct/...` (W2V-BERT via HF transformers `Wav2Vec2BertModel`; RepCodec `repcodec_model.py`+`vocos.py`+`amphion_codec/quantize/*`), `indextts/s2mel/modules/campplus/{DTDNN,layers}.py`.
- **BigVGAN v2 config** (nvidia bigvgan_v2_22khz_80band_256x): 80 mel, init 1536, upsample_rates `[4,4,2,2,2,2]` (product 256 = hop), kernels `[8,8,4,4,4,4]`, resblock "1", dilations `[[1,3,5]]×3`, snakebeta logscale, `use_tanh_at_final=false`, `use_bias_at_final=false`.
- **GPT v2**: dim 1280, 24 layers, 20 heads. conditioning_encoder Conformer(input 1024!, out 512, 6 blocks, 8 heads, ff 2048); perceiver(dim 1280, ctx 512, 32 latents). emo_conditioning_encoder Conformer(input 1024, out 512, 4 blocks, 4 heads, ff 1024); emo_perceiver(dim 1024, 1 latent, 4 heads). conv2d2 subsampling = **single** Conv2d + single Linear (flat keys, not arrays). Double LayerNorm: GPT2Model.ln_f then final_norm (both in checkpoint). AR loop feeds token back at mel-pos `len(mel_codes)+1` (mel_start = pos 0) — replicate exactly.
- **S2Mel** keys: `gpt_layer.layers.{0,1,2}` (1280→256→128→1024), `length_regulator.*` (content_in_proj, embedding[2048,512] unused, mask_token, model.{0,3,6,9}=Conv3x1 / {1,4,7,10}=GroupNorm(1,512) / 12=Conv1x1), `cfm.estimator.*` (DiT: x_embedder 80→512, cond_projection 512→512, t_embedder{,2} with stored `freqs`, cond_x_merge_linear 864→512, skip_linear 592→512, transformer.layers.0-12 {attention.wqkv 1536, wo, attention_norm/ffn_norm = AdaptiveLayerNorm(RMSNorm+project_layer 1024), feed_forward w1/w2/w3 SwiGLU 1536, skip_in_linear 1024→512}, wavenet {cond_layer, in_layers.0-7 k5, res_skip_layers.0-7}, conv1 512→512 Linear, conv2 Conv1d 512→80 k1, final_layer, res_projection). RoPE head_dim 64 base 10000 block 16384 (buffer, not in ckpt → allowed-missing `freqs_cis`).

## MLX-Swift gotchas hit (avoid re-discovering)

- Modules replaced by `quantize()` need `@ModuleInfo var` (not `let`) — see GPT2 c_attn/c_proj/c_fc.
- `Module` subclass no-arg `init()` needs `override init()`.
- `import MLXNN` for `Module`.
- `MLXArray(Range)` unsupported → wrap `MLXArray(Array(0..<n))`.
- Scalar-left math in chains can fail type inference; keep MLXArray on the left or wrap scalar in `MLXArray(...)`.
- `loadWeights(into:from:)` (Loading.swift) reports unmapped/missing keys; kaiser `*.filter`, `*.pe`, `freqs_cis` are allowed-missing (computed buffers).

## Next session

**Done (2026-06-07 session 2):** Task 6 pipeline (`Pipeline.swift`) — generate() ported &
verified e2e via `--gen-smoke` (GPT AR → forwardLatent → s2mel gpt_layer/vq2emb/length_reg →
CFM → BigVGAN → peak-norm; targetLen=Int(codeLen*1.72), cat=[prompt_condition,cond], trim
ref_mel frames). Tokenizer `tokenize`/`splitSegments`/`convertTokensToIds`. Helpers
`compressSilence`/`crossfadeSegments`/`timeStretchWSOLA`. `Mel.swift` ref_mel verified vs librosa.

**Done (2026-06-08 session 3): ALL torch-free preprocessing (tasks 7-11) — full e2e synthesis works.**
- **CAMPPlus + kaldi fbank** (`CAMPPlus.swift`, `KaldiFbank.swift`): D-TDNN + torchaudio-kaldi
  fbank (povey win, kaldi mel triangulated in mel space, 2^15-invariant). `style` (1,192).
  fbank kaldi mel == SeamlessM4T mel (verified equal). funasr checkpoint embedding_size is **192**.
- **RepCodec** (`RepCodec.swift`): VocosBackbone (ConvNeXt) + FactorizedVQ.quantize (cosine-nearest,
  raw codebook lookup, weight-norm folded). S_ref = continuous quantized emb (B,T,1024) → feeds
  the **existing continuous** length_regulator (content_in_proj) — no discrete path needed.
- **W2V-BERT 2.0** (`W2VBert.swift`): SeamlessM4T feat extractor (80-mel + per-bin norm ddof=1 +
  stride-2 stack→160) + feature_projection + **17** conformer layers (relative_key distance attn,
  GLU conv module causal-pad 30) → hidden_states[17] → normalize by stats. Only 17/24 layers loaded.
- **Wiring** (`ReferenceEncoder.swift`): `IndexTTSv2.makeSpeaker(audioURL:using:)` runs all stages +
  length_regulator → `SpeakerConditioning`. **Real CLI** in `main.swift`: `--ref --text --out`
  (+ `--emo-ref`, `--steps/--cfg/--seed/--temperature/--top-p/--top-k/--speed/--max-mel-tokens`).
  README written. Verified e2e on a synthetic ref.wav (218 mel tokens → 4.34s sane audio).

**Conversion (torch-free, numpy)** in `scripts/`: `torch_bin_to_safetensors.py` (depickles torch
zip .bin w/ custom unpickler), `convert_{campplus,semantic_codec,w2vbert}.py` (fold weight-norm,
transpose conv OIK→OKI / OIHW→OHWI, drop unused), `ref_{campplus,repcodec,w2vbert}.py` (numpy
float64 references for parity). MLX weights → `models/preprocessing/*_mlx.safetensors` +
`w2vbert_stats.safetensors`. KaldiFbank runs in **float64 on CPU** (Metal has no f64) — needed
because low-energy mel bins lose precision and the w2v per-bin norm amplifies it.

**KEY GOTCHA:** float64 ops must run inside `Device.withDefaultDevice(.cpu) { ... }` (incl. the
`asType(.float64)` cast) — float64 is unsupported on the Metal GPU and errors otherwise.

**Remaining (optional enhancements):** emo_matrix feat2.pt 8-emotion-weight control (needs feat2.pt
parse + emotion vector blend `emovec_mat + (1-Σw)*base`); SRT-timed batch synthesis. Raw torch-layout
downloads (campplus/semantic_codec/w2v-bert .safetensors, ~2.5GB) kept only for re-running numpy refs.
