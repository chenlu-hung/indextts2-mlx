#!/usr/bin/env python3
"""Convert amphion/MaskGCT semantic_codec (RepCodec) torch safetensors -> MLX layout.

Only the encoder + quantizer are needed for `quantize()` (the decoder is dropped).
- fold weight_norm (weight_g/weight_v -> weight) for in_project/out_project
- Conv1d weight (out,in,k) -> (out,k,in)
- drop decoder.*

Usage: uv run --with numpy --with safetensors python convert_semantic_codec.py in.safetensors out.safetensors
"""
import sys
import numpy as np
from safetensors.numpy import load_file, save_file


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    sd = load_file(in_path)
    out = {}
    # fold weight norm pairs
    g_keys = {k[:-2] for k in sd if k.endswith("_g")}
    handled = set()
    for base in g_keys:
        wv = sd[base + "_v"].astype(np.float32)
        wg = sd[base + "_g"].astype(np.float32)
        norm = np.sqrt((wv ** 2).sum(axis=tuple(range(1, wv.ndim)), keepdims=True))
        sd[base] = wg * (wv / (norm + 1e-8))
        handled.add(base + "_g"); handled.add(base + "_v")

    for k, v in sd.items():
        if k in handled:
            continue
        if k.startswith("decoder."):
            continue
        if k.endswith(".weight") and v.ndim == 3:
            v = np.ascontiguousarray(np.transpose(v, (0, 2, 1)))  # OIK -> OKI
        out[k] = np.ascontiguousarray(v.astype(np.float32))
    save_file(out, out_path)
    print(f"wrote {out_path} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
