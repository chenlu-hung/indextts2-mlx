#!/usr/bin/env python3
"""Convert CAMPPlus (funasr) torch-layout safetensors -> MLX-layout safetensors.

- Conv1d weight (out,in,k)    -> (out,k,in)
- Conv2d weight (out,in,h,w)  -> (out,h,w,in)
- drop *.num_batches_tracked
- rename block `tdnndN` -> `layers.{N-1}` (so Swift can use a [Module] array)

Usage: uv run --with numpy --with safetensors python convert_campplus.py in.safetensors out.safetensors
"""
import re
import sys
import numpy as np
from safetensors.numpy import load_file, save_file


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    sd = load_file(in_path)
    out = {}
    for k, v in sd.items():
        if k.endswith("num_batches_tracked"):
            continue
        nk = re.sub(r"tdnnd(\d+)", lambda m: f"layers.{int(m.group(1)) - 1}", k)
        if k.endswith(".weight") and v.ndim == 4:
            v = np.ascontiguousarray(np.transpose(v, (0, 2, 3, 1)))  # OIHW -> OHWI
        elif k.endswith(".weight") and v.ndim == 3:
            v = np.ascontiguousarray(np.transpose(v, (0, 2, 1)))     # OIK -> OKI
        out[nk] = v
    save_file(out, out_path)
    print(f"wrote {out_path} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
