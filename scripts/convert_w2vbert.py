#!/usr/bin/env python3
"""Convert facebook/w2v-bert-2.0 torch safetensors -> MLX layout.

Only feature_projection + encoder.layers.0..16 are needed (spk_cond_emb uses
hidden_states[17] = output of layer 16). Conv1d weights are transposed; layers
>= keep_layers and unused top-level tensors are dropped.

Usage: uv run ... python convert_w2vbert.py in.safetensors out.safetensors [keep_layers=17]
"""
import re
import sys
import numpy as np
from safetensors.numpy import load_file, save_file


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    keep = int(sys.argv[3]) if len(sys.argv) > 3 else 17
    sd = load_file(in_path)
    out = {}
    kept_layers = set()
    for k, v in sd.items():
        m = re.match(r"(?:.*\.)?encoder\.layers\.(\d+)\.", k)
        if m:
            li = int(m.group(1))
            if li >= keep:
                continue
            kept_layers.add(li)
        elif "feature_projection" not in k:
            # drop masked_spec_embed, adapter, encoder.layer_norm(if any), etc. not needed
            # (feature_projection + layers are all we use)
            continue
        # strip any model prefix so keys are feature_projection.* / encoder.layers.*
        nk = k
        nk = re.sub(r"^.*?(feature_projection\.)", r"\1", nk)
        nk = re.sub(r"^.*?(encoder\.layers\.)", r"\1", nk)
        if nk.endswith(".weight") and v.ndim == 3:
            v = np.ascontiguousarray(np.transpose(v, (0, 2, 1)))  # OIK -> OKI
        out[nk] = np.ascontiguousarray(v.astype(np.float32))
    print(f"kept layers: {sorted(kept_layers)}")
    save_file(out, out_path)
    print(f"wrote {out_path} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
