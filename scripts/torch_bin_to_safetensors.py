#!/usr/bin/env python3
"""Convert a PyTorch `.bin` checkpoint (torch.save zip format) to safetensors,
WITHOUT importing torch. Reads the zip + depickles the state_dict manually.

Usage: uv run --with numpy --with safetensors python torch_bin_to_safetensors.py in.bin out.safetensors
"""
import sys
import io
import struct
import pickle
import zipfile
import numpy as np

# torch storage dtype -> numpy dtype + itemsize
_DTYPES = {
    "FloatStorage": (np.float32, 4),
    "DoubleStorage": (np.float64, 8),
    "HalfStorage": (np.float16, 2),
    "BFloat16Storage": ("bfloat16", 2),
    "LongStorage": (np.int64, 8),
    "IntStorage": (np.int32, 4),
    "ShortStorage": (np.int16, 2),
    "CharStorage": (np.int8, 1),
    "ByteStorage": (np.uint8, 1),
    "BoolStorage": (np.bool_, 1),
}


class _Storage:
    def __init__(self, key, dtype_name, numel):
        self.key = key
        self.dtype_name = dtype_name
        self.numel = numel


def load_state_dict(bin_path):
    zf = zipfile.ZipFile(bin_path)
    names = zf.namelist()
    # find the archive prefix (e.g. "archive/")
    pkl_name = next(n for n in names if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("data.pkl")]

    def load_storage_bytes(key):
        return zf.read(prefix + "data/" + str(key))

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch") and name.endswith("Storage"):
                return name  # return the storage type *name* (string)
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if module == "torch._utils" and name == "_rebuild_parameter":
                return _rebuild_parameter
            if module == "collections" and name == "OrderedDict":
                from collections import OrderedDict
                return OrderedDict
            return super().find_class(module, name)

        def persistent_load(self, pid):
            # pid = ('storage', storage_type_name, key, location, numel)
            assert pid[0] == "storage"
            storage_type = pid[1]
            # storage_type may be a string (from find_class) or a class name
            type_name = storage_type if isinstance(storage_type, str) else storage_type.__name__
            return _Storage(pid[2], type_name, pid[4])

    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad,
                           backward_hooks, *args):
        np_dtype, itemsize = _DTYPES[storage.dtype_name]
        raw = load_storage_bytes(storage.key)
        if np_dtype == "bfloat16":
            # interpret as uint16, widen to float32 by shifting into upper half
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            flat = u32.view(np.float32)
        else:
            flat = np.frombuffer(raw, dtype=np_dtype)
        size = tuple(size)
        numel = int(np.prod(size)) if size else 1
        # account for storage_offset
        flat = flat[storage_offset: storage_offset + numel]
        if size:
            arr = np.lib.stride_tricks.as_strided(
                flat,
                shape=size,
                strides=tuple(s * flat.itemsize for s in stride),
            ).copy()
        else:
            arr = flat.reshape(())
        return arr

    def _rebuild_parameter(data, requires_grad, backward_hooks):
        return data

    with zf.open(pkl_name) as f:
        data = f.read()
    up = Unpickler(io.BytesIO(data))
    sd = up.load()
    return sd


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    sd = load_state_dict(in_path)
    # flatten OrderedDict to dict[str, np.ndarray]
    tensors = {}
    for k, v in sd.items():
        if isinstance(v, np.ndarray):
            tensors[k] = np.ascontiguousarray(v)
        else:
            print(f"  skip non-tensor key {k!r}: {type(v)}")
    print(f"loaded {len(tensors)} tensors")
    if "--print-keys" in sys.argv:
        for k in sorted(tensors):
            print(f"  {k}: {tensors[k].shape} {tensors[k].dtype}")
    from safetensors.numpy import save_file
    # safetensors requires contiguous; ensure dtype is supported
    save_file(tensors, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
