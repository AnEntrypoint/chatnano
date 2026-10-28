"""Write a Chatterbox-Nano weight directory with the large files stored in half precision.

    python export_fp16.py <snapshot_dir> <out_dir>

The served model is half precision already (fast_nano._apply_fp16_t3), so loading half-precision
files gives the same values as halving the float32 ones: the conversion is exact for the served
path. Floating tensors become float16; integer tensors (s3gen buffers) are kept as they are. The
small files the loader needs are copied through unchanged.
"""
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LARGE = ["t3_nano_v1.safetensors", "s3gen_meanflow.safetensors"]
COPY = ["ve.safetensors", "conds.pt", "added_tokens.json", "merges.txt", "special_tokens_map.json",
        "tokenizer_config.json", "vocab.json"]


def to_half(tensors):
    return {k: (v.to(torch.float16) if v.is_floating_point() else v) for k, v in tensors.items()}


def main(snapshot, out):
    snapshot, out = Path(snapshot), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in LARGE:
        src = (snapshot / name).resolve()
        tensors = load_file(src)
        save_file(to_half(tensors), out / name)
        print(f"{name}: {src.stat().st_size / 2**20:.1f} MiB -> {(out / name).stat().st_size / 2**20:.1f} MiB")
    for name in COPY:
        src = (snapshot / name).resolve()
        shutil.copyfile(src, out / name)
    print(f"copied {len(COPY)} small files; out = {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
