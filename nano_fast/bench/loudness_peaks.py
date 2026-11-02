"""Output loudness: reference generate() against the fast path, same seeds, peak and RMS of the watermarked wav.

The reference runs first, because FastNano halves the T3 of the same base model in place.
Usage: python3 -I loudness_peaks.py OUT.json
"""
import json
import sys

import torch

out_path = sys.argv[1]
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
from fast_nano import FastNano  # noqa: E402

TEXT = ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
        "Would you like me to get some prices for you?")
SEEDS = range(5)


def stats(wav):
    x = wav.float().flatten()
    return {"peak": round(float(x.abs().max()), 4), "rms": round(float(x.pow(2).mean().sqrt()), 4)}


base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
result = {"reference": {}, "fast": {}}
with torch.inference_mode():
    for seed in SEEDS:
        torch.manual_seed(seed)
        result["reference"][seed] = stats(base.generate(TEXT))
    fast = FastNano(base)
    for seed in SEEDS:
        torch.manual_seed(seed)
        result["fast"][seed] = stats(fast.generate(TEXT))
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print("RESULT", json.dumps(result), flush=True)
