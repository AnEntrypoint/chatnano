"""Speech tokens for several seeds and texts, eager decode attention or the Triton kernel, one mode per process.

Token identity across seeds is the check that the kernel's fp32 accumulation changes no sampled token. Writes
OUT.json; compare the two files token by token. Usage: python3 -I decode_attn_tokens.py eager|triton OUT.json
"""
import json
import sys

import torch

mode, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
import decode_attn as DA  # noqa: E402
from fast_nano import FastNano  # noqa: E402

TEXTS = {
    "short": ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
              "Would you like me to get some prices for you?"),
    "long": ("This is a considerably longer sentence, written to push the decoder past a few dozen speech "
             "tokens, so that the cache bucket and the stop check both get exercised across several "
             "boundaries in one run."),
    "paragraph": ("The new model arrived last week, and the team has been testing it against the old one for "
                  "three days. Most of the difference is in the first two seconds of each reply."),
}
SEEDS = (0, 1, 2, 3, 4)

if mode == "triton":
    DA.install()
elif mode != "eager":
    raise SystemExit(f"unknown mode {mode!r}")

base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
fast = FastNano(base)
out = {}
with torch.inference_mode():
    for name, text in TEXTS.items():
        for seed in SEEDS:
            torch.manual_seed(seed)
            out[f"{name}/{seed}"] = fast.speech_tokens(text).flatten().tolist()
with open(out_path, "w") as f:
    json.dump(out, f)
print("TOKENS_DONE", mode, {k: len(v) for k, v in out.items()}, flush=True)
