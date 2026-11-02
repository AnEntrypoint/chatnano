"""End to end: the compiled T3 token loop with eager decode attention or the Triton kernel, one mode per process.

The compiled step captures its attention at trace time, so each mode runs in its own process. Writes speech tokens
and token-loop medians for the two bench texts to OUT.json; compare the JSON files of the two modes.
Usage: python3 -I decode_attn_e2e.py eager|triton OUT.json
"""
import json
import statistics
import sys
import time

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
}

if mode == "triton":
    DA.install()
elif mode != "eager":
    raise SystemExit(f"unknown mode {mode!r}")

base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
fast = FastNano(base)
result = {"mode": mode, "tokens": {}, "token_loop_ms": {}, "first_call_s": {}}


def tokens(text):
    torch.manual_seed(0)
    return fast.speech_tokens(text).flatten().tolist()


with torch.inference_mode():
    for name, text in TEXTS.items():
        t0 = time.perf_counter()
        result["tokens"][name] = tokens(text)  # includes compilation
        result["first_call_s"][name] = round(time.perf_counter() - t0, 2)
        tokens(text)
        torch.cuda.synchronize()
        ts = []
        for _ in range(7):
            t = time.perf_counter()
            tokens(text)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        result["token_loop_ms"][name] = round(statistics.median(ts) * 1000, 1)
with open(out_path, "w") as f:
    json.dump(result, f)
print("RESULT", json.dumps({k: v for k, v in result.items() if k != "tokens"}), flush=True)
