"""Token loop and tokens: max-autotune (the default) against the compile with autotuning off, one mode per process.

Autotuning off keeps cudagraphs on and disables max_autotune and max_autotune_gemm, through torch.compile options.
The inductor attention-fusion pass (pattern_matcher) is turned off too: it fails on FastNano's eager attention,
whose divisor is a tensor, and it is the only other change. The compile options cannot be combined with a mode,
so the switch replaces torch.compile for this process. Writes tokens, token-loop medians and the first call
(compile) time to OUT.json. Usage: python3 -I compile_mode_e2e.py max-autotune|autotune-off OUT.json
"""
import json
import statistics
import sys
import time

import torch

mode, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")

if mode == "autotune-off":
    import torch._inductor.config as icfg
    icfg.pattern_matcher = False
    _compile = torch.compile

    def _compile_without_autotune(f, **kw):
        kw = {k: v for k, v in kw.items() if k != "mode"}
        kw["options"] = {"triton.cudagraphs": True, "max_autotune": False, "max_autotune_gemm": False}
        return _compile(f, **kw)

    torch.compile = _compile_without_autotune
elif mode != "max-autotune":
    raise SystemExit(f"unknown mode {mode!r}")

from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
from fast_nano import FastNano  # noqa: E402

TEXTS = {
    "short": ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
              "Would you like me to get some prices for you?"),
    "long": ("This is a considerably longer sentence, written to push the decoder past a few dozen speech "
             "tokens, so that the cache bucket and the stop check both get exercised across several "
             "boundaries in one run."),
}

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
