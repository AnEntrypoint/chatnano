"""Per-step logits hashes for one seeded generation, for comparing processes or compile states.

Run it in fresh processes, warm (default inductor cache) and cold (TORCHINDUCTOR_CACHE_DIR set to an empty
directory), then compare the step lists: the first differing step shows where a run's numbers first change.
Usage: python3 -I decode_step_logits_probe.py OUT.json
"""
import hashlib
import json
import os
import sys

import torch
import torch._inductor.config as icfg

# PROBE_ATEN_ONLY=1: autotune among cuBLAS (ATen) algorithms only; PROBE_NOAUTOTUNE=1: autotuning off (see
# compile_mode_e2e.py). Unset, the probe measures the default max-autotune compile.
if os.environ.get("PROBE_ATEN_ONLY"):
    icfg.max_autotune_gemm_backends = "ATEN"

sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
from fast_nano import FastNano  # noqa: E402

TEXT = ("The new model arrived last week, and the team has been testing it against the old one for "
        "three days. Most of the difference is in the first two seconds of each reply.")
out_path = sys.argv[1]
if os.environ.get("PROBE_NOAUTOTUNE"):
    icfg.pattern_matcher = False
    _compile = torch.compile
    torch.compile = lambda f, **kw: _compile(f, **{**{k: v for k, v in kw.items() if k != "mode"}, "options": {"triton.cudagraphs": True, "max_autotune": False, "max_autotune_gemm": False}})
base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
fast = FastNano(base)
hashes = []
orig = fast._sample


def hooked(procs, history, logits):
    hashes.append(hashlib.sha1(logits.float().cpu().numpy().tobytes()).hexdigest()[:16])
    return orig(procs, history, logits)


with torch.inference_mode():
    torch.manual_seed(2)
    fast.speech_tokens(TEXT)  # warm-up: compiles every shape
    hashes.clear()
    fast._sample = hooked
    torch.manual_seed(2)
    tokens = fast.speech_tokens(TEXT).flatten().tolist()
fast._sample = orig
with open(out_path, "w") as f:
    json.dump({"tokens": tokens, "logit_hashes": hashes}, f)
print("PROBE", len(tokens), len(hashes), flush=True)
