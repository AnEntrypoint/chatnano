"""Where do the paragraph seed-1 wav differences come from? Hooks the HiFT call inside generate() and records the mel
it receives and the generator state at that point, for graphs off and on, same seed and text.
Usage: python3 -I decoder_graph_hift.py OUT.json
"""
import hashlib
import json
import sys

import torch

out_path = sys.argv[1]
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
from fast_nano import FastNano  # noqa: E402

TEXT = ("The new model arrived last week, and the team has been testing it against the old one for "
        "three days. Most of the difference is in the first two seconds of each reply.")


def digest(t):
    return hashlib.md5(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:10]


base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
fast = FastNano(base)
fast.warm_decoder_graphs()
s3 = base.s3gen
seen = {}
orig = s3.mel2wav.inference
state = {"label": None}


def hooked(speech_feat, cache_source=None):
    seen[state["label"]] = {"mel": speech_feat.detach().clone(), "rng": digest(torch.cuda.get_rng_state()),
                            "mel_digest": digest(speech_feat)}
    return orig(speech_feat=speech_feat, cache_source=cache_source)


s3.mel2wav.inference = hooked
wavs = {}
with torch.inference_mode():
    for label, on in [("eager", False), ("graph", True), ("eager_again", False)]:
        fast.enable_decoder_graphs(on)
        state["label"] = label
        torch.manual_seed(1)
        wavs[label] = fast.generate(TEXT).float().flatten().cpu()
fast.enable_decoder_graphs(True)
out = {"rng_before_hift": {k: v["rng"] for k, v in seen.items()},
       "mel_digest": {k: v["mel_digest"] for k, v in seen.items()},
       "mel_max_abs_eager_vs_graph": float((seen["eager"]["mel"] - seen["graph"]["mel"]).abs().max()),
       "mel_max_abs_eager_vs_again": float((seen["eager"]["mel"] - seen["eager_again"]["mel"]).abs().max()),
       "wav_max_abs_eager_vs_graph": float((wavs["eager"] - wavs["graph"]).abs().max()),
       "wav_max_abs_eager_vs_again": float((wavs["eager"] - wavs["eager_again"]).abs().max())}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print("RESULT", json.dumps(out), flush=True)
