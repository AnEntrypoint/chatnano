"""Control for the decoder graphs: generate() for one text and seed, eager twice and graphs twice, with the tokens
compared too. Shows whether the wav differences between modes are larger than the run-to-run difference of eager.
Usage: python3 -I decoder_graph_control.py OUT.json TEXT_NAME SEED
"""
import json
import sys

import torch

out_path, name, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
from fast_nano import FastNano  # noqa: E402

TEXTS = {
    "short": ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
              "Would you like me to get some prices for you?"),
    "paragraph": ("The new model arrived last week, and the team has been testing it against the old one for "
                  "three days. Most of the difference is in the first two seconds of each reply."),
}
base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
fast = FastNano(base)
fast.warm_decoder_graphs()
runs = {}
with torch.inference_mode():
    for label, on in [("eager_a", False), ("eager_b", False), ("graph_a", True), ("graph_b", True)]:
        fast.enable_decoder_graphs(on)
        torch.manual_seed(seed)
        t = fast.speech_tokens(TEXTS[name]).flatten().tolist()
        torch.manual_seed(seed)
        runs[label] = (t, fast.generate(TEXTS[name]).float().flatten().cpu())
fast.enable_decoder_graphs(True)
out = {"tokens": {k: len(v[0]) for k, v in runs.items()},
       "tokens_equal": {"eager_a_vs_b": runs["eager_a"][0] == runs["eager_b"][0],
                        "graph_a_vs_b": runs["graph_a"][0] == runs["graph_b"][0],
                        "eager_a_vs_graph_a": runs["eager_a"][0] == runs["graph_a"][0]},
       "wav_max_abs": {"eager_a_vs_b": float((runs["eager_a"][1] - runs["eager_b"][1]).abs().max()),
                       "graph_a_vs_b": float((runs["graph_a"][1] - runs["graph_b"][1]).abs().max()),
                       "eager_a_vs_graph_a": float((runs["eager_a"][1] - runs["graph_a"][1]).abs().max())}}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print("RESULT", json.dumps(out), flush=True)
