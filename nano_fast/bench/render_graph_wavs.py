"""Renders the texts with graphs off and on, same seeds, to WAV (16-bit, 24 kHz) for the transcript check.
Usage: python3 -I render_graph_wavs.py OUT_DIR
"""
import os
import sys
import wave

import numpy as np
import torch

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
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


def write(path, wav, sr):
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm)


base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
fast = FastNano(base)
fast.warm_decoder_graphs()
with torch.inference_mode():
    for name, text in TEXTS.items():
        for seed in (0, 1):
            for mode, on in (("eager", False), ("graph", True)):
                fast.enable_decoder_graphs(on)
                torch.manual_seed(seed)
                wav = fast.generate(text).float().flatten().cpu().numpy()
                write(os.path.join(out_dir, f"{name}_s{seed}_{mode}.wav"), wav, base.sr)
                print("wrote", name, seed, mode, len(wav), flush=True)
fast.enable_decoder_graphs(True)
