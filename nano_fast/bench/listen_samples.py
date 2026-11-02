"""Listening samples for the paragraph pause and the long-paragraph join.

Uses serve's paragraph, chunk, unit-seed and pause functions in the same render loop as Server.pieces, but keeps
the audio whole so the chunk-join times can be printed. Fixed seed, so the files can be re-rendered. Writes
16-bit mono WAVs at 24 kHz. Run with the ml venv from the repo root.
"""
import os
import wave

import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/chatnano/nano_fast")
import serve  # noqa: E402

OUT = "/tmp/chatnano-local-check/samples"
SEED = 0

# two paragraphs: the 0.4 s pause after the first is what to listen for
TWO_PARAGRAPHS = (
    "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
    "Would you like me to get some prices for you?\n"
    "Right, let's move on to the second point. We can talk about delivery times once you have chosen a model."
)

# one paragraph of 93 words: chunked into 58 and 35 words, so there is one join to listen for
LONG = (
    "The new model arrived last week, and the team has been testing it against the old one for three days. "
    "Most of the difference is in the first two seconds of each reply, where the old model used to hesitate. "
    "The new one starts speaking almost at once, and it keeps the same tone through the whole sentence. "
    "We also changed how the pauses between sentences are handled, so the voice sounds calmer on long answers. "
    "If you want to hear the difference yourself, ask for a long reply and listen for the joins."
)


def render(server, text, seed=SEED):
    """The serve.Server.pieces path, kept as whole audio so the join positions are known."""
    paras = serve.paragraphs(text) or [text]
    parts, joins, pos, n = [], [], 0, 0
    pause = np.zeros(int(serve.PARAGRAPH_PAUSE_S * server.sr), dtype=np.float32)
    for p_index, p in enumerate(paras):
        cs = serve.chunks(p)
        for c_index, c in enumerate(cs):
            with server.lock, torch.inference_mode():
                torch.manual_seed(serve.unit_seed(seed, n))
                audio = server.fast.generate(c).squeeze(0).float().cpu().numpy()
            if c_index > 0:
                joins.append(pos)  # a join sits at the start of every chunk after the first
            parts.append(audio)
            pos += len(audio)
            n += 1
            if c_index == len(cs) - 1 and p_index < len(paras) - 1:
                parts.append(pause)
                pos += len(pause)
    return np.concatenate(parts), joins


def write_wav(path, audio, sr):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    os.makedirs(OUT, exist_ok=True)
    server = serve.Server()
    for name, text in (("paragraphs", TWO_PARAGRAPHS), ("long_paragraph", LONG)):
        audio, joins = render(server, text)
        path = os.path.join(OUT, f"{name}.wav")
        write_wav(path, audio, server.sr)
        joins_s = [round(j / server.sr, 2) for j in joins]
        print(f"{name}: {len(audio) / server.sr:.2f} s -> {path}; chunk joins at {joins_s} s", flush=True)
        for p in serve.paragraphs(text):
            print(f"  chunk words: {[len(c.split()) for c in serve.chunks(p)]}", flush=True)


if __name__ == "__main__":
    main()
