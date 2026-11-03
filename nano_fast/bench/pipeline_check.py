import argparse
import json
import os
import sys
import time
import wave

import numpy as np
import torch

LONG = ("The new model arrived last week, and the team has been testing it against the old one for three days. "
        "Most of the difference is in the first two seconds of each reply, where the old model used to hesitate. "
        "The new one starts speaking almost at once, and it keeps the same tone through the whole sentence. "
        "We also changed how the pauses between sentences are handled, so the voice sounds calmer on long answers. "
        "If you want to hear the difference yourself, ask for a long reply and listen for the joins.")
TEXTS = {
    "two_paragraphs": ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
                       "Would you like me to get some prices for you?\n"
                       "Right, let's move on to the second point. We can talk about delivery times once you have "
                       "chosen a model."),
    "long_paragraph": LONG,
    "four_units": LONG + "\n" + LONG,
}


def lsd_db(a, b):
    window = torch.hann_window(1024)
    spec = [torch.stft(x.flatten().float(), 1024, 256, window=window, return_complex=True).abs().pow(2)
            for x in (a, b)]
    da, db = (10 * torch.log10(s + 1e-10) for s in spec)
    return float(((da - db) ** 2).mean(dim=0).sqrt().mean())


def write_wav(path, audio, sr):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def compare(dir_a, dir_b):
    rows = {}
    by_key = {name.split("_", 1)[1]: name for name in os.listdir(dir_b) if name.endswith(".npy")}
    for name in sorted(os.listdir(dir_a)):
        if not name.endswith(".npy") or name.split("_", 1)[1] not in by_key:
            continue
        a = np.load(os.path.join(dir_a, name))
        b = np.load(os.path.join(dir_b, by_key[name.split("_", 1)[1]]))
        n = min(len(a), len(b))
        rows[name.split("_", 1)[1]] = {
            "len_a": int(len(a)), "len_b": int(len(b)),
            "max_abs_diff": float(np.abs(a[:n] - b[:n]).max()),
            "lsd_db": lsd_db(torch.from_numpy(a[:n]), torch.from_numpy(b[:n])),
        }
    print(json.dumps(rows, indent=1))


def render(server, text, seed):
    t0 = time.perf_counter()
    first = None
    parts = []
    for block in server.pieces(text, seed):
        if first is None:
            first = time.perf_counter() - t0
        parts.append(block)
    total = time.perf_counter() - t0
    return np.concatenate(parts), first, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--tag")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    sys.path.insert(0, os.path.join(args.root, "nano_fast"))
    import serve
    os.makedirs(args.out_dir, exist_ok=True)
    server = serve.Server()
    seeds = [int(s) for s in args.seeds.split(",")]
    for text in TEXTS.values():
        for seed in seeds:
            render(server, text, seed)
    times = {}
    for r in range(args.repeats):
        for name, text in TEXTS.items():
            for seed in seeds:
                audio, first, total = render(server, text, seed)
                row = times.setdefault(f"{name}/{seed}", {"first": [], "total": []})
                row["first"].append(first)
                row["total"].append(total)
                if r == 0:
                    stem = os.path.join(args.out_dir, f"{args.tag}_{name}_{seed}")
                    np.save(stem + ".npy", audio)
                    write_wav(stem + ".wav", audio, server.sr)
    summary = {k: {"first_median_s": round(float(np.median(v["first"])), 3),
                   "total_median_s": round(float(np.median(v["total"])), 3),
                   "total_all_s": [round(t, 3) for t in v["total"]]} for k, v in times.items()}
    with open(os.path.join(args.out_dir, f"{args.tag}_timing.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
