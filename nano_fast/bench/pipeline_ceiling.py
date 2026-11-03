import argparse
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import serve  # noqa: E402
from listen_samples import LONG, TWO_PARAGRAPHS  # noqa: E402

TEXTS = {"two_paragraphs": TWO_PARAGRAPHS, "long_paragraph": LONG, "four_units": LONG + "\n" + LONG}


def units_of(text):
    out = []
    for paragraph in serve.paragraphs(text) or [text]:
        out.extend(serve.chunks(paragraph))
    return out


def timed_unit(server, unit, seed, n):
    fast = server.fast
    span = {}
    speech_tokens = fast.speech_tokens

    def timed_speech_tokens(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = speech_tokens(*args, **kwargs)
        torch.cuda.synchronize()
        span["t3"] = time.perf_counter() - t0
        return out

    fast.speech_tokens = timed_speech_tokens
    try:
        with server.lock, torch.inference_mode():
            torch.manual_seed(serve.unit_seed(seed, n))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fast.generate(unit).squeeze(0).float().cpu().numpy()
            total = time.perf_counter() - t0
    finally:
        del fast.speech_tokens
    return span["t3"], total - span["t3"]


def ceiling(stages):
    saving = sum(min(stages[k][1], stages[k + 1][0]) for k in range(len(stages) - 1))
    total = sum(t3 + voc for t3, voc in stages)
    return saving, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    server = serve.Server()
    plans = {name: units_of(text) for name, text in TEXTS.items()}
    for units in plans.values():
        for n, unit in enumerate(units):
            timed_unit(server, unit, args.seed, n)
    runs = {name: [] for name in plans}
    for _ in range(args.repeats):
        for name, units in plans.items():
            runs[name].append([timed_unit(server, unit, args.seed, n) for n, unit in enumerate(units)])
    report = {}
    for name, rep in runs.items():
        ratios = []
        totals = []
        for stages in rep:
            saving, total = ceiling(stages)
            ratios.append(saving / total)
            totals.append(total)
        units = len(rep[0])
        report[name] = {
            "units": units,
            "ceiling_median": round(statistics.median(ratios), 4),
            "ceiling_all": [round(r, 4) for r in ratios],
            "total_median_s": round(statistics.median(totals), 3),
            "t3_median_s": [round(statistics.median(rep[r][k][0] for r in range(len(rep))), 3) for k in range(units)],
            "voc_median_s": [round(statistics.median(rep[r][k][1] for r in range(len(rep))), 3) for k in range(units)],
        }
    report["repeats"] = args.repeats
    report["seed"] = args.seed
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
