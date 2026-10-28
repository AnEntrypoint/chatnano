"""Generate per-character voice lines with Chatterbox-Nano.

Manifest-driven, so it runs the moment oscraft has an audio path and a roster.

manifest.json:
{
  "out_dir": "voices/out",
  "characters": [
    {"name": "axe", "ref": "voices/refs/axe.wav", "lines": ["Culling blade!", "Axe will decide."]}
  ]
}

Usage:
  python3 -I voices/make_voices.py voices/manifest.json [--fast|--baseline]

Notes:
- A reference clip must be >5 s (Chatterbox asserts this).
- Output is 24 kHz mono WAV.
- Reference clips must be ours or clearly licensed; do not use game rips.
"""
import argparse, json, os, sys, time
import torch


def load_model(device="cuda"):
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    return ChatterboxTurboTTS.from_pretrained(device=device, nano=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--fast", action="store_true", default=True,
                    help="use the static-cache + CUDA-graph path (default)")
    ap.add_argument("--baseline", dest="fast", action="store_false")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.manifest) as f:
        man = json.load(f)
    out_root = man.get("out_dir", "voices/out")
    os.makedirs(out_root, exist_ok=True)

    model = load_model(args.device)
    fast = None
    if args.fast:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, "/workspace/chatnano/nano_fast")
        from fast_nano import FastNano
        fast = FastNano(model)

    total_lines = 0
    for ch in man["characters"]:
        name = ch["name"]
        ref = ch.get("ref")
        lines = ch["lines"]
        cdir = os.path.join(out_root, name)
        os.makedirs(cdir, exist_ok=True)

        if ref:
            model.prepare_conditionals(ref)
        elif model.conds is None:
            raise SystemExit(f"{name}: no ref clip and model has no default conditionals")

        # warm the compiled path once per character
        if fast is not None:
            fast.generate(lines[0])

        t0 = time.perf_counter()
        for i, line in enumerate(lines):
            with torch.inference_mode():
                wav = fast.generate(line) if fast is not None else model.generate(line)
            import soundfile as sf
            sf.write(os.path.join(cdir, f"{i:03d}.wav"),
                     wav.squeeze().float().cpu().numpy(), model.sr)
        dt = time.perf_counter() - t0
        total_lines += len(lines)
        print(f"RESULT {name}: {len(lines)} lines in {dt:.2f}s "
              f"({dt/max(len(lines),1):.2f}s/line, ref={'yes' if ref else 'default'})")
    print(f"RESULT total: {total_lines} lines -> {out_root}")


if __name__ == "__main__":
    main()
