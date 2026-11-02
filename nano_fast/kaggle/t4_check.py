"""Targeted T4 check, run as a Kaggle script kernel (or locally with the environment knobs below).

tokens: five fresh processes, each printing the speech-token count and hash for the short and long texts at seed 0.
  Kernel v14 gave 162 tokens for the short text in every process but one (179). Fresh processes on this card tell
  whether the count varies between processes, which is the open question for the T4 anomaly.
audio: reference generate() against FastNano with the decoder graphs off and on, two seeds, three texts. The waves
  are transcribed with whisper-tiny and compared by word similarity to the text, and by log-spectral distance.
Writes OUT_DIR/t4_check.json and OUT_DIR/wav/*.wav. Knobs: CHATTERBOX_SRC, NANO_FAST_DIR, CHATNANO_WEIGHTS (a local
weights directory; if absent the hub is used), OUT_DIR, ON_KAGGLE=1 (installs the notebook's pip set first).
Usage: python run_t4_check.py [parent|tokens|audio]
"""
import difflib
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import wave

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
WEIGHTS = os.environ.get("CHATNANO_WEIGHTS", "")
TOKEN_PROCESSES = 5
SEEDS = (0, 1)
TEXTS = {
    "short": ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
              "Would you like me to get some prices for you?"),
    "long": ("This is a considerably longer sentence, written to push the decoder past a few dozen speech "
             "tokens, so that the cache bucket and the stop check both get exercised across several "
             "boundaries in one run."),
    "paragraph": ("The new model arrived last week, and the team has been testing it against the old one for "
                  "three days. Most of the difference is in the first two seconds of each reply."),
}
PIP = ["diffusers==0.29.0", "conformer==0.3.2", "s3tokenizer", "pyloudnorm", "omegaconf", "resemble-perth", "torchao"]


def locate(name, marker):
    # the mount path is not fixed (the notebook locates its suite the same way); prefer the dataset named `name`
    import glob
    hits = sorted(glob.glob(f"/kaggle/input/**/{marker}", recursive=True), key=lambda p: (name not in p, len(p)))
    if not hits:
        raise SystemExit(f"{marker} not found under /kaggle/input")
    return hits[0]


def source_dirs():
    if "CHATTERBOX_SRC" in os.environ:
        return os.environ["CHATTERBOX_SRC"], os.environ["NANO_FAST_DIR"]
    src = os.path.dirname(os.path.dirname(locate("chatnano-src", "chatterbox/src/chatterbox/tts_turbo.py")))
    fast = os.path.dirname(locate("chatnano-nano-fast", "fast_nano.py"))
    return src, fast


def load_model():
    src, fast = source_dirs()
    sys.path.insert(0, src)
    sys.path.insert(0, fast)
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    if WEIGHTS and os.path.isdir(WEIGHTS):
        return ChatterboxTurboTTS.from_local(WEIGHTS, device="cuda", nano=True)
    return ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def similarity(reference, hypothesis):
    return round(difflib.SequenceMatcher(None, words(reference), words(hypothesis)).ratio(), 3)


def lsd_db(a, b):
    import torch
    window = torch.hann_window(1024)
    spec = [torch.stft(torch.as_tensor(x).float().flatten(), 1024, 256, window=window, return_complex=True)
            .abs().pow(2) for x in (a, b)]
    da, db = (10 * torch.log10(s + 1e-10) for s in spec)
    return round(float(((da - db) ** 2).mean(dim=0).sqrt().mean()), 4)


def write_wav(path, wav, sr):
    import numpy as np
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm)


def child_tokens():
    import torch
    model = load_model()
    from fast_nano import FastNano
    fast = FastNano(model)
    result = {}
    with torch.inference_mode():
        for name in ("short", "long"):
            torch.manual_seed(0)
            toks = fast.speech_tokens(TEXTS[name]).flatten().tolist()
            result[name] = {"tokens": len(toks), "md5": hashlib.md5(json.dumps(toks).encode()).hexdigest()[:10]}
    print("RESULT " + json.dumps(result), flush=True)


def child_audio():
    import numpy as np
    import torch
    from scipy.signal import resample_poly
    from transformers import pipeline
    wav_dir = os.path.join(OUT, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    model = load_model()
    sr = model.sr
    refs = {}
    with torch.inference_mode():  # the reference first: FastNano halves this model's T3 in place
        for name, text in TEXTS.items():
            for seed in SEEDS:
                torch.manual_seed(seed)
                refs[(name, seed)] = model.generate(text).float().flatten().cpu().numpy()
    from fast_nano import FastNano
    fast = FastNano(model)
    t = time.perf_counter()
    warmed = fast.warm_decoder_graphs()
    warm_s = round(time.perf_counter() - t, 1)
    waves = {}
    with torch.inference_mode():
        for name, text in TEXTS.items():
            for seed in SEEDS:
                for mode, on in (("eager", False), ("graph", True), ("eager_again", False)):
                    fast.enable_decoder_graphs(on)
                    torch.manual_seed(seed)
                    waves[(name, seed, mode)] = fast.generate(text).float().flatten().cpu().numpy()
                waves[(name, seed, "reference")] = refs[(name, seed)]
        timing = {}
        for name, text in TEXTS.items():
            row = {}
            for mode, on in (("eager", False), ("graph", True)):
                fast.enable_decoder_graphs(on)
                times = []
                for _ in range(3):
                    torch.manual_seed(0)
                    t = time.perf_counter()
                    fast.generate(text)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t) * 1000)
                row[mode] = round(statistics.median(times), 1)
            timing[name] = row
    fast.enable_decoder_graphs(True)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0)
    result = {"warm_graphs": warmed, "warm_seconds": warm_s, "generate_ms": timing, "waves": {}}
    for (name, seed, mode), wav in waves.items():
        key = f"{name}/{seed}/{mode}"
        write_wav(os.path.join(wav_dir, f"{name}_s{seed}_{mode}.wav"), wav, sr)
        x16 = resample_poly(wav.astype(np.float32), 2, 3)
        hyp = asr({"raw": x16, "sampling_rate": 16000}, generate_kwargs={"language": "en", "task": "transcribe"})["text"]
        row = {"samples": int(wav.shape[-1]), "transcript": hyp.strip(),
               "similarity_to_text": similarity(TEXTS[name], hyp)}
        if mode != "reference":
            row["lsd_db_vs_reference"] = lsd_db(waves[(name, seed, "reference")], wav)
        if mode in ("graph", "eager_again"):  # eager_again is the control: eager against eager, same seed
            row["lsd_db_vs_eager"] = lsd_db(waves[(name, seed, "eager")], wav)
        result["waves"][key] = row
    print("RESULT " + json.dumps(result), flush=True)


def parent():
    import glob
    print("INPUTS", sorted(glob.glob("/kaggle/input/*")), "SOURCE_DIRS", source_dirs(), flush=True)
    if os.path.isdir("/kaggle/input") or os.environ.get("ON_KAGGLE") == "1":
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--retries", "2", "--timeout", "60"]
                       + PIP, check=True, timeout=900)
    os.makedirs(OUT, exist_ok=True)
    result = {"tokens": [], "audio": None}
    for i in range(TOKEN_PROCESSES):
        r = subprocess.run([sys.executable, __file__, "tokens"], capture_output=True, text=True, timeout=1500)
        lines = [line for line in r.stdout.splitlines() if line.startswith("RESULT ")]
        result["tokens"].append(json.loads(lines[-1][7:]) if lines else {"error": r.stderr[-1500:]})
        print("tokens process", i, result["tokens"][-1], flush=True)
    r = subprocess.run([sys.executable, __file__, "audio"], capture_output=True, text=True, timeout=3600)
    lines = [line for line in r.stdout.splitlines() if line.startswith("RESULT ")]
    result["audio"] = json.loads(lines[-1][7:]) if lines else {"error": r.stderr[-3000:]}
    with open(os.path.join(OUT, "t4_check.json"), "w") as f:
        json.dump(result, f, indent=1)
    print("DONE", json.dumps({"tokens": [t.get("short") for t in result["tokens"]]}), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "parent"
    {"parent": parent, "tokens": child_tokens, "audio": child_audio}[mode]()
