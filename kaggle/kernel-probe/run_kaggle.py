import glob
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

MODE = "probe"

CONFIGS = {
    "probe": dict(corpus_limit=400, eval_count=40, ref_count=8, steps=30, batch=4, ckpt_every=30, keep_last=1, gen_count=4, wer_count=0),
    "pilot": dict(corpus_limit=0, eval_count=200, ref_count=32, steps=2000, batch=8, ckpt_every=250, keep_last=2, gen_count=20, wer_count=100),
}

CHATTERBOX_COMMIT = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
DEPENDENCIES = [
    "librosa==0.11.0",
    "transformers==5.2.0",
    "diffusers==0.29.0",
    "conformer==0.3.2",
    "safetensors==0.5.3",
    "pyloudnorm",
    "omegaconf",
    "s3tokenizer",
    "resemble-perth",
    "spacy-pkuseg",
    "pykakasi==2.3.0",
    "soundfile",
    "jiwer",
]
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

WORK = Path("/tmp/chatnano")
OUT = Path("/kaggle/working")
CFG = CONFIGS[MODE]


def stream(cmd, log_path, cwd=None):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        code = process.wait()
    if code != 0:
        raise SystemExit(f"command failed with exit code {code}: {cmd[0]} ...")


def load_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded from Kaggle secrets", flush=True)
    except Exception as error:
        print(f"no HF_TOKEN secret ({type(error).__name__}); downloads are unauthenticated", flush=True)


def install_dependencies(log_path):
    stream(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         f"git+https://github.com/resemble-ai/chatterbox.git@{CHATTERBOX_COMMIT}"],
        log_path,
    )
    stream([sys.executable, "-m", "pip", "install", "-q", *DEPENDENCIES], log_path)


def fetch_ljspeech():
    WORK.mkdir(parents=True, exist_ok=True)
    corpus = WORK / "LJSpeech-1.1"
    if (corpus / "metadata.csv").exists():
        return corpus
    archive = WORK / "LJSpeech-1.1.tar.bz2"
    print("downloading LJSpeech (about 2.7 GB)", flush=True)
    urllib.request.urlretrieve(LJSPEECH_URL, archive)
    with tarfile.open(archive, "r:bz2") as handle:
        handle.extractall(WORK)
    archive.unlink()
    return corpus


def main():
    log_path = OUT / "run.log"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"mode={MODE} config={CFG}", flush=True)
    load_hf_token()
    install_dependencies(log_path)

    matches = glob.glob("/kaggle/input/**/nano_ternary.py", recursive=True)
    if not matches:
        raise SystemExit("ternary source dataset not mounted: no nano_ternary.py under /kaggle/input")
    src = Path(matches[0]).parent
    print(f"source directory: {src}", flush=True)

    corpus = fetch_ljspeech()
    data = WORK / "data"
    if data.exists():
        shutil.rmtree(data)
    stream(
        [sys.executable, src / "prepare_tokens.py", "--corpus", corpus, "--out", data, "--device", "cuda",
         "--eval-count", CFG["eval_count"], "--ref-count", CFG["ref_count"], "--limit", CFG["corpus_limit"]],
        log_path,
        cwd=src,
    )
    shutil.copy(data / "prepare_summary.json", OUT / "prepare_summary.json")

    run_dir = OUT / "run"
    stream(
        [sys.executable, src / "train_nano_ternary.py", "--data", data, "--out", run_dir, "--device", "cuda",
         "--steps", CFG["steps"], "--batch", CFG["batch"], "--ckpt-every", CFG["ckpt_every"],
         "--keep-last", CFG["keep_last"], "--eval-count", CFG["eval_count"], "--gen-count", CFG["gen_count"],
         "--wer-count", CFG["wer_count"]],
        log_path,
        cwd=src,
    )

    for old in run_dir.glob("step_*"):
        shutil.rmtree(old)
    resume = run_dir / "resume.pt"
    if resume.exists():
        resume.unlink()
    print("kept in /kaggle/working:", sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()), flush=True)


if __name__ == "__main__":
    main()
