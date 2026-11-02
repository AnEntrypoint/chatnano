"""Intelligibility check: Whisper-tiny transcripts of WAV files (16-bit or float32 mono), linear resample to 16 kHz.
Uses openai/whisper-tiny from the Hugging Face cache. Usage: python3 -I asr_check.py a.wav [b.wav ...]"""
import sys, wave
import numpy as np
import torch
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0 if torch.cuda.is_available() else -1)

def read(path):
    with wave.open(path) as w:
        sr = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2" if sw == 2 else "<f4").astype(np.float32)
    if sw == 2:
        x /= 32767.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr

for path in sys.argv[1:]:
    x, sr = read(path)
    t = np.linspace(0, len(x) / sr, int(len(x) * 16000 / sr), endpoint=False)
    y = np.interp(t, np.arange(len(x)) / sr, x).astype(np.float32)
    out = asr({"raw": y, "sampling_rate": 16000}, generate_kwargs={"language": "en", "task": "transcribe"})
    print(f"{path.split('/')[-1]} ({len(x)/sr:.1f}s): {out['text'].strip()}", flush=True)
