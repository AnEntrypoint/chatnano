"""Fixed default (autocast over fp32 S3Gen) against the fp32 S3Gen decoder in the same process, same RNG path,
so the only difference is the decoder precision. Also reports S3Gen parameter count and the memory it costs."""
import sys
import numpy as np
import torch
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
from fast_nano import FastNano  # noqa: E402

TEXT = ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
        "Would you like me to get some prices for you?")
base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
n_s3 = sum(p.numel() for p in base.s3gen.parameters())
print("S3GEN_PARAMS", n_s3, "fp32 MB", round(n_s3 * 4 / 2**20), "fp16 MB", round(n_s3 * 2 / 2**20), flush=True)
fast = FastNano(base)
print("ALLOC_AFTER_INIT_MB", round(torch.cuda.memory_allocated() / 2**20), flush=True)

def gen(on):
    fast.enable_fp16_decoder(on)
    with torch.inference_mode():
        torch.manual_seed(0)
        return fast.generate(TEXT).squeeze(0).float().cpu().numpy()

a = gen(True)    # the default: autocast over fp32 weights
b = gen(False)   # the fp32 decoder
def lsd(x, y):
    n = min(len(x), len(y)); x = torch.tensor(x[:n]); y = torch.tensor(y[:n]); win = torch.hann_window(1024)
    A = torch.stft(x, 1024, 256, window=win, return_complex=True).abs() ** 2
    B = torch.stft(y, 1024, 256, window=win, return_complex=True).abs() ** 2
    return float(torch.sqrt(((10 * torch.log10(A + 1e-10) - 10 * torch.log10(B + 1e-10)) ** 2).mean(0)).mean())
print("SAME_NOISE_LSD default_vs_fp32_decoder", round(lsd(a, b), 3), "dB; lengths", len(a), len(b), flush=True)
