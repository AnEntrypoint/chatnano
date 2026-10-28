import sys, time
src = sys.argv[1]
sys.path.insert(0, f"{src}/chatterbox/src"); sys.path.insert(0, f"{src}/nano_fast")
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from fast_nano import FastNano
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
fast = FastNano(base)
def timed(text):
    torch.manual_seed(0); torch.cuda.synchronize(); s = time.perf_counter(); fast.speech_tokens(text); torch.cuda.synchronize(); return time.perf_counter() - s
timed("Warm up sentence that sets the first graph.")
for t in ["A second, differently sized sentence to test recompilation.", "Third one here, somewhat longer than the others so the length changes again.", "Short."]:
    print(f"RESULT new-length first_call={timed(t):.2f}s", flush=True)
