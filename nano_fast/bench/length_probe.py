# Where does the reply stop early? Render one sentence repeated k times and compare the audio length
# with k times the one-sentence length. A ratio well below 1 means the model ended the reply early.
import sys, torch
sys.path.insert(0, "/workspace/chatnano/chatterbox/src"); sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS
from fast_nano import FastNano
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
fast = FastNano(base)
SENT = "This sentence is part of a long reply that the server should either speak in full or refuse."
def seconds(text, seed=0):
    torch.manual_seed(seed)
    with torch.inference_mode():
        w = fast.generate(text)
    return w.shape[-1] / base.sr
base_s = seconds(SENT)
print(f"RESULT one sentence = {base_s:.2f} s", flush=True)
for k in (2, 4, 8, 12, 16, 24, 32, 40):
    text = " ".join([SENT] * k)
    sec = seconds(text)
    print(f"RESULT k={k:2d} words={len(text.split()):4d} audio={sec:6.2f} s expected={k*base_s:6.2f} s ratio={sec/(k*base_s):.2f}", flush=True)
