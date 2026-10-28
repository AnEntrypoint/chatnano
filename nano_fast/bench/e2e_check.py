import sys, time, numpy as np, torch
sys.path.insert(0, "/workspace/chatnano/chatterbox/src"); sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS
import fast_nano
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
fast = fast_nano.FastNano(base)
TEXT = "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. Would you like me to get some prices for you?"
fast.generate(TEXT); torch.cuda.synchronize()
torch.manual_seed(0); s = time.perf_counter(); w = fast.generate(TEXT); torch.cuda.synchronize(); el = time.perf_counter()-s
wav = w.squeeze(0).numpy()
print(f"RESULT e2e fast generate {el:.2f}s audio={wav.shape[-1]/base.sr:.2f}s wm_msg={base.watermarker.get_watermark(wav, sample_rate=base.sr)}", flush=True)
