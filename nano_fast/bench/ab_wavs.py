# Writes fp32 and fp16-S3Gen WAVs for the same text and seed, for a listening A/B.
import sys, numpy as np, soundfile as sf, torch
sys.path.insert(0, "/workspace/chatnano/chatterbox/src"); sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS
import fast_nano
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
fast = fast_nano.FastNano(base)
TEXTS = {
    "a_hilarious": "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. Would you like me to get some prices for you?",
    "b_long": "This is a considerably longer sentence, written to push the decoder past a few dozen speech tokens, so that the cache bucket and the stop check both get exercised.",
    "c_hello": "Hello there, how are you doing today?",
}
out = "/workspace/chatnano/nano_fast/bench/ab"
for name, text in TEXTS.items():
    for label, on in (("fp32", False), ("fp16", True)):
        fast.enable_fp16_decoder(on)
        torch.manual_seed(0)
        w = fast.generate(text).squeeze(0).float().cpu().numpy()
        sf.write(f"{out}/{name}_{label}.wav", w, base.sr)
        print(f"wrote {name}_{label}.wav {len(w)/base.sr:.2f}s", flush=True)
fast.enable_fp16_decoder(False)
