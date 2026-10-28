import sys, time, statistics
src = sys.argv[1]
sys.path.insert(0, f"{src}/chatterbox/src"); sys.path.insert(0, f"{src}/nano_fast")
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from fast_nano import FastNano
TEXTS = [
    "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. Would you like me to get some prices for you?",
    "This is a considerably longer sentence, written to push the decoder past a few dozen speech tokens, so that the cache bucket and the stop check both get exercised across several boundaries in one run.",
]
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
fast = FastNano(base)
def base_tokens(text):
    ids = base.tokenizer(punc_norm(text), return_tensors="pt", padding=True, truncation=True).input_ids.to(base.device)
    return base.t3.inference_turbo(t3_cond=base.conds.t3, text_tokens=ids)
def t(fn, text, n=5):
    fn(text)
    ts = []
    for _ in range(n):
        torch.manual_seed(0); torch.cuda.synchronize(); s = time.perf_counter(); fn(text); torch.cuda.synchronize(); ts.append(time.perf_counter() - s)
    return statistics.median(ts)
for i, text in enumerate(TEXTS):
    b = t(base_tokens, text); f = t(fast.speech_tokens, text)
    print(f"RESULT text{i} base_median={b:.3f}s fast_median={f:.3f}s", flush=True)
