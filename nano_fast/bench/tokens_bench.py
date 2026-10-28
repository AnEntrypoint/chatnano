import sys, time, json
src, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"{src}/chatterbox/src"); sys.path.insert(0, f"{src}/nano_fast")
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from fast_nano import FastNano
TEXTS = [
    "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. Would you like me to get some prices for you?",
    "Hello there.",
    "This is a considerably longer sentence, written to push the decoder past a few dozen speech tokens, so that the cache bucket and the stop check both get exercised across several boundaries in one run.",
]
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
fast = FastNano(base)
def base_tokens(text):
    ids = base.tokenizer(punc_norm(text), return_tensors="pt", padding=True, truncation=True).input_ids.to(base.device)
    return base.t3.inference_turbo(t3_cond=base.conds.t3, text_tokens=ids).cpu()
res = {"base": [], "fast": [], "tb": [], "tf": []}
for t in TEXTS:
    torch.manual_seed(0); base_tokens(t); torch.manual_seed(0); fast.speech_tokens(t)
for t in TEXTS:
    torch.manual_seed(0); torch.cuda.synchronize(); s = time.perf_counter(); b = base_tokens(t); torch.cuda.synchronize(); res["tb"].append(time.perf_counter() - s)
    torch.manual_seed(0); torch.cuda.synchronize(); s = time.perf_counter(); f = fast.speech_tokens(t).cpu(); torch.cuda.synchronize(); res["tf"].append(time.perf_counter() - s)
    res["base"].append(b.flatten().tolist()); res["fast"].append(f.flatten().tolist())
json.dump(res, open(out, "w"))
print("lens base", [len(x) for x in res["base"]], "fast", [len(x) for x in res["fast"]])
print("times base", [round(x, 3) for x in res["tb"]], "fast", [round(x, 3) for x in res["tf"]])
