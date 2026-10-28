import sys, time, statistics
src = sys.argv[1]
sys.path.insert(0, f"{src}/chatterbox/src"); sys.path.insert(0, f"{src}/nano_fast")
import torch
from transformers import StaticCache, DynamicCache
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
import fast_nano
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
t3, tfmr = base.t3, base.t3.tfmr
text = "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. Would you like me to get some prices for you?"
ids = base.tokenizer(punc_norm(text), return_tensors="pt").input_ids.to(base.device)
sst = t3.hp.start_speech_token * torch.ones_like(ids[:, :1])
embeds, _ = t3.prepare_input_embeds(t3_cond=base.conds.t3, text_tokens=ids, speech_tokens=sst, cfg_weight=0.0)
L = embeds.shape[1]; N = 200
def run_eager(cache, label):
    with torch.inference_mode():
        out = tfmr(inputs_embeds=embeds, past_key_values=cache, use_cache=True, cache_position=torch.arange(L, device=base.device))
        tok = t3.speech_emb(torch.tensor([[t3.hp.start_speech_token]], device=base.device))
        ts = []
        for i in range(N):
            torch.cuda.synchronize(); s = time.perf_counter()
            o = tfmr(inputs_embeds=tok, past_key_values=cache, use_cache=True, cache_position=torch.tensor([L + i], device=base.device))
            torch.cuda.synchronize(); ts.append(time.perf_counter() - s)
        print(f"RESULT {label} median_step={statistics.median(ts[20:])*1000:.2f}ms", flush=True)
#run_eager(DynamicCache(), "dynamic-eager")
#run_eager(StaticCache(config=tfmr.config, max_cache_len=L + 1024, device=base.device, dtype=embeds.dtype), "static1280-eager")
#run_eager(StaticCache(config=tfmr.config, max_cache_len=L + 256, device=base.device, dtype=embeds.dtype), "static-tight-eager")
fast = fast_nano.FastNano(base)
def run_compiled(cache_len, label):
    cache = StaticCache(config=tfmr.config, max_cache_len=cache_len, device=base.device, dtype=embeds.dtype)
    with torch.inference_mode():
        fast._step(embeds, cache, torch.arange(L, device=base.device))
        tok = t3.speech_emb(torch.tensor([[t3.hp.start_speech_token]], device=base.device))
        ts = []
        for i in range(N):
            torch.cuda.synchronize(); s = time.perf_counter()
            fast.step(tok, cache, torch.tensor([L + i], device=base.device))
            torch.cuda.synchronize(); ts.append(time.perf_counter() - s)
        print(f"RESULT {label} median_step={statistics.median(ts[20:])*1000:.2f}ms", flush=True)
for k in (8, 256): run_compiled(L + N + k, f"static-pad{k}-compiled")
