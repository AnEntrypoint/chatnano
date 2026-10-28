import sys, time, numpy as np, torch
sys.path.insert(0, "/workspace/chatnano/chatterbox/src"); sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
import fast_nano
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
fast = fast_nano.FastNano(base)
REF = "/workspace/chatnano/nano_fast/fast_nano_out.wav"
TEXT = "Hello there, how are you doing today?"
def base_toks():
    ids = base.tokenizer(punc_norm(TEXT), return_tensors="pt", padding=True, truncation=True).input_ids.to(base.device)
    return base.t3.inference_turbo(t3_cond=base.conds.t3, text_tokens=ids).cpu()
torch.manual_seed(0); default_fast = fast.speech_tokens(TEXT).cpu()
torch.manual_seed(0); default_base = base_toks()
print("RESULT default identical", bool(torch.equal(default_fast, default_base)), flush=True)
for label in ("first", "second"):
    s = time.perf_counter(); fast.use_reference(REF); torch.cuda.synchronize()
    print(f"RESULT use_reference {label}={time.perf_counter()-s:.3f}s", flush=True)
torch.manual_seed(0); ref_fast = fast.speech_tokens(TEXT).cpu()
base.prepare_conditionals(REF); torch.manual_seed(0); ref_base = base_toks()
print("RESULT reference identical", bool(torch.equal(ref_fast, ref_base)), flush=True)
fast.use_reference(None)
torch.manual_seed(0); back = fast.speech_tokens(TEXT).cpu()
print("RESULT switch-back identical", bool(torch.equal(back, default_base)), flush=True)
