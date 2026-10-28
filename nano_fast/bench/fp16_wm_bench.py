import sys, time, statistics, copy
src = "/workspace/chatnano"
sys.path.insert(0, f"{src}/chatterbox/src"); sys.path.insert(0, f"{src}/nano_fast")
import numpy as np, torch, librosa
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
import fast_nano
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
TEXTS = ["Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. Would you like me to get some prices for you?",
         "This is a considerably longer sentence, written to push the decoder past a few dozen speech tokens, so that the cache bucket and the stop check both get exercised.",
         "Hello there, how are you doing today?"]
def tok_fn(text, t3):
    ids = base.tokenizer(punc_norm(text), return_tensors="pt", padding=True, truncation=True).input_ids.to(base.device)
    return t3.inference_turbo(t3_cond=base.conds.t3, text_tokens=ids)
def med(fn, n=3):
    fn(); ts=[]
    for _ in range(n):
        torch.cuda.synchronize(); s=time.perf_counter(); r=fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-s)
    return statistics.median(ts), r
# ---- watermark CPU vs GPU
wav = base.generate(TEXTS[0]).squeeze(0).cpu().numpy()
wm = base.watermarker
tc, out_cpu = med(lambda: wm.apply_watermark(wav, sample_rate=base.sr), 3)
wm.perth_net.to("cuda")
tg, out_gpu = med(lambda: wm.apply_watermark(wav, sample_rate=base.sr), 3)
print(f"RESULT watermark cpu={tc*1000:.0f}ms gpu={tg*1000:.0f}ms maxdiff={np.max(np.abs(out_cpu-out_gpu)):.2e} "
      f"msg_cpu={wm.get_watermark(out_cpu, sample_rate=base.sr)} msg_gpu={wm.get_watermark(out_gpu, sample_rate=base.sr)}", flush=True)
wm.perth_net.to("cpu")
# ---- fp16 T3 vs fp32 T3: step time and speaker similarity
ve = base.ve
def emb16(w):
    w16 = librosa.resample(np.asarray(w, dtype=np.float32), orig_sr=base.sr, target_sr=16000)
    return ve.embeds_from_wavs([w16], sample_rate=16000).reshape(-1)
def cos(a,b): return float(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b))
ref = emb16(base.conds.gen["prompt_feat"].new_zeros(0).cpu().numpy() if False else np.zeros(1)) if False else None
fp32_wavs, fp32_toks = [], []
for t in TEXTS:
    torch.manual_seed(0); toks = tok_fn(t, base.t3); fp32_toks.append(toks.cpu())
    torch.manual_seed(0); fp32_wavs.append(base.generate(t).squeeze(0).cpu().numpy())
t3_fp32_time = med(lambda: tok_fn(TEXTS[0], base.t3), 3)[0]
base.t3.half()
import dataclasses
for k, v in list(vars(base.conds.t3).items()):
    if torch.is_tensor(v) and v.is_floating_point(): setattr(base.conds.t3, k, v.half())
for m in (base.t3.tfmr, base.t3.speech_head, base.t3.speech_emb): m.half()
t3_fp16_time = med(lambda: tok_fn(TEXTS[0], base.t3), 3)[0]
print(f"RESULT t3-token-time fp32={t3_fp32_time:.3f}s fp16={t3_fp16_time:.3f}s", flush=True)
fp16_wavs = []
for t in TEXTS:
    torch.manual_seed(0)
    w = base.generate(t) if False else None
    toks = tok_fn(t, base.t3)
    toks = toks[toks < 6561].to(base.device)
    sil = torch.tensor([fast_nano.S3GEN_SIL]*3).long().to(base.device)
    wv, _ = base.s3gen.inference(speech_tokens=torch.cat([toks, sil]), ref_dict=base.conds.gen, n_cfm_timesteps=2)
    fp16_wavs.append(wv.squeeze(0).float().cpu().numpy())
for i in range(len(TEXTS)):
    print(f"RESULT text{i} tokens_identical={bool(torch.equal(fp32_toks[i], tok_fn(TEXTS[i], base.t3).cpu()))} "
          f"speaker_sim_fp16_vs_fp32={cos(emb16(fp16_wavs[i]), emb16(fp32_wavs[i])):.4f}", flush=True)
