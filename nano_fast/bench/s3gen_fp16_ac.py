import sys, time, statistics, numpy as np, torch, librosa, copy
sys.path.insert(0, "/workspace/chatnano/chatterbox/src"); sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm, S3GEN_SIL
import fast_nano
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
TEXTS = ["Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. Would you like me to get some prices for you?",
         "This is a considerably longer sentence, written to push the decoder past a few dozen speech tokens, so that the cache bucket and the stop check both get exercised.",
         "Hello there, how are you doing today?"]
def toks_of(t):
    torch.manual_seed(0)
    ids = base.tokenizer(punc_norm(t), return_tensors="pt", padding=True, truncation=True).input_ids.to(base.device)
    x = base.t3.inference_turbo(t3_cond=base.conds.t3, text_tokens=ids)
    x = x[x < 6561].to(base.device)
    return torch.cat([x, torch.tensor([S3GEN_SIL]*3, device=base.device).long()])
def s3(toks, dtype):
    torch.manual_seed(0)
    return base.s3gen.inference(speech_tokens=toks, ref_dict=base.conds.gen, n_cfm_timesteps=2)[0].squeeze(0).float().cpu().numpy()
def emb16(w):
    return base.ve.embeds_from_wavs([librosa.resample(w.astype(np.float32), orig_sr=base.sr, target_sr=16000)], sample_rate=16000).reshape(-1)
def cos(a,b): return float(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b))
TOK = [toks_of(t) for t in TEXTS]
def med(fn):
    fn(); ts=[]
    for _ in range(3):
        torch.cuda.synchronize(); s=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-s)
    return statistics.median(ts)
with torch.inference_mode():
    fp32 = [s3(t, None) for t in TOK]
    t32 = med(lambda: s3(TOK[0], None))
    try:
        ac = lambda t: torch.autocast('cuda', dtype=torch.float16)
        def s3ac(toks):
            with torch.autocast('cuda', dtype=torch.float16):
                return s3(toks, None)
        fp16 = [s3ac(t) for t in TOK]
        t16 = med(lambda: s3ac(TOK[0]))
        for i in range(3):
            print(f"RESULT text{i} speaker_sim={cos(emb16(fp16[i]), emb16(fp32[i])):.6f} maxabs={np.max(np.abs(fp16[i]-fp32[i])):.2e} rms_ref={np.sqrt(np.mean(fp32[i]**2)):.3f} len={len(fp16[i])}", flush=True)
        print(f"RESULT s3gen time fp32={t32:.3f}s fp16={t16:.3f}s", flush=True)
    except Exception as e:
        print("RESULT s3gen fp16 failed:", type(e).__name__, str(e)[:200], flush=True)
import librosa as lb
def lsd(a, b):
    A = np.abs(lb.stft(a.astype(np.float32), n_fft=1024, hop_length=256)) + 1e-7
    B = np.abs(lb.stft(b.astype(np.float32), n_fft=1024, hop_length=256)) + 1e-7
    return float(np.mean(np.sqrt(np.mean((10*np.log10(A**2) - 10*np.log10(B**2))**2, axis=0))))
with torch.inference_mode():
    for i in range(3):
        torch.manual_seed(1); alt = base.s3gen.inference(speech_tokens=TOK[i], ref_dict=base.conds.gen, n_cfm_timesteps=2)[0].squeeze(0).float().cpu().numpy()
        print(f"RESULT text{i} LSD_fp16_vs_fp32={lsd(fp16[i], fp32[i]):.3f} LSD_fp32_seed1_vs_seed0={lsd(alt, fp32[i]):.3f}", flush=True)
