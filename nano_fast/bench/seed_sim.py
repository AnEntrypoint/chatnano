import sys, numpy as np, torch, librosa
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
from chatterbox.tts_turbo import ChatterboxTurboTTS
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
TEXTS = ["Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. Would you like me to get some prices for you?",
         "This is a considerably longer sentence, written to push the decoder past a few dozen speech tokens, so that the cache bucket and the stop check both get exercised.",
         "Hello there, how are you doing today?"]
def emb16(w):
    w16 = librosa.resample(np.asarray(w, dtype=np.float32), orig_sr=base.sr, target_sr=16000)
    return base.ve.embeds_from_wavs([w16], sample_rate=16000).reshape(-1)
def cos(a,b): return float(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b))
for i, t in enumerate(TEXTS):
    torch.manual_seed(0); a = base.generate(t).squeeze(0).cpu().numpy()
    torch.manual_seed(1); b = base.generate(t).squeeze(0).cpu().numpy()
    print(f"RESULT text{i} fp32_seed0_vs_seed1_sim={cos(emb16(a), emb16(b)):.4f}", flush=True)
