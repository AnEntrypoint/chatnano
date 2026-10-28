# Where does windowed streaming diverge from the single-pass render? Replays the stream's render points
# with fast._advance, compares the committed mel with the single-pass mel on the same tokens, then vocodes
# both. Configurations come from argv: tail margin (tokens) and left-context (tokens, 100000 = from token 0).
import sys, numpy as np, torch, librosa
sys.path.insert(0, "/workspace/chatnano/chatterbox/src"); sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS
import fast_nano

tail = int(sys.argv[1]) if len(sys.argv) > 1 else 10
context = int(sys.argv[2]) if len(sys.argv) > 2 else fast_nano.STREAM_CONTEXT
fast_nano.TAIL = tail
fast_nano.STREAM_MARGIN = fast_nano.LOOKAHEAD + tail
fast_nano.STREAM_CONTEXT = context

base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
fast = fast_nano.FastNano(base)
TEXT = " ".join(["This sentence is part of a long reply that the server should either speak in full or refuse."] * 4)
mel1 = base.conds.gen["prompt_feat"].shape[1]
noise = torch.randn(1, 80, mel1 + 2 * 1000 + 16, device=base.device, dtype=torch.float32,
                    generator=torch.Generator(device=base.device).manual_seed(0))
torch.manual_seed(0)
toks = [t for t in fast._token_stream(TEXT, 0.8, 1000, 0.95, 1.2, 1000) if t < 6561]

st = fast._new_state(); tokens = []; nxt = fast_nano.STREAM_FIRST
for t in toks:
    tokens.append(t)
    if len(tokens) >= nxt:
        fast._advance(st, tokens, False, noise, mel1); nxt = len(tokens) + fast_nano.STREAM_STEP
fast._advance(st, tokens, True, noise, mel1)
streamed = st["mel"]
with torch.inference_mode(), torch.random.fork_rng(devices=[base.device]):
    _, single = fast._render_mel(toks, 0, True, noise, mel1)
single = single[:, :streamed.shape[1]]
d = (streamed - single).abs()
print(f"RESULT tail={tail} context={context} frames={streamed.shape[1]} mel_mean_abs_diff={float(d.mean()):.3f}", flush=True)
with torch.inference_mode():
    w_s = fast._vocode(streamed).cpu().numpy(); w_1 = fast._vocode(single).cpu().numpy()
n = min(len(w_s), len(w_1))
S = np.abs(librosa.stft(w_s[:n], n_fft=1024, hop_length=256)) + 1e-7
R = np.abs(librosa.stft(w_1[:n], n_fft=1024, hop_length=256)) + 1e-7
dd = np.sqrt(np.mean((20*np.log10(S) - 20*np.log10(R))**2, axis=0))
print(f"RESULT tail={tail} context={context} vocoded_lsd_db={float(dd.mean()):.2f} target_db=1.5", flush=True)
