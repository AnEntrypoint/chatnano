# Natural streaming: token identity, time to first audio, and windowing error versus one-window render.
import sys, json, time, numpy as np, soundfile as sf, torch
sys.path.insert(0, "/workspace/chatnano/chatterbox/src"); sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS
import fast_nano
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
fast = fast_nano.FastNano(base)
TEXTS = [
    "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. Would you like me to get some prices for you?",
    "This is a considerably longer sentence, written to push the decoder past a few dozen speech tokens, so that the cache bucket and the stop check both get exercised across several boundaries in one run.",
    "Hello there, how are you doing today?",
]
# new_tokens.json lists texts in tokens_bench order: [0]=hilarious, [1]=hello, [2]=long
recorded_by_text = json.load(open("/workspace/chatnano/nano_fast/bench/new_tokens.json"))["fast"]
recorded = [recorded_by_text[0], recorded_by_text[2], recorded_by_text[1]]
def lsd(a, b):
    import librosa as lb
    A = np.abs(lb.stft(a.astype(np.float32), n_fft=1024, hop_length=256)) + 1e-7
    B = np.abs(lb.stft(b.astype(np.float32), n_fft=1024, hop_length=256)) + 1e-7
    n = min(A.shape[1], B.shape[1])
    return float(np.mean(np.sqrt(np.mean((10*np.log10(A[:, :n]**2) - 10*np.log10(B[:, :n]**2))**2, axis=0))))
out = "/workspace/chatnano/nano_fast/bench/ab"
for i, text in enumerate(TEXTS):
    for _ in fast.stream(text): pass  # warm the graphs for this text's shapes
    torch.manual_seed(0)
    raw = list(fast._token_stream(text, 0.8, 1000, 0.95, 1.2, 1000))
    toks = [t for t in raw if t < 6561]
    print(f"RESULT text{i} raw_identical_to_recorded={raw == recorded[i]} n={len(toks)}", flush=True)
    torch.manual_seed(0); torch.cuda.synchronize(); t0 = time.perf_counter()
    pieces, first = [], None
    for p in fast.stream(text):
        torch.cuda.synchronize()
        if first is None: first = time.perf_counter() - t0
        pieces.append(p)
    torch.cuda.synchronize(); total = time.perf_counter() - t0
    streamed = np.concatenate(pieces)
    mel1 = base.conds.gen["prompt_feat"].shape[1]
    noise_ref = torch.randn(1, 80, mel1 + 2 * 1000 + 16, device=base.device, dtype=torch.float32,
                            generator=torch.Generator(device=base.device).manual_seed(0))
    single = fast.render_whole(toks, noise_ref, mel1)
    single = np.asarray(single)
    print(f"RESULT text{i} TTFA={first:.2f}s total={total:.2f}s pieces={len(pieces)} streamed_audio={len(streamed)/base.sr:.2f}s "
          f"single_audio={len(single)/base.sr:.2f}s", flush=True)
    print(f"RESULT text{i} LSD_stream_vs_single_render={lsd(streamed, single):.3f}dB", flush=True)
    sf.write(f"{out}/stream_text{i}.wav", streamed, base.sr)
    sf.write(f"{out}/single_text{i}.wav", single, base.sr)
