import hashlib
import json
import math
import statistics
import sys
import time

import soundfile as sf
import torch

out_path = sys.argv[1]
WAV_DIR = sys.argv[2]
WAV_LABELS = ("short", "paragraph")
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
from fast_nano import FastNano  # noqa: E402

TEXTS = {
    "hello": "Hello there.",
    "short": ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
              "Would you like me to get some prices for you?"),
    "long": ("This is a considerably longer sentence, written to push the decoder past a few dozen speech "
             "tokens, so that the cache bucket and the stop check both get exercised across several "
             "boundaries in one run."),
    "paragraph": ("The new model arrived last week, and the team has been testing it against the old one for "
                  "three days. Most of the difference is in the first two seconds of each reply."),
}
ROUNDS = 7
SEED = 0
HARMONICS = 9
SAMPLE_RATE = 24000
VOICED_THRESHOLD = 10
SINE_AMP = 0.1
NOISE_STD = 0.003


def digest(t):
    return hashlib.md5(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:10]


def lsd_db(a, b):
    window = torch.hann_window(1024, device=a.device)
    spec = [torch.stft(x.flatten().float(), 1024, 256, window=window, return_complex=True).abs().pow(2)
            for x in (a, b)]
    da, db = (10 * torch.log10(s + 1e-10) for s in spec)
    return float(((da - db) ** 2).mean(dim=0).sqrt().mean())


def time_once(fn):
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000


base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
fast = FastNano(base)
fast.enable_decoder_graphs(False)
hift = base.s3gen.mel2wav
hift.stft_window = hift.stft_window.to("cuda")
hift_eager = hift.inference
captured = {}


def record(speech_feat, cache_source=None):
    captured["mel"] = speech_feat.detach().clone()
    return hift_eager(speech_feat=speech_feat, cache_source=cache_source)


def autocast():
    return torch.autocast("cuda", dtype=torch.float16, enabled=fast.fp16_decoder, cache_enabled=False)


def run_hift_eager(mel):
    with autocast():
        return hift_eager(speech_feat=mel, cache_source=torch.zeros(1, 1, 0, device="cuda"))[0]


def draw_noise(samples):
    phase = torch.distributions.Uniform(-math.pi, math.pi).sample((1, HARMONICS, 1)).to("cuda")
    sine_noise = torch.randn(1, HARMONICS, samples, device="cuda")
    uv_noise = torch.randn(1, 1, samples, device="cuda")
    return phase, sine_noise, uv_noise


def graph_body(mel, phase, sine_noise, uv_noise):
    f0 = hift.f0_predictor(mel)
    f0_up = hift.f0_upsamp(f0[:, None])
    harmonic = torch.arange(1, HARMONICS + 1, device=mel.device, dtype=f0_up.dtype).view(1, HARMONICS, 1)
    theta = 2 * math.pi * (torch.cumsum((f0_up * harmonic / SAMPLE_RATE).float(), dim=-1) % 1)
    phase = phase.clone()
    phase[:, 0, :] = 0
    sine = SINE_AMP * torch.sin(theta + phase)
    uv = (f0_up > VOICED_THRESHOLD).float()
    noise_amp = uv * NOISE_STD + (1 - uv) * SINE_AMP / 3
    sine = sine * uv + noise_amp * sine_noise
    merged = torch.tanh(hift.m_source.l_linear(sine.transpose(1, 2)))
    return hift.decode(x=mel, s=merged.transpose(1, 2))


def istft_capturable(magnitude, phase):
    n_fft, hop = hift.istft_params["n_fft"], hift.istft_params["hop_len"]
    window = hift.stft_window
    magnitude = torch.clip(magnitude, max=1e2)
    spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    frames = torch.fft.irfft(spec.transpose(1, 2), n=n_fft, dim=-1) * window
    frame_count = frames.shape[1]
    total = (frame_count - 1) * hop + n_fft
    overlap_add = torch.nn.functional.fold
    y = overlap_add(frames.transpose(1, 2), (1, total), (1, n_fft), stride=(1, hop)).reshape(magnitude.shape[0], total)
    envelope = overlap_add((window ** 2).view(1, n_fft, 1).expand(1, n_fft, frame_count), (1, total), (1, n_fft),
                           stride=(1, hop)).reshape(total)
    return (y / envelope)[:, n_fft // 2: total - n_fft // 2]


pool = torch.cuda.graph_pool_handle()


def capture(mel):
    samples = mel.shape[-1] * 480
    static_mel = mel.clone()
    static = draw_noise(samples)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), autocast():
        for _ in range(2):
            graph_body(static_mel, *static)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    eager_istft = hift._istft
    hift._istft = istft_capturable
    try:
        with torch.cuda.graph(graph), autocast():
            out = graph_body(static_mel, *static)
    finally:
        hift._istft = eager_istft
    return static_mel, static, graph, out


def run_hift_graph(static_mel, static, graph, out, mel, seed):
    torch.manual_seed(seed)
    for dst, src in zip(static, draw_noise(mel.shape[-1] * 480)):
        dst.copy_(src)
    static_mel.copy_(mel)
    graph.replay()
    return out


def run_hift_reimpl_eager(mel, static):
    eager_istft = hift._istft
    hift._istft = istft_capturable
    try:
        with autocast():
            return graph_body(mel, *static)
    finally:
        hift._istft = eager_istft


def row_for(mel, label):
    cap_t = time.perf_counter()
    static_mel, static, graph, out = capture(mel)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - cap_t) * 1000

    eager_ms, reimpl_ms, graph_ms, replay_ms = [], [], [], []
    for _ in range(ROUNDS):
        eager_ms.append(time_once(lambda: run_hift_eager(mel)))
        reimpl_ms.append(time_once(lambda: run_hift_reimpl_eager(mel, draw_noise(mel.shape[-1] * 480))))
        graph_ms.append(time_once(lambda: run_hift_graph(static_mel, static, graph, out, mel, SEED)))
        replay_ms.append(time_once(graph.replay))

    torch.manual_seed(SEED)
    e_wav = run_hift_eager(mel).float().flatten()
    e_state = digest(torch.cuda.get_rng_state())
    torch.manual_seed(SEED)
    e2_wav = run_hift_eager(mel).float().flatten()
    g_wav = run_hift_graph(static_mel, static, graph, out, mel, SEED).float().flatten().clone()
    g_state = digest(torch.cuda.get_rng_state())

    if label in WAV_LABELS:
        sf.write(f"{WAV_DIR}/{label}_eager.wav", e_wav.cpu().numpy(), SAMPLE_RATE, subtype="PCM_16")
        sf.write(f"{WAV_DIR}/{label}_graph.wav", g_wav.cpu().numpy(), SAMPLE_RATE, subtype="PCM_16")

    return {
        "mel_frames": int(mel.shape[-1]),
        "samples": int(e_wav.numel()),
        "capture_ms": round(capture_ms, 1),
        "hift_eager_ms": round(statistics.median(eager_ms), 2),
        "hift_reimpl_eager_ms": round(statistics.median(reimpl_ms), 2),
        "hift_graph_total_ms": round(statistics.median(graph_ms), 2),
        "hift_graph_replay_ms": round(statistics.median(replay_ms), 2),
        "saving_ms": round(statistics.median(eager_ms) - statistics.median(graph_ms), 2),
        "graph_only_saving_ms": round(statistics.median(reimpl_ms) - statistics.median(replay_ms), 2),
        "wav_max_abs_graph_vs_eager": round(float((e_wav - g_wav).abs().max()), 6),
        "wav_lsd_db_graph_vs_eager": round(lsd_db(e_wav, g_wav), 4),
        "wav_max_abs_eager_vs_eager": round(float((e_wav - e2_wav).abs().max()), 6),
        "wav_lsd_db_eager_vs_eager": round(lsd_db(e_wav, e2_wav), 4),
        "rng_state_eager": e_state,
        "rng_state_graph": g_state,
    }


result = {"rows": {}, "buckets": {}}
with torch.inference_mode():
    mels = {}
    for name, text in TEXTS.items():
        hift.inference = record
        torch.manual_seed(SEED)
        fast.generate(text)
        hift.inference = hift_eager
        mels[name] = captured["mel"]

    for name, mel in mels.items():
        result["rows"][name] = row_for(mel, name)
        print(name, json.dumps(result["rows"][name]), flush=True)

    long_mel = mels["long"]
    for frames in (64, 128, 192, 256, 320, 384, 448, 512):
        result["buckets"][frames] = row_for(long_mel[..., :frames].contiguous(), str(frames))
        print("bucket", frames, json.dumps(result["buckets"][frames]), flush=True)

with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print("RESULT", json.dumps(result), flush=True)
