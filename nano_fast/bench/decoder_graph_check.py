"""Flow decoder graphs against the eager decoder, in one process: same texts, same seeds, same weights.

For each text and seed the S3Gen flow runs with the graphs off and on, from the same generator state. Compared: the
mel (max and mean abs difference) and the wav (max abs difference and LSD, 1024/256 STFT). Timings are medians of
the flow alone and of generate(), interleaved. Usage: python3 -I decoder_graph_check.py OUT.json
"""
import json
import statistics
import sys
import time

import torch

out_path = sys.argv[1]
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS, S3GEN_SIL  # noqa: E402
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
SEEDS = (0, 1, 2)


def lsd_db(a, b):
    window = torch.hann_window(1024)
    spec = [torch.stft(x.flatten().float(), 1024, 256, window=window, return_complex=True).abs().pow(2)
            for x in (a, b)]
    da, db = (10 * torch.log10(s + 1e-10) for s in spec)
    return float(((da - db) ** 2).mean(dim=0).sqrt().mean())


def median_ms(fn, n):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return round(statistics.median(times) * 1000, 1)


base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
fast = FastNano(base)
warm_t = time.perf_counter()
warmed = fast.warm_decoder_graphs()
warm_s = round(time.perf_counter() - warm_t, 2)
s3 = base.s3gen
result = {"warm": {"graphs": warmed, "seconds": warm_s}, "texts": {}, "graphs_captured": None}
with torch.inference_mode():
    for name, text in TEXTS.items():
        torch.manual_seed(0)
        toks = fast.speech_tokens(text).flatten()
        toks = toks[toks < 6561].to("cuda")
        toks = torch.cat([toks, torch.tensor([S3GEN_SIL] * 3, device="cuda").long()])

        def flow():
            with torch.autocast("cuda", dtype=torch.float16, enabled=fast.fp16_decoder):
                return s3.flow_inference(speech_tokens=toks, ref_dict=base.conds.gen, n_cfm_timesteps=2,
                                         finalize=True)

        row = {"speech_tokens": int(toks.numel()), "seeds": {}}
        for seed in SEEDS:
            fast.enable_decoder_graphs(False)
            torch.manual_seed(seed)
            mel_e = flow().clone()
            fast.enable_decoder_graphs(True)
            torch.manual_seed(seed)
            mel_g = flow().clone()
            fast.enable_decoder_graphs(False)
            torch.manual_seed(seed)
            wav_e = fast.generate(text).clone()
            fast.enable_decoder_graphs(True)
            torch.manual_seed(seed)
            wav_g = fast.generate(text).clone()
            d = (mel_g - mel_e).abs()
            row["seeds"][seed] = {
                "mel_frames": int(mel_e.shape[-1]),
                "mel_max_abs": round(float(d.max()), 5),
                "mel_mean_abs": round(float(d.mean()), 6),
                "wav_samples": [int(wav_e.shape[-1]), int(wav_g.shape[-1])],
                "wav_max_abs": round(float((wav_g - wav_e).abs().max()), 5),
                "wav_lsd_db": round(lsd_db(wav_e, wav_g), 4),
            }
        fast.enable_decoder_graphs(False)
        row["flow_eager_ms"] = median_ms(flow, 7)
        fast.enable_decoder_graphs(True)
        flow()
        row["flow_graph_ms"] = median_ms(flow, 7)
        gen_e, gen_g = [], []
        for _ in range(3):
            fast.enable_decoder_graphs(False)
            gen_e.append(median_ms(lambda: fast.generate(text), 1))
            fast.enable_decoder_graphs(True)
            gen_g.append(median_ms(lambda: fast.generate(text), 1))
        row["generate_eager_ms"] = statistics.median(gen_e)
        row["generate_graph_ms"] = statistics.median(gen_g)
        result["texts"][name] = row
        result["graphs_captured"] = len(fast.decoder_graphs.graphs)
fast.enable_decoder_graphs(True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print("RESULT", json.dumps(result), flush=True)
