# Chatterbox-Nano fast path

`fast_nano.py` runs Nano's autoregressive loop with a static KV cache and a compiled
decode step (`torch.compile(mode="max-autotune")`, eager attention). Speech tokens match
`ChatterboxTurboTTS.generate()` for a fixed seed. Three stages are fp16 by default: T3 (with the
speech head kept in fp32), S3Gen under fp16 autocast with fp32 weights (`enable_fp16_decoder(False)`
restores fp32), and the watermark runs on the GPU (same decoded message, sample difference about 2e-5).
The flow's CFM decoder replays as CUDA graphs, one per 64-frame length bucket, captured at server startup
(`enable_decoder_graphs(False)` runs it eagerly; the mel matches eager in 12 of 12 flow comparisons and the
generator advances identically). See `bench/README.md` for the checks.

`serve.py` is the resident server: it loads and warms the model once. Each line of the text is a
paragraph. A paragraph over 72 words is split at sentence ends into chunks of at most 72 words,
rendered one at a time and joined by concatenation (Nano ends replies over about 72 words early,
`bench/length_probe.py`). Audio is sent as float32 PCM in 250 ms blocks as each render completes,
with a 0.4 s pause after each paragraph. A single sentence over 72 words gets 413. Measurements and
decisions: `reports/Chatterbox Nano maximum speed RTX 3060.md`.

Run from the chatterbox venv or the global env (`python3 -I nano_fast/fast_nano.py both`).

## Measured at the first fast-path commit (RTX 3060 Laptop 6 GB, torch 2.6.0+cu126, 6.8 s clip)

These numbers predate the cache bucketing, the tight first cache and the fp16 decoder. The
current figures are in the report.

| Measurement | Result |
|---|---|
| Decode step, greedy, dynamic cache (eager) | 7.6 ms |
| Decode step, greedy, static cache (eager) | 8.2 ms (tokens match dynamic) |
| Decode step, greedy, static + CUDA graphs | 5.3 ms (tokens match dynamic) |
| End to end, 7 interleaved runs (median) | baseline 2.41 s, fast 1.99 s |
| End to end, best of 7 | baseline 1.82 s, fast 1.84 s |

The laptop GPU is noisy (medians drift between runs by up to 0.5 s), so the
end-to-end gain is about 15-20% on median and not clearly measurable on the best
run. Prefill runs eagerly, because its shape differs from the decode step.

Known limits: the compiled step warms up on the first call (several seconds of
compile). A new text length can still cost about 2 s to compile when it crosses a
cache bucket. The stop check runs every 16 steps (one host sync per 16 tokens). Tokens
sampled after the stop are cut, so a reply ends at the same token as `generate()`, and
the RNG is restored to the state a per-step check would leave, so the audio is the same.
