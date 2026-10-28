# Chatterbox-Nano fast path

`fast_nano.py` runs Nano's autoregressive loop with a static KV cache and a
CUDA-graph decode step (`torch.compile(mode="reduce-overhead")`). Speech tokens match
`ChatterboxTurboTTS.generate()` for a fixed seed. Two stages differ by default: S3Gen runs
under fp16 autocast (`enable_fp16_decoder(False)` restores fp32), and the watermark runs on
the GPU (same decoded message, sample difference about 2e-5).

`serve.py` is the resident server: it loads and warms the model once, renders each reply whole,
and sends it in 250 ms blocks. Requests longer than 72 words get 413, because Nano ends longer
replies early (`bench/length_probe.py`). Measurements and decisions: `reports/Chatterbox Nano
maximum speed RTX 3060.md`.

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
cache bucket. The stop check runs every step, so a reply ends at the same token as
`generate()`.
