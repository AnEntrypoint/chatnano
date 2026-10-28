# Chatterbox-Nano local hot-path audit (speed)

Date: 2026-10-23. Scope: `chatterbox/src` (Turbo/Nano class, T3 loop), `nano_fast/`, `ternary/` and `KAGGLE_TERNARY_NOTES.md` (speed only), git history. No source files in `/workspace/chatnano` were edited. The only write is this file. Benchmark scripts and logs live outside the repo in `/tmp/chatnano_audit_bench/`.

Read first: `research_notes/Chatterbox Nano low latency setup/nano_official_setup.md`. Everything there (install pins, no `dtype` knob, no streaming, hard-coded `n_cfm_timesteps=2`, `generate()` signature, no official compile or fp16 statement) is not repeated here. What is new or corrected is marked **[NEW]** or **[CORRECTION]**.

## Summary

1. **[NEW] The fast path recompiles on every new prompt length.** `FastNano.speech_tokens` sets `StaticCache(max_cache_len=L + max_new_tokens)`, so the cache tensor shape changes with the text length `L`. The `reduce-overhead` compiled step then builds a new graph. Measured: a second, different sentence costs 4.4 s versus 1.3 s for a repeat of the first one (same run, same process). Bucketing the cache length to a constant removes this: a new sentence then costs 0.4-0.5 s, with no new graph.
2. **[NEW] The fast path cannot compile in the repo's own venv.** `chatterbox/.venv` (torch 2.6.0+cu126) has no triton, so `torch.compile(mode="reduce-overhead")` fails with `Cannot find a working triton installation`. The README numbers therefore did not come from this venv. They reproduce only in the global env `/data/cache/venvs/ml` (torch 2.14.1+cu130, triton 3.8.0). That env is a different torch version from the one the README states.
3. **[CORRECTION] README end-to-end numbers are not reproducible from the committed script.** `nano_fast/README.md` says "7 interleaved runs (median)". `fast_nano.py`'s `timed()` does one warmup and 3 timed runs, and no committed script interleaves baseline and fast 7 times. The 2.41 s / 1.99 s figures come from code that is not in the repo.
4. **[NEW] Measured stage breakdown** for one 8 s clip (about 197 speech tokens) on the RTX 3060 Laptop: autoregressive loop about 1.06 s, S3Gen vocoder 0.225 s, watermark 0.14 s (CPU), reference embedding 0.10 s only when `audio_prompt_path` is passed. The AR loop is roughly 75% of the request.
5. **[NEW] Per-step host syncs remain** in both the baseline loop and `fast_nano.py`. Each step still evaluates `torch.all(next == stop)` on the host, and the baseline also evaluates `torch.all(logits == -inf)`. Removing them is expected to help, but I did not isolate that gain.
6. Model load (5.4 s) and the reference embedding (0.10 s) are not repeated per request in any repo caller. Keep one process and one `FastNano` instance.

## Hot path, per request

Code references are to `chatterbox/src/chatterbox/tts_turbo.py` and `models/t3/t3.py`.

| Stage | Where | Per request? | Measured (ml env) | Already optimized in `nano_fast`? |
|---|---|---|---|---|
| Model load `from_pretrained(nano=True)` | `tts_turbo.py` L192 | No, once per process | 5.42 s | n/a (the fast path reuses the instance) |
| Reference clip embedding `prepare_conditionals` | `tts_turbo.py` L241-270 | Only if `audio_prompt_path` is passed | 0.102 s median (clip `voices/refs/pd2_rg.wav`) | Not in `fast_nano`. Not needed by repo callers (see below) |
| Text tokenization + `punc_norm` | `tts_turbo.py` `generate` | Yes | negligible (not timed separately) | Same as baseline |
| Prefill (eager, one forward over text + prompt) | `t3.inference_turbo` | Yes | inside AR figure | Eager in both; not optimized |
| AR decode loop | `t3.inference_turbo` L435-460 (baseline), `fast_nano.py` `speech_tokens` (fast) | Yes, per token | baseline 1.062 s for 197 tokens (about 5.4 ms/token) | Partly: static cache plus CUDA graph for the step |
| OOV filter, silence append | `tts_turbo.py` L305-310 | Yes | negligible | Same |
| S3Gen vocoder (`s3gen.inference`, `n_cfm_timesteps=2`) | `tts_turbo.py` L312-316 | Yes | 0.225 s for 8.00 s audio (RTF about 0.028) | Not optimized; same call |
| Watermark `perth.PerthImplicitWatermarker` | `tts_turbo.py` L319 | Yes | 0.138 s on 8 s audio (CPU numpy in / out) | Not optimized |
| `.cpu().numpy()` and `torch.from_numpy` | `tts_turbo.py` L318, L320 | Yes | negligible | Same |

Reference clip use in the repo: `voices/make_voices.py`, `voices/make_variants.py`, `ternary/nano_ternary.py` and `ternary/prepare_tokens.py` call `generate(text)` without `audio_prompt_path`. They use the built-in `models/chatterbox-nano/conds.pt` voice. The `voices/refs/*.wav` files are not passed to `generate` anywhere in the repo, so the reference embedding cost is zero for the current callers. If a caller passes `audio_prompt_path` per request, it costs about 0.10 s per request, and that result can be cached per path (see item B below).

## What `nano_fast/fast_nano.py` does today

- Prefill runs eagerly through `self._step`, which is the raw function. Decode runs through `torch.compile(self._step, mode="reduce-overhead")`.
- `StaticCache` is created per request with `max_cache_len=L + 1000` and `dtype=embeds.dtype` (fp32). The sampler is the same as in `inference_turbo`.
- Per token: `t3.speech_emb(tok)`, the compiled step, `_sample` (logits processors, softmax, `multinomial`) on GPU, and `torch.cat(generated[1:])` for the repetition-penalty history. A `torch.tensor([L+i], device=...)` is built on the host and copied each step.
- EOS check: `if torch.all(tok == stop)` runs on the host each step, so it synchronizes each step.
- EOS trimming: the loop only drops a final stop token. Tokens generated after a stop, if the loop were to go on, would not be removed. This is harmless today because the loop breaks at the first stop check.
- `generate()` is the same vocoder and watermark path as baseline. Output is not byte-identical to baseline because RNG draws happen in a different order. This is already noted in the README.

## Measurements

All numbers are from `/tmp/chatnano_audit_bench/` on the RTX 3060 Laptop GPU (6144 MiB, driver 580.178.04), with the global env `/data/cache/venvs/ml/bin/python3` (torch 2.14.1+cu130, transformers 5.2.0, triton 3.8.0). Scripts were run with `PYTHONDONTWRITEBYTECODE=1 python3 -I`.

Run 1, `stage_bench.py`, command:
```
cd /tmp/chatnano_audit_bench && PYTHONDONTWRITEBYTECODE=1 /data/cache/venvs/ml/bin/python3 -I stage_bench.py > run_ml.log 2>&1
```
Results (`RESULT` lines in `run_ml.log`):
- `load_from_pretrained_s=5.42`
- `prepare_conditionals_from_clip_s median=0.102`
- `baseline_ar_loop_s median=1.062 tokens=197`
- `s3gen_vocoder_s median=0.225 audio_s=8.00`
- `watermark_cpu_s=0.138`
- `fast_generate_call1_s=11.28 graphs+=1` (includes compile)
- `fast_generate_call2_diffText_s=4.40 graphs+=1` (new prompt length, recompile)
- `fast_generate_call3_s=1.31 graphs+=0` (repeat of call 1 text)

Run 2, `variant_bench.py`, command:
```
cd /tmp/chatnano_audit_bench && PYTHONDONTWRITEBYTECODE=1 /data/cache/venvs/ml/bin/python3 -I variant_bench.py > run_variant.log 2>&1
```
`Variant` is `FastNano` with two changes: `max_cache_len` fixed at 1200, and the EOS check every 16 steps. Results:
- `FastNano` text 1: `ar_s=3.347 tokens=162 new_graphs=1`; text 2: `ar_s=4.844 new_graphs=1`; text 3: `ar_s=0.485 new_graphs=0`
- `Variant` text 1: `ar_s=4.085 new_graphs=1`; text 2: `ar_s=0.524 new_graphs=0`; text 3: `ar_s=0.375 new_graphs=0`
- `Variant` repeat pass: `0.725`, `0.518`, `0.378` (no new graphs)

Interpretation:
- Recompile on prompt length is confirmed (`FastNano` text 2 added a graph; `Variant` did not).
- Warm per-token cost in the fast loop is about 4.7 ms (`Variant` 112 tokens in 0.524 s, including prefill). This is close to the README's 5.3 ms, but I did not measure the decode step in isolation, so I cannot confirm the README's per-step figures.
- The `Variant` output token counts differ from `FastNano` only because the RNG is not reseeded per call. I did not check audio equality, and `Variant` as written is not safe to ship (see item A).
- The AR-loop baseline (`inference_turbo`) was not run in `variant_bench.py`. Its 1.062 s figure comes from run 1 on a single text, so the direct comparison is limited to that one text.

Attempted but blocked: running `stage_bench.py` in `chatterbox/.venv` failed at the first compile with `Cannot find a working triton installation`. That is finding 2.

## Recommended changes not yet applied

Ordered by expected value. Expected gains are estimates unless marked measured. None of these change voice identity, since the voice comes only from the conditionals and the S3Gen reference, which are untouched. Neither does any of them change dtype, streaming, or the Nano checkpoint.

**A. Bucket the static-cache length (measured, correctness-sensitive).** Use one fixed `max_cache_len` per process, for example `bucket = max(1200, L + max_new_tokens)` rounded up to a multiple of 256, or a fixed 1200 with a fallback to a second bucket when `L + max_new_tokens > bucket`. This removes the 3-4 s recompile on each new prompt length (measured 4.4 s vs 0.52 s for a new sentence). Requirement: the bucket must be at least `L + max_new_tokens`, or the loop must cap `max_new_tokens` to `bucket - L`. The `Variant` test used a fixed 1200 without a fallback, so it would overflow for long prompts. Trade-off: a larger cache means slightly more memory (negligible at 110M params).

**B. Cache the reference conditionals per clip (cheap, applies only to callers that pass `audio_prompt_path`).** Key `prepare_conditionals` by the file path and mtime, and reuse the result. Measured cost avoided: 0.10 s per request. Zero gain for the current repo callers, which use the built-in `conds.pt`.

**C. Remove the per-step host sync.** Check EOS every N steps (N = 16 in `Variant`), and truncate the output at the first stop token in the block. As written, `Variant` does not truncate, so it would keep tokens after EOS and add garbage audio. Also build the position tensor once on the device (`pos_all[L+i:L+i+1]`, which `Variant` already does) instead of `torch.tensor(..., device=...)` each step. Expected gain: a fraction of the 4.7 ms per token. I did not isolate it, so the gain is not measured.

**D. Keep the baseline loop out of production (no code change needed).** The baseline `inference_turbo` has two host syncs per step (`-inf` check and EOS check) and `tqdm` overhead. Routing production through `FastNano` (after A and C) is the change that matters. Expected gain: the AR share of the 1.06 s baseline, which is most of the 1.4 s total per clip.

**E. Watermark (not applied, unverified).** `PerthImplicitWatermarker.apply_watermark` costs 0.14 s on the 8 s clip, and 0.29 s on random 8 s audio in a second check. That is about 10% of the request. Moving it to GPU or batching it is not something I verified; the Perth source is a neural net, and changing where it runs must keep the watermark output identical. Do not remove it: every output is watermarked, and that is the Resemble contract.

**F. S3Gen vocoder (not applied, unverified).** 0.225 s for 8 s of audio is already RTF 0.028 and `n_cfm_timesteps` is already 2. The remaining options are compiling the estimator or using a CUDA graph for it. Both are unmeasured here, and the compile path fails in the repo venv. Lower priority than A-C.

**Not proposed, by constraint:** fp16 or bf16 (no dtype knob, and no official support statement), streaming (the Nano pipeline has none), and any change to `conds` or the reference clip (changes voice identity).

## Gaps

- No isolated measurement of the decode step, prefill, or the per-step sync cost.
- The README end-to-end numbers (2.41 s, 1.99 s, 7 runs) could not be reproduced with the committed `fast_nano.py`. The committed script runs 3 timed iterations.
- The environment the README used (torch 2.6.0 with triton) is not present in the repo venv. The `ml` env is torch 2.14.1, so its compile behaviour may differ from the README's.
- Audio equality between `FastNano`, `Variant` and baseline was not checked.
- No measurement of GPU memory use.
- The `ternary/` and `KAGGLE_TERNARY_NOTES.md` material is about training and Kaggle queueing. It has no per-token speed data relevant to inference, so it is not used here.
