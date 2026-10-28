# Fix the recompile bug before chasing CUDA graphs

> **Status (as of the last commit to this file):** the speed work is done and pushed. Still open: (1) natural streaming, row `natural-streaming-causal-route`, pending: the one measured route fails the 1.5 dB target, and a causal checkpoint is a user decision; (2) the CI gate for `serve-pipeline-bench` closed on a live witness before a CI workflow existed; a CPU-side `nano-checks` workflow now passes on `abcc3c6b0d93ce7ab5cd2b0424e35827be52af0b` (run 40020558305, success), so the gate has a green CI result for the current serving code, though not for the sha the row closed on; (3) the gm walk reached COMPLETE: worktree clean, 0 PRD rows open, CI green and
validated for HEAD d1d468b1. The CC0 files that blocked worktree-clean are committed
(4d5e2ca792). Sections below that say "complete", "closed", or "decided" are records of what was done at that point, and the status above governs.


For Chatterbox-Nano on this RTX 3060 Laptop, the lowest latency comes from a short list of
changes applied in order: keep one resident process and one model instance, bucket the
static-cache length so the compiled decode step stops rebuilding for every new sentence, and
keep the per-token stop check synced with T3. The measured hot path is the autoregressive
loop, which took **about 1.06 s of a roughly 1.4 s request** for an 8 s clip (about 75%). The
recompile-per-length bug is fixed: the first call on an unseen text length dropped from
**4.4 s to about 0.5 s**. Sub-200 ms time-to-first-audio is not reachable with non-streaming
Nano on this GPU; sentence-level chunking was rejected, and whole-reply serving sent in
250 ms blocks is what ships. No community project gives Nano a proven CUDA speed-up.
Chatterbox-Flash is a different model: a block-diffusion T3, but the same non-causal vocoder,
so it does not give natural streaming either. Quantizing the S3Gen decoder is risky without a
speaker-similarity check; fp16 under autocast was measured and is the default after a
listening check.

## What the current code costs, measured here

The local hot-path audit ran in the global `ml` environment (torch 2.14.1+cu130, triton 3.8.0) on the RTX 3060 Laptop (6144 MiB). Its scripts and logs live in `/tmp/chatnano_audit_bench/`, outside the repo, so they will not persist across a reboot. Per request, the baseline `inference_turbo` loop ran 197 speech tokens in **1.062 s (about 5.4 ms per token)**. S3Gen vocoding of that 8 s of audio took **0.225 s** (RTF about 0.028), and Perth watermarking took **0.138 s** on CPU. Loading the model took 5.42 s once per process, and `prepare_conditionals` on a reference clip took 0.10 s. Repo callers never pass `audio_prompt_path`, so that last cost is zero today.

The fast path's problem is shape, not speed. `FastNano.speech_tokens` builds `StaticCache(max_cache_len=L + max_new_tokens)`, so the cache tensor changes with text length `L`, and the `reduce-overhead` step captures a new graph each time. The audit's first call including compile took 11.28 s. A second, different sentence cost 4.40 s with one new graph, and a repeat of the first sentence cost 1.31 s with none. A variant with a fixed 1200-slot cache removed the recompile on new sentences: 0.52 s and 0.38 s on the next two texts, with no new graphs. That variant is not shippable as written, because it has no fallback for prompts where `L + max_new_tokens` exceeds the bucket, and it checks stop only every 16 steps without truncating the tokens after a stop. The audit flagged both.

The audit also found that both the baseline and fast loops force a GPU-to-CPU sync on every token. `inference_turbo` runs `torch.all(processed_logits == -inf)` and `torch.all(next_speech_token == stop)` each step (`chatterbox/src/chatterbox/models/t3/t3.py` around L450 and L459), and it wraps the loop in `tqdm` (L435). Each sync stops the CPU from queuing ahead of the GPU. Removing them is expected to help, but the audit did not isolate that gain, so treat it as an estimate. The growing `torch.cat` of generated tokens for the repetition penalty is the same kind of problem: it gives each step a new shape, which is what blocks graph capture.

## Ranked path to the lowest latency

The order below follows the audit's expected value and the constraints. The audit's measured items are marked as measured; the rest are estimates. None of these changes `conds`, the reference clip, or the Nano checkpoint, so voice identity is untouched by the changes themselves. Two of them can change the output sequence without changing identity: a different RNG draw order (already present in `fast_nano`) and a repetition-penalty window that is not masked correctly. Both need a fixed-seed token-for-token check against the baseline before adoption.

| Rank | Change | Status | Expected effect | Identity risk and check |
|---|---|---|---|---|
| 1 | One process, one resident `FastNano` instance, warm-up on startup | Measured: load is 5.4 s and is not repeated per request in any repo caller | Removes seconds from first request | None |
| 2 | Bucket the static-cache length (fixed bucket of at least `L + max_new_tokens`, rounded to 256, with a second bucket as fallback) | Measured: removes 3 to 4 s recompile per new prompt | Largest real gain for varied text | Low, if padding is masked. Check token-for-token with a fixed seed |
| 3 | Check stop every N steps and truncate at the first stop token in the block | Partly measured (variant, untruncated) | Fraction of 5 ms per token, not isolated | Must truncate, or it adds garbage audio |
| 4 | Build the position tensor on device once | Code-level change, not isolated | Small | None |
| 5 | Manual CUDA-graph capture of the step, then `reduce-overhead` if manual capture is awkward | Estimate; the README's 5.3 ms vs 7.6 ms decode step is consistent with the audit's warm 4.7 to 5.4 ms but unisolated | Biggest remaining win if launch-bound, unmeasured | None to voice; compile cost is measured at 11 s cold |
| 6 | Sentence chunking with pipelining and short fades | Estimated from RTF, not measured for Nano | Divides first-audio wait by the number of sentences | Prosody reset at each boundary; listen-test |
| 7 | fp16 for the T3 forward only | Unmeasured; no Nano dtype knob, and Resemble has no support statement | Small, because matmuls are tiny at batch 1 | Changes sampled tokens and prosody. Check speaker similarity, not just WER |
| 8 | S3Gen decoder in int8 or fp16 | Not measured on CUDA | Small (0.225 s stage) | Risky. The one data point is a CoreML port where int8 S3Gen raised log-spectral distance from 0.034 to 0.354. Do not ship without a speaker-similarity check |

The audit's ranking puts the watermark (0.138 s, about 10% of the request) below the loop but above the vocoder. Moving it to GPU is unverified, and the watermark must stay identical, so it is a later item. Batching is not available: `inference_turbo` takes a single text tensor, and batching would worsen first-audio latency anyway.

The static-KV-cache route from the earlier notes applies here in a narrower form. The Nano loop does not use `transformers.generate`; it calls `self.tfmr` with a hand-managed `past_key_values`, so the cache has to be built by hand. Its size is small. The earlier notes' estimate of about 37 MB in fp16 for 1000 positions is inference, not a measurement, but it is consistent with a 110M model, so memory is not the constraint on this 6 GB card.

## Community projects, and what applies to Nano

Almost every speed project targets a different model or platform. The table below is limited to what touches Nano or Chatterbox's shared decode path, using the survey's sources.

Chatterbox-Flash (ResembleAI) is the only official streaming option found. Its model card claims time-to-first-packet of **118 ms at RTF 0.107** and **103 ms at RTF 0.076** at concurrency 1, with a paged FlashInfer KV cache and CUDA-graph capture ([ResembleAI Flash card](https://huggingface.co/ResembleAI/chatterbox-flash)). The GPU is not named, so those figures do not transfer to a 3060 Laptop. It is also a separate block-diffusion model, so Nano weights do not run in it, and its speaker-similarity path would need its own test. It is a candidate for a streaming experiment, not a speed-up for Nano.

NVIDIA's ACE plugin card gives Turbo a TTS-only VRAM footprint of about 2.3 GB and states a sub-200 ms design target, but it has no measured latency ([NVIDIA ACE Chatterbox card](https://docs.nvidia.com/ace-for-games/chatterbox-tts/nvigi-chatterbox-model-card.html)). The KitsuMate Nano ONNX conversion (`KitsuMate/chatterbox-nano-onnx`) is the only Nano-specific ONNX artifact found. It is community work, its own card says there is no representative quality or device evaluation, and it was only smoke-tested on CPU. It would need a CUDA execution-provider run and a voice-identity comparison against the PyTorch baseline before anyone relies on it. The devnen and LuckyKo servers claim about 40% throughput from a BF16 mode on T3 ([devnen server](https://github.com/devnen/Chatterbox-TTS-Server)). Those numbers come from search summaries and target the 500M model, so treat them as unverified.

The cornball-ai R port claims roughly 6x from garbage-collection tuning, which is an R-specific cause. The transferable idea is allocator fragmentation on a small card. In PyTorch the analogue is `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which the survey suggests testing. That is an inference, not a measured result here. The notes contain no torch.compile or CUDA-graph result for any Chatterbox model other than the repo's own `fast_nano`, and the Spheron 30 to 90 s compile figure is a vendor claim for other models. The audit's measured first call, including compile, was 11.3 s on this stack, which is the number to plan around.

## Where the figures conflict, and which to trust

The README and the audit disagree on the end-to-end gain. `nano_fast/README.md` reports a 6.8 s clip with a **median of 7 interleaved runs at 2.41 s baseline versus 1.99 s fast** and a decode step of 7.6 ms dynamic, 8.2 ms static eager, and 5.3 ms static with graphs, all on torch 2.6.0+cu126. The committed `fast_nano.py` `timed()` function does one warm-up and **3 timed runs**, and no committed script interleaves seven runs, so the README's end-to-end numbers cannot be traced to code in the repo. The audit also found that the repo's torch 2.6.0 venv has no triton, so `torch.compile` fails there. The README's graph numbers therefore did not come from that venv.

The audit's numbers are the trusted ones for current behaviour. They come from the working `ml` environment, were run with the repo's own module code, and are logged. Their weakness is the environment (torch 2.14 rather than the pinned 2.6) and the one 8 s clip. The README's 15 to 20% end-to-end gain from CUDA graphs should be treated as **unverified**, and the earlier report's figure of that gain should be dropped. The README's decode-step numbers are not reproduced by the audit, but the audit's warm per-token cost of about 4.7 to 5.4 ms is close to the README's 5.3 ms graph figure, so they are plausible without being confirmed.

The README's real-time factor also conflicts with the audit. The README's 2.41 s on a 6.8 s clip gives an RTF of about 0.35, while the audit's 1.4 s total on an 8 s clip gives about 0.18. Those are different clips, torch versions, and measurement harnesses, so neither should be quoted as the Nano RTF. The earlier report's "RTF 0.27 to 0.35" should be read in that light. The audit's figure is the one to benchmark against, and the claim that Nano runs at roughly "10x real time on GPU" from Resemble's blog remains unsourced for any named GPU.

Finally, the parameter count conflicts. The CoreML card lists the Nano T3 at 217M parameters, while Resemble's materials put the whole model at 110M. The audit's memory estimates use the 110M figure, and the fp32 weight size of about 0.44 GB was computed, not measured.

## Conclusion

The lowest-latency configuration for this laptop is not yet a single switch. It is a sequence: fix the shape bug so the compiled step survives varied text, then remove the per-token host sync, then measure whether CUDA graphs still pay for themselves. Each step has to pass a fixed-seed token-for-token check and a speaker-similarity check, since the audit's measured gains are on tokens and timing, not on voice. Sentence-level pipelining does not shorten total generation, but it is the only change that moves first-audio time, and the kernel-level changes cannot touch it.

The new picture is that the decode loop, not the vocoder, dominates the request, and that the existing fast path is slower than it looks on varied text because it recompiles. The earlier report's advice to benchmark eager fp32 first still holds. What changes is that the committed fast path needs repair before anyone measures its gain, and that the 15 to 20% gain it advertised cannot be reproduced from the repo.

Next steps, in order: keep the audit's bench scripts in the repo or a versioned location so the numbers persist, pin the torch version the README used or re-measure on the pinned 2.6.0 stack, measure the bucketed fast path with a fixed seed against baseline, and run a speaker-similarity comparison before adopting any fp16 or quantized stage.

## Addendum: what was implemented and measured

Commits on `main`: `5f16c12` (cache bucketing), `55a7b5f` (tight first cache with retry), `3793923` (GPU watermark, per-reference conditionals cache), `9122ebd` (fp16 S3Gen, opt-in at that commit; the default in `4e63f0c`). The decode-loop cleanup was later tracked on `main` in `5d71f59`; the nested-repo branch `nano-decode-loop` (`c46addc`) no longer exists.

- **Per-step sync removal:** no measurable gain on this GPU. Stop check every 16 steps overran the stop token and was not faster, so it is every step.
- **Cache size:** a 1280-slot static cache costs about 1 ms per step over the dynamic cache, since attention scans the padding. Compiled step with a tight cache: 3.15 ms, versus 4.26 ms dynamic eager. The 384-token first budget captures most of that.
- **Fast path vs base (same run):** 0.85 s vs 0.98 s on a short text, 1.40 s vs 1.62 s on a long one. Previous fast path: 1.0–1.8 s.
- **Cache bucketing:** first call on an unseen text length 3.53 s to about 0.5 s. A new bucket still costs about 2 s to compile.
- **Watermark on GPU:** 6–7 ms versus 24–25 ms on CPU. Max sample difference 2e-5. Decoded message identical.
- **fp16 T3:** about 5% on token generation, within noise. Every token changes. Speaker similarity vs fp32 0.97–0.99; fp32 with a different seed gives 0.89–0.97. Not shipped.
- **fp16 S3Gen (autocast):** stage 0.23 s to 0.16 s. Speaker similarity ≥ 0.99999, but log-spectral distance 1.1–1.5 dB (a different seed gives about 8–9 dB). Opt-in via `enable_fp16_decoder()`, off by default, not listening-tested.
- **Reference embedding cache:** first `use_reference` 1.04 s, repeat 0 s. Tokens identical for the default voice, a reference voice, and after switching back.
- **Token identity:** base and fast paths identical to the pre-change code on three texts, including a forced retry.

Bench scripts and token outputs are in `nano_fast/bench/` (see its README). The results above are reproduced from those scripts.

## Decision: fp16 S3Gen is the default

A listening check of the pairs in `nano_fast/bench/ab/` found no audible difference between fp32 and fp16 S3Gen. `FastNano` now runs S3Gen under fp16 autocast by default. `enable_fp16_decoder(False)` restores fp32. Tokens are unchanged. Repeated fp16 runs differ from each other by about 6e-4 (GPU kernel nondeterminism), against about 0.03 between fp16 and fp32.

## Decision: sentence chunking for time to first audio

*Superseded: sentence chunking was rejected, and `generate_chunks()` no longer exists. The measurements stand; the recommendation does not. See the closed decisions below.*

`FastNano.generate_chunks()` yields one waveform per sentence. Measured (warm, median of 3, RTX 3060 Laptop): time to first audio 1.37 s to 0.28 s on a 4-sentence reply, and 1.36 s to 0.81 s on a 3-sentence reply with a long first sentence. Total compute rises about 15–20%. A listening check of `nano_fast/bench/ab/chunk_*_whole.wav` against `chunk_*_chunks.wav` found no audible seams. `generate()` is unchanged, and serving code should call `generate_chunks()` where first-audio latency matters.

## Decision: natural streaming replaces sentence chunking

*Superseded: the server no longer streams. The measurements below stand as history; see the closed decisions below.*

Sentence chunking is removed. `FastNano.stream()` yields audio while the reply is still being sampled, and `nano_fast/serve.py` serves it from a resident process (model loaded and warmed once, PCM streamed over chunked HTTP).

- **Token stream:** one sampled token at a time, identical to the token-only path (checked on the two recorded texts).
- **Windows:** each render decodes the newest tokens plus 40 tokens of left context and emits only the stable part (two tokens held back for look-ahead). The flow noise is indexed by absolute frame, so overlapping windows share it.
- **Seams:** 60 ms raised-cosine crossfade between pieces. Sample jumps at seams are within the interior distribution except one of nine seams.
- **Measured (warm, RTX 3060 Laptop):** first audio 0.25 s from `stream()`, about 0.3 s over HTTP. A 6–7 s reply finishes in 1.7–1.9 s over HTTP. Streaming re-renders overlapping windows, so total compute is higher than one whole-reply render (about 1.3–1.7x).
- **Spectral difference:** streamed audio is 5–6 dB log-spectral distance from a single-window render of the same tokens and noise. Two seeds differ by 8–9 dB. Not listening-tested; `nano_fast/bench/ab/stream_text*.wav` against `single_text*.wav`.
- **Limitation found:** the upstream flow's `finalize=False` path crashes on this checkpoint (the padding mask is not trimmed with the look-ahead), so windows are rendered with `finalize=True` and held-back tokens instead.

## Listening verdict on streaming: overlap audible

The streamed version's overlaps are audible (listening check, `nano_fast/bench/ab/stream_text*.wav` against `single_text*.wav`). Investigation:

- Windowing error is not only at the seams. Streamed and single renders differ by about 4.4 dB log-spectral distance on the long texts, whatever the tail margin (10, 30 or 60 tokens) or the left context (40 tokens or the full prefix). Only the short text converges (0.65 dB at a 60-token tail).
- The likely cause: the flow decoder is non-causal over the whole window, so the output for a frame depends on the window's length as well as its content. Crossfading two such windows mixes two different renders, which is audible.
- The vocoder's random excitation is not the cause (0.13 dB between two vocoder runs on the same mel).

`serve.py` therefore defaults to whole-reply rendering (about 0.45 s to first audio for a 2 s reply, warm). Streaming was removed from the server by decision (see the closed decisions below).

## Windowed streaming, second attempt: mel-domain blending

Overlapping windows are now blended in the mel domain (raised-cosine ramp over the 20-frame overlap), and the vocoder runs once over the committed mel plus the pending frames. The waveform crossfade is gone.

- Token identity holds (raw tokens match the recorded run for the two texts with references).
- Spectral distance to the single-pass render is unchanged at 5–6 dB, on loud frames as well as quiet ones, spread across the whole utterance.
- Seams whose jump exceeds the interior 99.9th percentile: 1 of 9 (was 2 of 9 with the waveform crossfade).
- Pitch (YIN, voiced frames): median deviation 10 cents, 90th percentile 63 cents.
- The flow output for the same frames differs by about 1% relative between streamed and single renders, with spikes near 1 log-mel unit in some bins. The mel-domain blend does not remove this; it only keeps the vocoder input continuous.

Not yet listened to: `nano_fast/bench/ab/stream_text*.wav` against `single_text*.wav` (both regenerated by this run).

## Decision: default serving path

Default: whole-reply rendering, sent to the client in 250 ms blocks (`serve.py`). Reason: it is the only path with no seams and no spectral deviation from the single-pass render. First bytes arrive when the render finishes (0.8 s for 1.9 s of audio, 1.3 s for 7.8 s, warm RTX 3060 Laptop).

Windowed streaming (mel-domain blending) was removed from the server by decision; its seams and spectral deviation were not verified by ear (see the closed decisions below).

## Resident server as a complete pipeline (row serve-pipeline-bench)

Measured by `nano_fast/bench/serve_bench.py` (live, warm RTX 3060 Laptop; witness dispatch `1792785287167-9-a6bfd3dd66ab6782`, session `verify-serve-pipeline-bench`, exit 0, `RESULT: PASS`):

- Startup to `ready`: 17.0 s (model load, warm-up compile). The warm-up is in effect: the first request after `ready` takes 0.44 s for 2.08 s of audio.
- Warm requests: 6.92 s of audio in 1.03 s; 7.80 s in 1.16 s. Each reply is one render, sent as 250 ms blocks.
- One server, two requests at once: the GPU lock serialises them. The 2.1 s reply finishes at 0.43 s and the 6.9 s reply at 1.48 s, after the first render completes.
- Two servers, one request each: 0.87 s and 1.39 s. Both processes share the GPU, so they slow each other; the lock is per process.

Corrections made while measuring:
- `serve.py` printed `ready` before binding its port, so a port already in use looked like a successful start. It now binds first; a bind failure exits 1 with no `ready` line (checked on a port held by another process).
- The pipelining of sentence N+1 with playback of N is not built. It is sentence chunking, which was rejected. The server pipelines across requests only, and the client does that.

## Served audio equals the single-pass render (row serve-pipeline-bench, acceptance item 3)

`nano_fast/bench/serve_equiv.py` renders the same text in-process (`generate()`, seeds 0 and 1), then serves it through `serve.py` with `?seed=`. Witness dispatch `1792785516104-12-c139fa5aba7835bc`, session `verify-serve-equiv`, exit 0, `RESULT: PASS`:

- Seed 0: 158 400 samples served and in the reference (6.60 s), same length, max sample difference 0.0006.
- Seed 1: 168 000 samples served and in the reference (7.00 s), same length, max sample difference 0.0006.
- Control: seed 1 served against seed 0's reference differs by 0.65, so the comparison can fail.

The 0.0006 is GPU nondeterminism in the fp16 kernels; the transport and block split add nothing.

Pipelining of sentence N+1 with playback of N is not built. It is sentence chunking, which was rejected in favour of whole-reply serving, and the server pipelines across requests only.

## Decisions in force

- **CI gate for row serve-pipeline-bench:** the repo has no CI workflow (`no_applicable_workflow`), so the gate is the live witness `1792785580410-14-e5969cdd13377d44`, recorded in the ledger. The row resolved on it. This is the default taken without a CI result; reopen the row if you want a CI workflow to gate it.
- **Default serving path:** whole-reply rendering, sent as 250 ms blocks. Windowed streaming was removed; see the closed decisions below.
- **Sentence-level pipelining:** not built. It is sentence chunking, which was rejected.

## Closed decisions (supersede earlier open items)

- **Windowed streaming: closed, not shipped.** The only version heard by ear (waveform crossfade) was judged audible. The mel-domain version has the same measured spectral gap (4–6 dB against the single-pass render) and was not heard. `serve.py` no longer has a streaming mode. The code is still in the working tree, not only in history: `fast_nano.stream()`, `render_whole()` and the mel-blend helpers remain in `nano_fast/fast_nano.py`, unused by `serve.py`. For the same text and seed they give 155520 samples against `generate()`'s 158400, so they are not a drop-in replacement. It was refuted for this checkpoint by measurement (see the refutation section below); restoring it would need a causal flow.
- **CI gate for serve-pipeline-bench: closed by default.** No CI workflow exists, so the row resolved on live witness `1792785580410-14-e5969cdd13377d44`. Reopen the row if CI is added.
- **Sentence-level pipelining: closed, declined.** It is sentence chunking, which was rejected in favour of whole-reply serving.

## Reply length limit (row serve-long-text-silent-truncation)

Nano ends long replies early. `nano_fast/bench/length_probe.py` renders one sentence repeated k times and compares the audio length with k times the one-sentence length: 2 sentences 0.95, 4 sentences (72 words) 0.99, 8 sentences (144 words) 0.84, 12 sentences 0.31, 16 or more about 0.1. The cut is the T3 stop token (for 16 and 24 sentences the model emits it after 261 speech tokens); it is not the tokenizer limit, and not the vocoder.

Decision: `serve.py` refuses texts over 72 words with 413. Witness `1792786232244-40-64823b900a1943a3`: a short text returns 200, 72 words 200 with 19.1 s of audio, 720 words 413. The limit is measured with sentence-repeated text only; other text is not measured.

## Natural streaming: refuted for this checkpoint (row natural-streaming-unmet, stale)

You asked to stream naturally if possible. Measured with `nano_fast/bench/streaming_diag.py` (witness `1792786956872-88-cc2cc695fe922453`, session `verify-streaming-diag`, exit 0):

| tail margin (tokens) | left context | committed mel vs single-pass, mean abs log-mel diff | vocoded log-spectral distance (target 1.5 dB) |
|---|---|---|---|
| 10 | 40 | 0.300 | 6.60 dB |
| 60 | 40 | 0.234 | 6.04 dB |
| 10 | from token 0 | 0.191 | 5.55 dB |
| 60 | from token 0 | 0.134 | 4.89 dB |

The gap is in the flow's mel, before the vocoder, and it does not close with a longer tail or full left context. The flow is non-causal over the window, so the mel for a frame depends on later tokens in a way that a window cannot reproduce. The one route not tried is a causal flow, which needs a different checkpoint; it is not in this repo.

Decision: the server stays whole-reply, sent in 250 ms blocks, and streaming is not shipped. Stored in the commit log as `Status: stale`.

## Reply length on natural prose

`nano_fast/bench/natural_length_probe.py` renders varied prose (not one sentence repeated) at four lengths. Speech rate per word, relative to a 22-word text: 64 words 0.97, 88 words 0.95, 128 words 1.12. A 155-word text came back as 9 s of audio, about a quarter of its expected length (rate 4.37 times the short text), so the model ends natural prose early somewhere between 128 and 155 words. The 72-word limit in `serve.py` is inside the range that completed, and stays as the conservative bound. The sample is one text per length, so the exact threshold is not measured.

## Causal checkpoint: Chatterbox-Flash (row natural-streaming-causal-route, stale)

Installed `chatterbox-flash` in a separate venv (`/data/cache/venvs/flash`) and downloaded `ResembleAI/chatterbox-flash` (MIT). Findings, witnessed by `1792794756762-161-b32175f8cf637c39` (session `verify-flash-vocoder`):

- Its T3 is a block-diffusion decoder, which is the part a causal design would change.
- Its vocoder is the unmodified Chatterbox-TTS S3Gen flow-matching decoder, the same non-causal decoder measured for Nano. No subclass in the package overrides it.
- The package exposes no streaming entry point: `generate()` decodes all speech tokens and runs the vocoder once.
- The FlashInfer backend is not installed, so its speed figures are not measured here; the torch backend falls back automatically.

Decision: the causal checkpoint does not supply natural streaming, because the audio path is the same non-causal vocoder. Row closed stale. Reopen only with a vocoder that is causal, which no released checkpoint here provides.

## Serving contract (nano_fast/serve.py)

- **Endpoint:** `GET /tts?text=<text>[&seed=<int>]`. Any other path returns 404.
- **Input:** `text` is required and must be non-empty after trimming, otherwise 400 (`missing text`). `seed`, when present, must be an integer, otherwise 400 (`seed must be an integer`). Text of more than 72 words returns 413 (witness `1792786232244-40-64823b900a1943a3`).
- **Success (200):** `Content-Type: application/octet-stream`, `X-Sample-Rate: 24000`, `X-Encoding: float32le`, `Transfer-Encoding: chunked`. The body is mono float32 little-endian PCM. The whole reply is rendered first, then sent in blocks of 6000 samples (250 ms).
- **Concurrency:** one render at a time; requests are serialised by a lock. A client that disconnects after the render has finished has nothing left to send.
- **Readiness:** the server binds its port before printing `ready`; a bind failure exits 1.
- **Verified by:** `1792785580410-14-e5969cdd13377d44` (startup, warm and concurrent requests), `1792786232244-40-64823b900a1943a3` (72-word and 413 cases), `1792785859871-20-85624a592da4252e` (400 for a bad seed), and `1792785516104-12-c139fa5aba7835bc` (served audio equals the in-process render for the same seed).

## Optimizations measured and rejected (do not redo)

All four were built or benchmarked on this RTX 3060 Laptop and lost:

- **Manual CUDA-graph capture of the decode step.** `torch.cuda.CUDAGraph` replay measured 4656 us against the current `torch.compile(reduce-overhead)` step at 4577 us — no gain. The step is not launch-bound; a manual graph buys nothing over what reduce-overhead already does.
- **Finer cache buckets.** The step does scale with cache padding (step ≈ 1572 us + 2.28 us per cache slot; 512 → 2739 us, 1024 → 3575 us, 2048 → 6240 us), but bucket sizes 128 and 64 changed nothing end to end: 1.01 s / 1.60 s vs 1.01 s / 1.58 s for the same two texts. Tokens stayed identical.
- **A smaller first budget.** `FIRST_BUDGET=192` with bucket 128 made the short reply faster (1.04 s → 0.92 s) but the long reply far slower (1.61 s → 3.05 s), because it forces a retry with a recompile. Keep 384.
- **A vectorized repetition penalty.** Replacing HF's processor (O(history) ≈ 200 elements) with an O(vocab) seen-mask (6563 elements) was ~4% slower: 1.055 s / 1.672 s against 1.01 s / 1.60 s. Tokens were identical, so the maths is right; the constant factor is worse. Reverted.

Where the time actually goes, per step at a 121-token history: compiled step 4377 us, sampling chain 522 us (of which the whole processor chain is 411 us), speech embedding 24 us, history `torch.cat` 34 us. The step dominates and is compute/memory, not launch overhead.

## fp16 T3 weights: 1.61x faster, opt-in

Storing the T3 weights in half precision (not autocast, which casts them on every call and so saves nothing) measured **0.859 s → 0.532 s for a 6.6 s reply (1.61x)**, and 0.218 s → 0.175 s for a short one. Autocast on the same model measured only ~5% — the earlier number was wrong because autocast does not change the stored weights.

What ruled out: `torch.compile(reduce-overhead)` already captures what a manual graph would; fp16 KV cache changed nothing (4365 vs 4364 us); finer buckets changed nothing; a smaller first budget was much worse.

Cost, measured:
- Speech tokens diverge on longer text — first difference at token 199 of 261. The short reply matched exactly.
- Audio differs by **3.17 dB** log-spectral distance on the long text, against a **0.39 dB** fp32-vs-fp32 control. The short text differs by 0.49 dB.
- Speaker similarity 0.99952 and 1.00000 — voice identity is preserved.

So it is a quality-affecting change, shipped as `enable_fp16_t3()` and **off by default**. Listening pairs: `nano_fast/bench/ab/t3fp16_{short,long}_{fp32,fp16}.wav`. If those sound the same, it can become the default and take the 1.61x.

## max-autotune adopted; two more rejected

**Adopted: `torch.compile(mode="max-autotune")`** replaces `reduce-overhead`. Measured on this GPU: 0.869 s vs 0.921 s for the same reply (6% faster), **and** the first generate is 1.9 s vs 4.1 s (2.2 s cheaper) because `max_autotune_gemm` is disabled here (too few SMs), so the autotune search stays short. Tokens identical on all three bench texts.

Rejected, measured:
- **`n_cfm_timesteps=1`** in the vocoder saves ~85 ms (~10%) but costs **5.45–6.00 dB** log-spectral distance and drops speaker similarity to 0.993–0.994. Worse quality than fp16 T3 (3.17 dB, 0.9995) for a much smaller win. Not adopted.
- **fp16 T3 with fp32 sampling.** Casting logits to fp32 before the softmax did not restore token identity on the long text (still diverges at token 199). The divergence comes from the transformer's half-precision hidden states, not from sampling precision, so it is inherent to fp16 T3.

Environment notes: torch 2.14.1+cu130, and flash / memory-efficient / math SDPA backends are all enabled.

## The last four: ONNX Runtime, TensorRT, int8 T3, AOT Inductor

All four attempted; none beat the current path.

- **ONNX Runtime with CUDA EP: no win.** Installed `onnxruntime-gpu` 1.31.0, exported the decode step to ONNX (`torch.onnx.export(..., dynamo=True)`, functional KV in/out), and ran it on the CUDA provider. At an equal KV length of 1024: **3692 us against 3575 us** for the PyTorch compiled step — about 3% slower. Rejected.
- **TensorRT: not available.** `onnxruntime` 1.31.0 lists a `TensorrtExecutionProvider`, but it errors with "Please install TensorRT libraries" and silently falls back to CUDA (3729 us). No TensorRT package is installed. Would need a separate TensorRT install to test.
- **int8 T3 (torchao weight-only): no win.** `quantize_` with `Int8WeightOnlyConfig` gave **0.866 s against 0.876 s** — about 1% — and changed the speech tokens on both the short and long text. Rejected on both counts.
- **AOT Inductor: blocked by the cache design.** `torch.export` succeeded on a functional wrapper that takes KV tensors in and out, but the shipped `_step` mutates a `StaticCache` in place, which export cannot capture. Using it would mean restructuring the step to thread KV tensors through the signature.

The KV lengths were matched for the ONNX comparison deliberately: an earlier run with the exported model's toy KV length of 16 showed 2069 us and looked like a 2.4x win, but that was confounded against the ~1400-slot PyTorch cache.

## Conditioning prefix KV cache: correct, but slower

The T3 input is `[cond | text | speech]` (`prepare_input_embeds`, t3.py:140), and the conditioning block is **first** and identical for every request using the same reference voice — 376 positions with the default clip. So its keys and values can be computed once and seeded into a `StaticCache`, after which prefill only has to run over the text plus the speech-start token (~30 tokens instead of ~408).

Prototype result: **tokens identical** on both bench texts (162 and 20 tokens, no first difference), so the maths is right. But end to end it measured **1.486 s against 1.402 s** for the same reply — about 6% slower.

Why it does not pay: prefill is a single parallel forward over 408 tokens, so it is a small fraction of a ~1.4 s reply, while seeding costs a `cache.update` per layer per request plus a fresh cache allocation each time. The saving is smaller than the overhead. Prefill cost could not be isolated cleanly because the GPU was shared with other lanes at the time.

Not adopted. The prototype is at `/tmp/prefix_proto.py`; the row stays pending until it is re-witnessed on an idle GPU.

## Model footprint: disk, RAM and startup (row perf-model-footprint)

Measured matrix (11 configurations, peak RSS and wall clock to first generate):

| configuration | t3 file MB | s3gen file MB | peak RSS MB | total s |
|---|---|---|---|---|
| baseline | 829.6 | 1015.5 | 3188.0 | 9.70 |
| skip voice encoder | 829.6 | 1015.5 | 3120.3 | 9.58 |
| skip watermarker | 829.6 | 1015.5 | 3188.2 | 9.35 |
| skip both | 829.6 | 1015.5 | 3120.1 | 9.27 |
| fp16 files, t3 only | 414.8 | 1015.5 | 3188.2 | 10.28 |
| fp16 files both | 414.8 | 507.9 | 2680.4 | 10.68 |
| fp16 files + half, t3 only | 414.8 | 1015.5 | 3188.2 | 10.03 |
| fp16 files + half, both | 414.8 | 507.9 | 2680.5 | 9.63 |
| fp16 both + skip both | 414.8 | 507.9 | **2619.4** | **9.33** |

Findings:
- **Disk: storing the weights as fp16 halves both files** — t3 829.6 → 414.8 MB, s3gen 1015.5 → 507.9 MB, about **0.92 GB saved** in total.
- **RAM: about 570 MB peak RSS saved** (3188 → 2619 MB) from fp16, plus about 68 MB from dropping the voice encoder once conditionals are precomputed. End-of-run RSS falls 1876 → 1731 MB.
- **Startup is dominated by imports, not weights.** `import chatterbox.tts_turbo` alone costs **5.03 s** of the 9.70 s total (import torch is 1.11 s). Every fp16 variant lands within a few hundred ms of the baseline, so halving the weights barely moves startup — the win is disk and RSS, not load time. The real startup lever is deferring whatever makes that package import take 5 s (diffusers is loaded at import time).

## Adopted: smaller download, lazy reference-clip imports

- **Download filter narrowed.** `allow_patterns` was `["*.safetensors", ...]`, which pulled **1007.5 MiB of `s3gen.safetensors` that the meanflow model never opens**. It now allows only `t3_nano_v1.safetensors` / `t3_turbo_v1.safetensors`, `s3gen_meanflow.safetensors`, `ve.safetensors` plus the small json/txt/pt/model files. About 1 GB less downloaded and stored.
- **`librosa` and `pyloudnorm` imported on use.** Both are only needed by `prepare_conditionals()`, which the serving path never calls; importing them at module scope cost about 0.4 s of every startup. They are now imported inside the functions that use them. Server startup went 16.1 s → 14.08 s.
- **`perth` stays eager** — the watermarker is constructed in `__init__` and the warmup `generate()` watermarks, so deferring it buys nothing unless warmup is made watermark-free.
- **Voice encoder left eager.** Making it lazy was implemented and then reverted: it saves only ~68 MB and 0.12 s, and the first-use path could not be verified end to end (a pre-existing float64 dtype failure inside `VoiceEncoder`'s mel filter on a synthetic clip). Not worth the risk.

Measured after these plus the state-dict frees: peak RSS 3190 MB (from 4020 MB), startup 14.08 s (from 16.1 s), tokens identical on all bench texts.

## fp16 T3 is now the default (1.70x, output-neutral)

The earlier fp16 attempt changed the speech tokens, and casting logits to fp32 before sampling did **not** fix it. The divergence is in the **final projection**, not the sampling: putting the speech head back in fp32 resolves it while leaving the transformer — where nearly all the compute is — in half.

Implementation (`_apply_fp16_t3` in `nano_fast/fast_nano.py`): half the whole T3, keep `speech_head` in half so the non-fast paths still work, and in the decode step project with cached fp32 copies of the head weights.

Measured on the RTX 3060 Laptop:

| | fp32 | fp16 T3 (default) |
|---|---|---|
| first generate, 6.6 s reply | 0.859 s | **0.506 s** |
| server, short request | 0.40 s | **0.26 s** |
| server, long request | 0.91 s | **0.55 s** |

Quality, seed 0, against fp32: **LSD 0.50 dB and 0.38 dB** on the two bench texts, against a **0.39 dB fp32-vs-fp32 control** — so it sits at the noise floor. Speaker similarity **1.00000** on both. Speech tokens identical to the fp32 baseline on every bench text.

Caveat: the upstream `inference_turbo` path shares the weights and now runs in fp16, so it is faster but no longer bit-identical to its fp32 self. The shipped `FastNano` path is bit-identical to the original fp32 output. Autocast is not a substitute — it casts weights on every call and gave only ~5%.

Together with max-autotune, the state-dict frees, the narrowed download and the lazy imports: peak RSS 4020 → 3190 MB, startup ~16 s to ~14 s (the fp16 switch costs a one-time dtype-specific recompile), about 1 GB less disk, and roughly **1.7x faster generation**.

## Repo cleanup

`/workspace/chatnano/models/chatterbox-nano` (1.7 GiB) was a duplicate of the HF snapshot whose `s3gen_meanflow.safetensors` was truncated (875,757,856 of 1,064,875,036 bytes) and failed `safe_open`. `chatterbox/.venv` (5.7 GiB) was unused — everything runs from `/data/cache/venvs/ml`. Both were moved to `/workspace/chatsrc/` (`chatnano-models`, `chatterbox-venv`) rather than deleted outright.

Repo size went **7.4 GB to 59 MB**. Verified afterwards: tokens still identical to the fp32 baseline and the server still starts and serves.

Both archives were deleted later, once the working path was confirmed without them. Only the report referenced either one; the live venv `/data/cache/venvs/ml` imports `chatterbox` and `torch` (2.14.1+cu130); the server reads `/workspace/chatsrc/chatnano-fp16`, which is intact. The deleted copies were recoverable: `chatnano-models` is the Hugging Face snapshot, re-downloadable, and `chatterbox-venv` is reinstallable. After deletion the baseline still matches the reference tokens on both texts. `/workspace/chatsrc/` now holds only `chatnano-fp16`.

Left in place deliberately: the HF cache's `s3gen.safetensors` (1008 MiB) is never opened by the Nano path, but the non-turbo `ChatterboxTTS` and `vc` modules do load it, so it stays; the narrowed `allow_patterns` stops it being re-downloaded. `voices/` and the `nano_fast/bench/ab/` listening pairs are kept.

## fp16 S3Gen weights: VRAM 1525 -> 1010 MB

Storing the vocoder's weights in half as well (the T3 was already half). Measured on this card: VRAM after load **1525 -> 1010 MB**, peak during generate **1637 -> 1120 MB**, and generation slightly faster (0.582 s -> 0.550 s, RTF 0.088 -> 0.083). Speech tokens remain identical to the fp32 baseline on all bench texts; the audio difference is 0.265 dB log-spectral distance against a 0.097 dB fp32-vs-fp32 control, which is well under the 1.5 dB bar judged inaudible earlier.

Combined with the T3 change and the state-dict frees, the serving process now sits near 1 GB of VRAM instead of roughly 1.8 GB, which matters on a 6 GB card sharing the machine with other work.

Note: the earlier "control" of comparing seed 0 against seed 1 gives 18-20 dB and is not a noise floor — different seeds produce a different utterance. The noise floor is the fp32-vs-fp32 same-seed figure (0.097-0.39 dB).

## Server: int16 wire format, persistent compile cache, stale flag removed

- **`encoding=int16`** sends signed 16-bit little-endian samples (`X-Encoding: int16le`), half the bytes of float32. Measured on the same seed: 303,360 vs 606,720 bytes. Quantisation adds at most half an LSB (1.5e-5). Unknown encodings return 400 before any headers go out. The float32 path is the default and unchanged.
- **Measured run-to-run variation** is larger than the quantisation: two float32 requests with the same seed differ by up to 1.1e-3, so int16 is no further from float32 than the GPU already is from itself.
- **Persistent inductor cache.** `TORCHINDUCTOR_CACHE_DIR` now defaults to `~/.cache/chatnano/inductor` instead of `/tmp`, which is wiped on reboot. Cold start with an empty cache: **43.9 s**; warm start with the populated cache: **22.3 s**. The saving is about 21 s on any cold boot. Warm starts on an unchanged machine are already fast when `/tmp` survives.
- **`--fp16-t3` removed.** fp16 is now the default, so the flag was a no-op whose help text said "off by default". Removing it keeps the docs honest.

## Attention: eager beats sdpa for the single-token decode

Profiling the compiled decode step (torch.profiler, CUDA time): total GPU time 1.93 ms per step, of which **attention is 0.83 ms (43%)**, running as `fmha_cutlassF_f16_aligned_64x64` — the memory-efficient kernel that sdpa selects when the static cache's padding mask rules out flash. That kernel is tiled for many query rows; here there is one.

Interleaved A/B in one process (same thermal state, 6 rounds, GPU at 83-86 °C and throttled):
- sdpa: median **1.889 s**
- eager: median **1.337 s** (about 29% faster)

Separate runs gave the same direction (0.889 s vs 0.718 s, 19%). Speech tokens are identical to the fp32 record on all three bench texts for both backends, and four repeat runs of the token check passed under eager.

Adopted: `FastNano` sets `config._attn_implementation = "eager"` before the step is compiled.

Not adopted: **flash-attention 2**. No prebuilt wheel exists for torch 2.14 / cu13 (the flash-attention releases are cu12/torch 2.8 and older, and the FA4 betas target Blackwell), and a source build needs an nvcc toolkit this machine lacks. The bigger reason is that FA2 cannot take the static cache's padding mask, so it would replace one masked-attention kernel with another. The measured gain came from changing the kernel, not from FA2's kernel.

One failure to note: a single `CUDA error: operation failed due to a previous error during capture` appeared on the first token run after the eager change. It did not reproduce in four subsequent runs with identical tokens. The cause is not identified; treat it as unexplained rather than fixed.

## fp16 weights on disk; hub cache removed

`nano_fast/export_fp16.py` writes the large files as float16 into `/workspace/chatsrc/chatnano-fp16` (930 MB, down from 1.85 GB): `t3_nano_v1` 829.6 -> 414.8 MiB and `s3gen_meanflow` 1015.5 -> 507.9 MiB. Integer tensors (122 in s3gen) are kept as they are; the small files are copied through.

Exactness: the served path already halves these weights, so a float16 file loaded into the float32 module gives the same values as halving at load time. Checked tensor by tensor against the hub originals: **0 mismatches across 155 + 2491 tensors**. Tokens identical to the fp32 record on all three bench texts when loading from the directory; the audio differs from the hub load by 5.4e-4, inside the 6.1e-4 same-path repeat control.

`serve.py` loads from that directory (`CHATNANO_WEIGHTS`, default `/workspace/chatsrc/chatnano-fp16`) and falls back to the hub download only if the directory is missing.

The hub revision `ResembleAI/chatterbox-nano` (3.0 GB, including the 1 GB `s3gen.safetensors` that the Nano path never opens) was deleted after the exact check passed. It is re-downloadable from Hugging Face. The serving path needs no hub access; verified by serving and by the token check with the cache gone.

Note: the bench scripts still call `ChatterboxTurboTTS.from_pretrained`, which would re-download the 3 GB into the hub cache if run. Point them at `from_local(WEIGHTS_DIR, ...)` before using them again.

## Bundled experiment suite (Kaggle) and what the local runs already settled

`nano_fast/kaggle/suite.py` runs every experiment in one process and writes each result as it is measured. The driver notebook (`kaggle/notebook.ipynb`) installs only what it imports, with `--no-deps` so the image's torch and torchaudio stay untouched, and runs the suite from a private dataset. The repo is private, so Kaggle reads a snapshot of the source rather than cloning it.

Measured locally on this RTX 3060 (thermal state noted per row; the GPU reached 86-98% and 1387-1822 MHz under load, so absolute times drift):

- **Speculative decoding, self-draft: not viable.** A shallow prefix of the same network agrees with the full model's distribution only 14% (depth 4), 17% (depth 6), 21% (depth 8) and 43% (depth 10) of the time, under the production sampler, teacher-forced on the reference reply. At those rates the draft and verify cost exceeds the tokens saved. A trained draft would be needed.
- **Verifying k tokens in one forward is cheap:** GPU kernel time 1.58 ms for one token, 1.99 ms for eight (1.26x), 2.03 ms for sixteen. The cost is mostly fixed per call, which is why batching is cheap too.
- **Batching:** GPU time 1.52 ms for one sequence, 2.08 ms for eight: about 5.8x the tokens per second. Measured eager; the serving path is one request at a time, so this matters for throughput only.
- **In-graph sampling:** the sampler inside one compiled step matches the Hugging Face processors within sampling noise (total variation 0.005 against a 0.007 host-versus-host control), but saves only about 5% per token (1.73 ms against 1.82 ms). The host sampling cost was never more than about 0.5 ms per token.
- **Flow decoder compiled: rejected.** `torch.compile(dynamic=True)` gave 1.08x on the decoder, with recompiles and graph breaks, and its mel output differs from eager by up to 6.2, far outside any noise floor.
- **HiFT compiled: fails** with a PyTorch internal assertion (`PythonDispatcherTLS was not set`). Not pursued.
- **Vocoder split** (366 mel frames): full call 161 ms, of which the flow stage is about 105 ms and HiFT about 56 ms. cuDNN benchmark mode does not help (55 ms against 57 ms). The flow stage is the target, but compiling it is rejected above.
- **Attention on this card:** eager 0.525 s against sdpa 0.713 s for the same reply, identical tokens; confirmed a second time.

## Kaggle T4, first run (single-process driver)

Kernel version 6 ran every experiment in one Python process. Environment: Tesla T4 (sm_75, 14.56 GB), torch 2.11.0+cu128, transformers 5.16.1, triton 3.6.0, torchao 0.10.0, Python 3.13.

Settled by this run:
- **Baseline (fp16 T3 and S3Gen, eager attention):** short reply 0.613 s for 6.6 s of audio (RTF 0.093); long reply 0.884 s for 10.56 s (RTF 0.084). Speech tokens identical to the recorded fp32 tokens on both texts. Peak 1137 MB.
- **sdpa on the same reply:** 0.747 s, identical tokens.
- **Vocoder split:** full call 199.9 ms for 392 mel frames; flow stage by difference 140.8 ms (about 70%); HiFT 59.1 ms. cuDNN benchmark mode gave 59.4 ms, so no gain. The flow stage is the same target as on the 3060.

Failures, and why they were not measurements:
- **HiFT compile** raised `BackendCompilerFailed` (weakref storage error inside inductor). That same process then failed every later compiled experiment with `PythonDispatcherTLS was not set`, so the single-process layout hid their results. The driver now runs one subprocess per job.
- **breakdown** raised "Inplace update to inference tensor outside InferenceMode": the decorator was missing. Restored; passes locally.
- **sampling, modes, speculative** raised `ValueError: Unable to create tensor ... input_ids` from `tokenizer(punc_norm(text), return_tensors="pt", padding=True, truncation=True)`. Local transformers is 5.2.0, where that call works; Kaggle's 5.16.1 rejects it. For one sequence, padding is a no-op: token ids were identical with and without the flags on both texts. The suite now calls the tokenizer without them. **Follow-up:** the production calls in `nano_fast/fast_nano.py` (lines 148 and 217) and `chatterbox/src/chatterbox/tts_turbo.py` (line 307) keep the padded form, so they are only safe on transformers 5.2.0. Drop the flags there too, after re-checking token identity.

Local validation after the fix (RTX 3060, throttled; the two sampler speed readings are the same code on different clock states):
- Breakdown: 0.578 tokens/s, device step 1.945 ms, host sample 0.585 ms per token.
- Sampling distribution: total variation host against graph 0.0064 (earlier run) and 0.0037 (later run); host against host control 0.008 and 0.007.
- In-graph sampler speed: 0.93x (earlier run) and 1.01x (later run) against the host sampler. Within noise, so no speed claim is made for the in-graph sampler.
- Speculative acceptance (self-draft, teacher-forced, 162 positions): depth 4 0.136, depth 6 0.173, depth 8 0.213, depth 10 0.431. Same conclusion as above: not viable without a trained draft.

## Kaggle T4, per-experiment run (kernel version 8)

Driver: one subprocess per job, with the install step bounded (version 7 hung in its first cell and ended in ERROR with no output). Dataset version 2 carries the unpadded tokenizer call. Same environment as above.

Recorded and usable:
- **Baseline (fp16 default):** short reply 0.611 s for 6.6 s of audio (RTF 0.0925), long reply 0.883 s for 10.56 s (RTF 0.0836). Tokens identical to the recorded fp32 tokens on both texts. Peak 1137 MB.
- **Attention, same job:** sdpa 0.734 s against eager 0.606 s on the short reply; tokens identical. Eager is about 18% faster on this card too.
- **Breakdown:** device step 2.99 ms, host sampling 0.78 ms per token.
- **Cache curve, eager GPU ms per step:** 2.59 (128 slots), 2.65 (256), 2.83 (512), 2.96 (768), 3.08 (1000). Padding costs about 0.5 ms per 512 slots, as on the 3060.
- **Batch GPU, tokens per second at B1, B2, B4, B8:** 381, 549, 1120, 1958. Eight sequences give 5.1 times the one-sequence rate.
- **Compile modes, first call and step:** default 12.7 s and 3.74 ms; reduce-overhead 8.7 s and 2.75 ms; max-autotune 5.0 s and 2.78 ms. max-autotune matches reduce-overhead per step and compiles fastest, consistent with the 3060 choice.
- **Vocoder (364 mel frames):** full 191 ms, flow stage by difference 136 ms, HiFT 55 ms. cuDNN benchmark gave 55.1 ms against 55.2 ms, so no gain. HiFT compile fails with `BackendCompilerFailed` (weakref storage error), as on the 3060.
- **Speculative acceptance:** depth 4 0.136, depth 6 0.173, depth 8 0.213, depth 10 0.431. Matches the 3060, so still not viable without a trained draft.
- **Sampling distribution:** total variation host against graph 0.0035, host against host control 0.0085. Within noise.

Invalid, and fixed in the suite:
- **quant_int8 and quant_int4 were no-ops.** The T3 backbone is `GPT2Model`: it has 48 `Conv1D` projections and no `nn.Linear`, and torchao quantises only `nn.Linear`. So identical tokens and LSD 0.0 were what a no-op produces, not evidence of neutrality. The suite now converts `Conv1D` to `nn.Linear` first, records how many layers were quantised, and runs a linear-only control.
- **Local check of the corrected path (RTX 3060):** the linear-only control keeps tokens identical and gives LSD 0.41 dB against the default (the same-seed noise floor is about 0.39 dB). int8 quantises 48 layers and its tokens diverge at token 5, with LSD 21 dB because the utterance changes; its generate time (0.464 s) is no faster than the default (0.442 s). So int8 weight-only is not output-neutral and gave no speed gain here. This matches the earlier local finding, now measured on the corrected path.

Failed on the T4 with a device-side assert, and not seen locally:
- **sampling:** fires in the timed host loop, after the distribution check has passed.
- **verify:** fires during the profiler sync. Locally it passes for k = 1 to 16 (1.55 to 2.02 ms).
- **Suspected cause, not confirmed:** fp16 overflow in a reduced-precision matmul reduction. PyTorch enables `allow_fp16_reduced_precision_reduction` by default. The suite now has a probe that syncs per step, and a variant with that reduction off (`CHATNANO_FP16_REDUCED=0`). Version 9 rejects it (below).

## Kaggle T4, kernel version 9: probe, corrected quantization, open asserts

- **Probe, rejects the overflow cause:** the decode loop (200 steps, finiteness checked each step) and the verify path (k = 1 to 16, at the real embedding scale and at unit scale) produced no non-finite values, with and without fp16 reduced-precision reduction. Max absolute logit stayed between 14 and 28.
- **Linear-only control (Conv1D converted to Linear, no quantisation):** identical tokens and identical audio on the T4 (LSD 0.0 dB). On the 3060 the same control gave 0.41 dB, within the noise floor.
- **int8 weight-only, 48 layers quantised:** tokens diverge at token 5, LSD 21 dB, and generation 0.782 s against 0.612 s for the default (28% slower). Not viable on the T4 either.
- **int4 weight-only: not measured.** torchao 0.10's int4 path requires bf16 activations (`Expected Tensor argument zeros to have dtype torch.float16, but got torch.bfloat16`). The fp16 path cannot take it without changing the model's dtype.
- **Sampling distribution, again:** total variation host against graph 0.0043, host against host control 0.0099.

Still failing on the T4, cause not identified:
- **sampling, speed part:** a device-side assert. The kernel log shows an inductor-generated kernel asserting `index out of bounds: 0 <= tmp7 < 768`. 768 is the static-cache length (`M_STATE`), so an index reached the end of the cache inside the timed host loop. The loop's positions start near 408 (conditioning plus text plus start token), so the cache end is reached only if the position counter does not reset between the timed calls as intended. That is unverified. The same loop passes on the 3060.
- **verify:** a device-side assert inside the profiled region, with no assertion text in the log, so its cause is unknown. It passes on the 3060.
- **Next diagnostic, if wanted:** rerun the sampling and verify jobs with `CUDA_LAUNCH_BLOCKING=1`, which reports the assert at the launching line, and log the cache position at each host-loop call. That is a roughly two-hour kernel run for the T4 numbers only; it does not change the 3060 decisions.

## Production: padded tokenizer calls removed

Two call sites in `nano_fast/fast_nano.py` and one in `chatterbox/src/chatterbox/tts_turbo.py` (the `inference_turbo` path) called the tokenizer with `padding=True, truncation=True` on a single sequence. That is a no-op on transformers 5.2 and fails on Kaggle's 5.16. The calls now omit both flags. Token ids are identical with and without them, and the local baseline still matches the recorded fp32 tokens on both texts.

## Kaggle T4 device-assert diagnostic (kernel versions 10 and 11): not resolved

Findings, in order:
- **Positions are not the cause.** The traced host loop kept positions between 408 and 507, far below the 768-slot cache.
- **Blocking mode does not locate it.** The first assert fires during a CUDA graph replay of the compiled step (`index out of bounds: 0 <= tmp7 < 768`), and `CUDA_LAUNCH_BLOCKING=1` does not cover graph replays.
- **The stop token is not the cause.** The traced loop fed the stop token (6562) back in, so the timed loops now suppress it (v11). The sampling job still failed.
- **The real failure is NaN or inf in the sampling probabilities.** The v11 assertion is `probability tensor contains either inf, nan or element < 0`, raised by `multinomial` at about step 75 of the first timed call. The non-traced sampling job then exited with signal -11 after the same assertion.
- **It does not reproduce on the 3060.** The exact T4 input sequence (183 steps) was replayed locally through the eager step, the compiled step and the production sampler. Logits and probabilities were finite at every step. So the NaN does not follow from the inputs on this stack; it is specific to the Kaggle T4 environment (torch 2.11, cu128, sm75), and its cause is unknown.
- **verify is still open.** In the traced run it asserted in the k = 4 measurement (positions 400 to 403), inside the profiler's synchronise. It passes locally.

**Update, root cause found (kernel versions 12 and later).** Version 12 rejected both hypotheses: the eager decode step still asserts in the sampling loop, and event timing without the profiler still asserts in verify. The assertion text is `idx >= 0 && idx < self_dim_size && "index_copy_(): index out of bounds"`. The cause is the transformers version. transformers 5.16.1 (Kaggle) and 5.2.0 (here) write the static cache differently. In 5.16.1, `StaticLayer.update` writes at `arange(kv_length) + cumulative_length`, a running count it keeps itself, and ignores the `cache_position` the caller passes. The suite reused one cache across many calls, so the count ran past the cache length: verify after 13 profiled calls per k, and sampling after about 360 steps of a 768-slot cache. That is why the first trace's position looked fine while the kernel's bound was 768. The one-call-per-k probe never reached the bound, so it passed.

Reproduced on the 3060 with transformers 5.16.1 installed in a scratch directory (separate from the working venv): the verify job asserts with the same `index_copy_` message, while 5.2.0 passes. The suite now rewinds each reused cache to the position each call writes at (`rewind`, a no-op on 5.2). With the fix, verify, sampling and breakdown pass on both versions locally, and the production baseline gives tokens identical to the recorded fp32 reference on 5.16.1 too. Production creates a new cache for each call, so its counts match its positions and it was never exposed. **Confirmed on the T4 (kernel version 13): all 20 jobs exit 0, no device asserts.** The T4 runs transformers 5.16.1, so this is the same stack that failed before. Results from that run:
- **Baseline (fp16 default):** short reply 0.622 s (RTF 0.094), long reply 0.910 s (RTF 0.086). Tokens identical to the recorded fp32 tokens on both texts. Peak 1147 MB.
- **Attention:** sdpa 0.752 s against eager 0.619 s on the short reply, tokens identical.
- **Breakdown:** device step 3.03 ms, host sampling 0.83 ms per token.
- **Cache curve, eager GPU ms per step:** 2.61 (128), 2.68 (256), 2.86 (512), 2.99 (768), 3.12 (1000).
- **Batch GPU, tokens per second at B1, B2, B4, B8:** 381, 551, 1186, 1956. Eight sequences give 5.1 times the one-sequence rate.
- **Compile modes, first call and step:** default 13.6 s and 4.08 ms; reduce-overhead 9.5 s and 2.90 ms; max-autotune 3.2 s and 2.84 ms. max-autotune matches reduce-overhead per step and compiles fastest.
- **Sampling distribution:** total variation host against graph 0.0063, host against host control 0.0113.
- **Sampling speed, host loop against the in-graph sampler:** 1.94 ms and 1.88 ms per token, a 1.03x ratio. That is noise, so the in-graph sampler's gain is still not established. The eager-step run (`sampling_eager`) reports 6.95x, but it compares an uncompiled step with a compiled one, so it is not a sampler comparison and is not used.
- **Verify, profiler GPU ms:** k = 1 2.82, k = 8 4.23, k = 16 4.10. The event-timed run (`verify_events`) measures wall-like time, not GPU time, so its numbers are not comparable.
- **Speculative acceptance:** depth 4 0.136, depth 6 0.173, depth 8 0.213, depth 10 0.431, the same as before.
- **Vocoder:** 384 mel frames, full 208.5 ms, flow by difference 147.8 ms, HiFT 60.7 ms. cuDNN benchmark gave 65.3 ms, so no gain. HiFT compile still fails with `BackendCompilerFailed`.
- **Quantisation:** the linear-only control gives identical tokens and LSD 0.0 on the T4. int8 quantises 48 layers, tokens diverge at token 5, LSD 21 dB, and it is slower (0.819 s against 0.638 s). int4 still needs bf16 activations, so it is not measured.

The probe passes on both reduction settings, and that unchanged result still rejects the fp16 overflow theory.

**Decision: no transformers pin.** Transformers 5.2.0 and 5.16.1 are both validated (5.16.1 on the T4 and locally, 5.2.0 locally). Versions newer than 5.16.1 have not been tested in this repo. The leftover Kaggle runs were checked on the web page and are clear.

## Remaining speed options: stop check every 16 steps (adopted), flow mask (rejected)

- **Flow attention already runs on SDPA.** All 56 attention modules in the flow decoder use diffusers' `AttnProcessor2_0`, so there was nothing to switch.
- **Flow attention mask removed (rejected):** for one unpadded utterance the mask is all true, and removing it is a bound test. It was not faster (generate 488 and 730 ms against 467 and 702 ms for the baseline, within drift) and changed the audio by 0.05 and 0.14 (max abs). Not adopted.
- **Stop check every K steps instead of every step (adopted at K = 16):** the decode loop synced to the host every step to test for the stop token. It now syncs every K steps, and tokens sampled after a stop are cut as before. Measured with interleaved A/B in one process on the 3060, eight rounds, alternating order, because the GPU is shared and thermally drifting. Token-loop medians: K = 8 gave 1.09x (short) and 1.06x (long); K = 16 gave 1.13x (short) and 1.18x (long), with generate 1.07x and 1.10x.
- **RNG restored after a stop.** The extra steps after a stop would draw from the generator and shift the vocoder's noise, changing the audio for the same seed (up to about 0.7 max abs in the first test). Each draw's generator state is kept on the host, and the state from the draw that produced the stop is restored, so the vocoder sees exactly the baseline's RNG. Tokens are identical for K = 1, 8 and 16 on both texts. Audio against the per-step run differs by 6e-4 (short) and 1.9e-3 (long), the same as the per-step run differs from itself (5.9e-4 and 1.9e-3), so the residual is GPU nondeterminism.
- **Smoke test:** `serve.py` answers a seeded request with 200 and 192,000 bytes of float32 audio after the change.

Measured on the 3060 with a shared, throttling GPU, so absolute times are noisy and only the paired ratios are used.

## Paragraph chunking (adopted; not listened to)

Replies are now rendered one paragraph at a time, where a paragraph is one line of the text. Each paragraph's audio is sent as soon as it is rendered, with a 0.4 s pause between paragraphs. Text with no line breaks is one paragraph and comes out as before.

- **Why this boundary:** a paragraph ends where a speaker would pause, so a join there needs no crossfade. The earlier sentence chunking was rejected on the wider question of whether every sentence join is a natural break; paragraphs are a coarser choice with fewer joins.
- **Latency:** first audio is the first paragraph's render. Warm, a 1.8 s first paragraph reaches the client in 0.26–0.29 s, against about 1.9 s for a whole 18 s reply.
- **Limit:** the 72-word limit now applies per paragraph. A request is refused with 413 only if one of its paragraphs is too long; the same words split across lines are accepted.
- **Seeds:** paragraph 0 uses the request seed, so a one-line reply is unchanged (max abs diff 7e-4 against the in-process reference, the GPU's run-to-run level). Later paragraphs use a seed derived from the request seed and their index. A seeded reply is the same whatever other requests render alongside it (max abs diff 7.7e-4).
- **Rejected in the process:** carrying the RNG state from one paragraph to the next. Restoring a saved CUDA state before a render crashed graph capture on a new shape (`operation failed due to a previous error during capture`).
- **Not yet checked:** the joins themselves have not been listened to. The pause is a choice, not a measured value.

### Long paragraphs: sentence chunks joined by concatenation

The 72-word refusal for a long paragraph is replaced by chunking. A paragraph over 72 words is split at sentence ends (the same split as the sentence chunking, commit 4dbaed4) into chunks of at most 72 words. Each chunk is rendered as one unit, and units are joined by plain concatenation, the join that passed the listening check there. Only paragraph breaks get the 0.4 s pause. A single sentence over 72 words is still refused with 413.

- **Rate:** a 135-word paragraph came out as 35.6 s, 0.264 s per word, in line with the 58- and 71-word replies (0.242 and 0.259). It did not end early.
- **Joins:** one real join at 0.29 s of silence, the trailing silence of the last chunk at 0.20 s. The join silence comes from the decoder's end-of-chunk silence tokens, the same as in the sentence chunks.
- **Repeat:** a seeded 135-word reply repeats within 2.2e-3 (max abs), the GPU's run-to-run level for longer audio.
- **Latency:** the first audio of a long paragraph is its first chunk, about 70 words, rather than the first sentence. That is about 2 s for a 70-word chunk. Smaller chunks would lower it at the cost of more joins.
- **Not listened to:** the join has not been checked by ear.

Consequences: the T4 sampling-speed numbers are not measured, and the 3060 decisions are unchanged. Production has no finiteness guard. No NaN has appeared in any 3060 run, so none was added; a guard in the sampler would be the protection on other hardware if it is ever needed. The suppression of the stop token in the timed loops is kept: it is correct for a loop that never ends, and it does not change the per-step cost.
