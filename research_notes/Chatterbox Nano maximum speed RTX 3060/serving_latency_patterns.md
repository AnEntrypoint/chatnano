# Serving latency patterns for a non-streaming Chatterbox-Nano on RTX 3060 Laptop (6 GB)

Date of research: 2026-10-23. Scope: what happens around the model (chunking, warm-up, residency, conditioning cache, batching, audio output, process model, measurement). Model internals are out of scope.

Sources read first: `/workspace/chatnano/research_notes/Chatterbox Nano low latency setup/nano_official_setup.md` (not repeated here except where corrected), `/workspace/chatnano/nano_fast/README.md` and `fast_nano.py`, `/workspace/chatnano/chatterbox/src/chatterbox/tts_turbo.py`.

Caveat on sources: web results were used from search snippets only (no WebFetch). Most TTS-latency sources are vendor blogs or issue trackers. Numbers attributed to them below are as reported in snippets and are not independently verified. Items marked "inference" are my reasoning from the local code and measurements, not from a source.

---

## Local facts that constrain the design (from the repo, verified in code)

- `generate(text)` is one blocking call that returns one full tensor (`tts_turbo.py` ~L272-320). Output is `[1, N]` at `S3GEN_SR = 24000` Hz (`models/s3gen/const.py` L1).
- Text path: `punc_norm` (L36-70) capitalises the first letter, collapses whitespace, replaces `…`, `:`, dashes and smart quotes, and **appends a full stop if the text does not end in `. ! ? - ,`**. So every call, including each chunk, gets a sentence-final cadence. Chunks must therefore be cut only at real sentence boundaries, or a fake full stop lands mid-sentence.
- Tokenizer call uses `truncation=True` (L295) with no explicit max length in the call. Text beyond the tokenizer's max length is **silently dropped**. Max length value not checked. Chunking also protects against this.
- Conditioning: `prepare_conditionals` (L241-260) builds `self.conds` (T3 and S3Gen reference dicts) once. Every later `generate()` reuses `self.conds`. So a fixed reference clip costs nothing per request once the model is prepared. Voice identity comes only from this reference (as the constraint notes).
- Watermarking (`perth`, CPU) runs inside every `generate()` call (L319), so it is paid per chunk.
- `fast_nano.py` (repo) shows the measured speed path: static KV cache plus `torch.compile(mode="reduce-overhead")` CUDA-graph decode step; warm-up compile takes "several seconds" (its README).

## Measured numbers already in the repo (nano_fast/README.md, RTX 3060 Laptop 6 GB, torch 2.6.0+cu126, 6.8 s clip)

- Decode step (greedy): 7.6 ms dynamic cache, 8.2 ms static eager, 5.3 ms static + CUDA graphs.
- End to end: baseline median 2.41 s, fast median 1.99 s; best-of-7 baseline 1.82 s, fast 1.84 s. The README says the laptop GPU is noisy (medians drift up to 0.5 s).
- Derived RTF (median / audio duration), inference from these numbers: baseline ~0.27 to 0.35, fast ~0.29. This is roughly 3x faster than real time. See the correction section below.
- Rough breakdown (inference): 6.8 s of audio is about 170 speech tokens at 25 Hz. 170 steps x 7.6 ms is about 1.3 s, so the autoregressive loop is most of the ~1.8 to 2.4 s. S3Gen plus watermark is the rest.

---

## Key Question 1: What gives the largest time-to-first-audio (TTFA) reduction for a non-streaming model?

### Takeaway
For a model with no streaming, TTFA equals the full generation time of whatever unit you send. Sentence-level chunking with the first chunk kept short is the largest lever that is under your control, then pipelining so playback of chunk N overlaps synthesis of N+1. Keeping the model resident and warm matters more than either if the first request otherwise pays load or compile cost (seconds, not milliseconds).

### Cited Findings
- Vendor and issue-tracker guidance: start synthesis on the first clause or sentence, pipeline it ahead of playback, and keep the first chunk small. Serial synthesis then playback was the diagnosed cause of slow first audio in one open issue; pipelining hides it, and a smaller first chunk trades off naturalness. — [Synapse issue #22](https://github.com/sahil-sharma-50/Synapse/issues/22) (snippet)
- The same issue reports that a lazily spawned TTS sidecar puts model load inside the user-visible latency window. — [Synapse issue #22](https://github.com/sahil-sharma-50/Synapse/issues/22) (snippet)
- Clause-level chunking at commas and conjunctions is the default in some hosted TTS APIs (ElevenLabs and Deepgram Aura, per one source). — [Smallest.ai latency budget](https://smallest.ai/blog/designing-voice-assistants-stt-llm-tts-tools-and-latency-budget) (snippet; secondary attribution, not verified against the vendors' docs)
- Cold start on CUDA is dominated by one-time work: Inductor compile, CUDA graph capture, lazy cuBLAS kernel selection. One paper reports a first forward pass of 12.18 s falling to 580 ms after graph breaks were fixed. — [Spheron cold start](https://www.spheron.network/blog/gpu-cold-start-llm-inference-2026/), [Spheron torch.compile guide](https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/), [GraphMend arXiv 2509.16248](https://arxiv.org/pdf/2509.16248) (figures from snippets, not fetched)
- Guidance to run 5 to 10 warm-up forward passes at server startup and exclude them from measurements. — [Spheron torch.compile guide](https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/) (snippet)
- Inductor cache survives process restarts in `~/.cache/torch/inductor/` but not PyTorch version upgrades. — [Spheron cold start](https://www.spheron.network/blog/gpu-cold-start-llm-inference-2026/) (snippet)
- Perceptual targets often cited: TTFA under about 200 ms feels responsive, under 100 ms feels instant. Vendor-stated targets range from 100 to 300 ms on hosted streaming endpoints. — [Gradium TTFA](https://gradium.ai/blog/time-to-first-audio), [Luke Oliff TTFA](https://lukeocodes.dev/tts-latency-time-to-first-audio), [Rime latency](https://docs.rime.ai/docs/latency) (snippets)

### Inferences (mine, for this hardware)
- With no streaming, sentence-chunked TTFA is roughly `T_load_if_any + T_gen(first_sentence)` instead of `T_gen(full_reply)`. For a reply of three similar sentences, the first sentence is about one third of the work. Gain is proportional to the number of sentences, not fixed.
- At RTF about 0.3 (measured above), a first sentence of about 2 to 3 s audio synthesises in roughly 0.6 to 1.0 s. Sub-200 ms TTFA is **not reachable** with this model in non-streaming mode on this GPU. Plan for roughly 0.5 to 1 s TTFA for the first sentence, and measure before committing to a number.
- Ranking of levers for this setup (inference, to be confirmed with measurement):
  1. Model resident and warm before the first request (removes seconds of load or compile). Largest effect if the app currently lazy-loads.
  2. First-sentence-first chunking with pipelining (removes most of the generation time for later sentences from the user's wait).
  3. Keep the first chunk short, e.g. one clause or one short sentence, but not shorter than a natural phrase (quality trade-off, see Q2).
  4. CUDA-graph decode from `fast_nano.py` (about 15 to 20% end to end on median, noisy). Worth it only after 1 to 3, and only if the warm-up cost is paid at startup.
  5. Watermark and loudness normalisation: CPU work per chunk, likely small, but measure it. Not optimised here.

### Gaps
- No source measured sentence-chunked TTFA for Chatterbox-Nano or any comparable non-streaming model on a 6 GB laptop GPU. The TTFA numbers above are my estimates from RTF, not measurements.
- I did not measure how much of the 2.4 s baseline is load versus generation in the app's actual startup path. Check how the app loads the model.

---

## Key Question 2: Audio quality risks of sentence chunking for voice consistency, and mitigations

### Takeaway
Cutting at sentence boundaries with a 30 to 50 ms fade avoids most clicks. Prosody still resets at each boundary because each chunk is generated in isolation. The reference voice stays fixed, since `conds` is shared, so the main drift is in pacing, intonation and energy, not timbre. Mitigations are structural (cut where the speech already pauses, keep the first chunk long enough, reuse conditioning), not model-side.

### Cited Findings
- Naive chunk-and-concatenate causes energy discontinuities, warbling at boundaries, uneven speaking rate, and lost intonation, because each chunk is generated in isolation. — [MagpieTTS-LF arXiv 2606.18485](https://arxiv.org/pdf/2606.18485) (snippet); also [Smallest.ai long-form](https://smallest.ai/blog/long-form-tts-how-to-generate-consistent-speech-at-scale) (snippet)
- Split at sentence boundaries so each chunk starts and ends on a complete grammatical unit. Never split inside a number, abbreviation, quoted phrase or pronunciation token. — [Smallest.ai long-form](https://smallest.ai/blog/long-form-tts-how-to-generate-consistent-speech-at-scale) (snippet)
- Fades of about 30 to 50 ms are common. Very broad crossfades smear consonants and can duplicate room tone. Flow-matching models can produce digital clicks and hallucinated breaths at segment edges; one paper applies a 50 ms linear fade-out to each segment and adds a short silence gap. — [open-unified-tts](https://github.com/loserbcc/open-unified-tts) (snippet), [Calliope arXiv 2602.10735](https://arxiv.org/pdf/2602.10735) (snippet, the 50 ms fade claim was in a search summary; check the paper before citing it)
- Paragraph breaks and similar natural pauses mask a seam. Insert pauses from document structure (sentence, paragraph, heading) rather than relying on fades alone. — [Smallest.ai long-form](https://smallest.ai/blog/long-form-tts-how-to-generate-consistent-speech-at-scale) (snippet)
- Overlap-and-seam approach: overlap chunks by 2 to 3 sentences, then choose a seam by cross-correlation or silence detection and crossfade. — source unclear from snippet (a "guide" in search results); treat as unverified.
- Research fix for prosody continuity: pass context across chunks at inference time (stateful inference over sentence chunks). Requires model-side changes, so not available to a wrapper around `generate()`. — [MagpieTTS-LF arXiv 2606.18485](https://arxiv.org/pdf/2606.18485) (snippet; a different model)

### Inferences
- Voice drift across chunks: the reference voice is identical for all chunks (`conds` reused), so timbre should be stable. Pace and intonation vary because T3 samples each chunk (`temperature=0.8`, `top_p=0.95`, `top_k=1000`, `repetition_penalty=1.2` defaults in `generate()`). Greedy decoding (`temperature` to 0 or top_k=1) would reduce per-chunk variance but may cause repetition or flat prosody. Test both on a fixed set of sentences and listen.
- Seeding each chunk with the same seed does not fix prosody continuity; it only makes runs reproducible. Do not rely on it for consistency.
- Per-chunk punc_norm adds a final full stop to any chunk that does not end in punctuation. Chunk boundaries must be at sentence-final punctuation, which also gives the natural fall the model expects at the end of a sentence.
- Overlap seams cost real compute: each overlapping sentence is synthesised twice. With RTF about 0.3 this is cheap for short replies, but it adds latency to every chunk. Use it only for long documents where the boundary artefacts are audible in your listening test.
- Recommended mitigations, cheapest first:
  1. Cut only at sentence-final punctuation; never cut inside numbers, abbreviations, or quotes (the cut is a regex on the text, not a model call).
  2. Keep the first chunk to one full sentence (or two short ones), not a clause. Short fragments give flat intonation.
  3. Insert a silence of about 80 to 200 ms between sentences (tune by ear; the existing trailing silence in S3Gen output is already a few tokens). Use a longer pause at paragraph breaks.
  4. Apply a 20 to 40 ms linear fade-in and fade-out to each chunk, and compare against 50 ms. The fade is a sample-level operation, so its CPU cost is negligible.
  5. Fix one reference clip and one set of sampling parameters for the whole reply; never re-run `prepare_conditionals` per chunk.
  6. If boundary artefacts remain, try the overlap-and-seam approach on long documents only.

### Gaps
- No measured data on Nano's chunk boundary artefacts. The 20 to 50 ms fade and 80 to 200 ms pause ranges are heuristics from sources about other models. Listen-test on this model before fixing values.
- No source covers Chatterbox-specific prosody-reset severity.

---

## Key Question 3: Does batching help or hurt latency at batch sizes relevant to one user?

### Takeaway
For one interactive user, batching hurts TTFA and does not help the reply's total latency unless the GPU is underused in a way the repo does not show. Keep batch size 1 with pipelined sentences for live speech. Use batching, if at all, for offline pre-rendering of fixed phrases.

### Cited Findings
- Batching trades throughput for latency. A batch takes longer to compute, and it also takes time to assemble and transfer. — [arXiv 2304.13541 (D-STACK)](https://arxiv.org/pdf/2304.13541) and [arXiv 2010.13103 (LazyBatching)](https://arxiv.org/pdf/2010.13103) (snippets)
- Dynamic batching adds a wait window. Under low traffic, a longer window only delays requests. — [arXiv 2010.13103](https://arxiv.org/pdf/2010.13103) (snippet)
- Past a knee point, larger batches buy little throughput and add latency. The knee depends on the GPU and the model. — [arXiv 2304.13541](https://arxiv.org/pdf/2304.13541) and [intuitionlabs](https://intuitionlabs.ai/articles/gpu-inference-performance-prefill-decode-batching) (snippets; the LLM-serving claims do not transfer directly to a 110M autoregressive TTS)
- Small batches: throughput scales almost linearly with concurrent requests for small batch sizes. — snippet from search summary, source not pinned. Treat as unverified for this model.

### Inferences
- The repo's `generate()` takes one text, and `inference_turbo` is called with a single `text_tokens` tensor. I did not confirm that `inference_turbo` or the S3Gen path accepts a padded batch. Batching would need code changes to T3 and to the per-sentence speech-token split. Treat batching as not available without that work.
- If batching were available: total time for N sentences in one padded batch is about the longest sentence's autoregressive steps, if the per-step GPU cost barely grows with batch size. At batch 1 the decode step is small (about 5 to 8 ms per step as measured), so it may be launch or memory-bound. If so, batch 2 to 4 could cost little extra per step. But TTFA gets **worse**: you must wait for the whole batch, so the first sentence is delayed to the batch's slowest member.
- Net for one user: batching at size N is a TTFA loss proportional to the batch's longest sentence. The only win is total wall time for the whole reply, and that matters less than TTFA for speech that plays while it is being generated.
- Pipelining at batch 1 gives most of the total-time benefit anyway: after the first chunk, synthesis runs ahead of playback (RTF about 0.3).

### Gaps
- No measurement of per-step decode time versus batch size for Nano. This is the number needed to judge batching properly; it requires a batched T3 path that does not exist in the repo.
- Could not find a source on batched TTS on consumer GPUs at batch 2 to 4.

---

## Recommended serving design for this hardware (inference from the findings above)

Single process, two threads plus one audio sink:

1. **Startup (once):** load `ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)`, call `prepare_conditionals(reference_clip)` once, then run 5 to 10 warm-up `generate()` calls on a short sentence (plus the CUDA-graph warm-up from `fast_nano.py` if adopted). Do this before accepting the first user request. Keep the model resident; do not lazy-load or unload between replies.
2. **Chunking (CPU, per reply):** split the text into sentences at sentence-final punctuation. Never split inside numbers, abbreviations, or quotes. Keep the first chunk to one full sentence. Do not split below sentence level.
3. **Synthesis worker (thread or process, batch 1):** calls `generate()` per sentence in order and pushes PCM to a queue. Keep the sampling parameters fixed for the whole reply.
4. **Post-process (CPU):** 16-bit PCM at 24 kHz, 20 to 40 ms fades per chunk, 80 to 200 ms inter-sentence silence (longer at paragraph breaks). Convert to int16 once; avoid encoding to MP3 or Opus per chunk (adds latency and a codec frame per chunk; the sink can be raw PCM).
5. **Playback (audio sink):** starts when chunk 1 is ready. A small device buffer (about 50 to 100 ms) keeps latency low. Underrun is avoided if each chunk's synthesis time is below its audio duration, which holds at RTF about 0.3 with margin for the noisy laptop GPU.
6. **Process model:** keep the GPU model in one process. If the UI or LLM runs in the same Python process, move synthesis (or playback) to a separate process with a queue, because the Python-side autoregressive loop, sampling, and per-call CPU watermarking hold the GIL. PyTorch CUDA kernels release the GIL, but the Python loop does not.
7. **Cache:** cache fully rendered WAVs for fixed phrases (greetings, confirmations). This is the only case where a batch render is worth it.

Expected latency effect of each element (inference, to be measured):

| Element | Expected effect on TTFA | Expected effect on total | Cost or risk |
|---|---|---|---|
| Model resident and warm | Removes seconds of load/compile on the first request (largest if currently lazy) | None | Holds VRAM (small for 110M) |
| Sentence chunking, first sentence first | Roughly divides first-audio wait by number of sentences | Small increase (per-call overhead) | Prosody reset at boundaries |
| Pipelining (synthesise N+1 while playing N) | Hides later chunks entirely | Small reduction | Needs queue discipline |
| CUDA-graph decode (`fast_nano.py`) | About 15 to 20% on median | Same | Warm-up compile cost, sampling not byte-identical |
| Cache reference conditioning | None per reply once prepared | None | None |
| Batch size > 1 | Worse (waits for batch) | Possibly better, unmeasured | Needs a new batched path |
| Fade + pause at boundaries | None | Adds the pause duration | Tuning by ear |
| PCM output, no per-chunk codec | Avoids codec frame latency | None | Larger bytes over a local link (negligible in-process) |

## Pitfalls

- **Voice drift** from per-chunk sampling (pace, intonation). Fix sampling parameters and the reference clip; listen-test.
- **Prosody reset** at every boundary. Mitigate by cutting at sentence boundaries and by pauses, not by cutting mid-sentence.
- **Silent truncation** of long text by the tokenizer `truncation=True`. Chunking avoids it, but do not send one oversized chunk. Check the tokenizer's max length.
- **Fake full stop** added by `punc_norm` to any chunk that does not end in `. ! ? - ,`. Only cut at real sentence ends.
- **Cold start counted in TTFA.** Measure the first request separately; it includes load, compile, cuBLAS selection.
- **Memory on 6 GB.** The 110M weights are small (fp32 about 0.44 GB, estimate from the earlier setup note; not measured). The real risk is the static KV cache, which `fast_nano.py` sizes to `L + max_new_tokens` per call. Allocate it once at the maximum length and reuse, to avoid fragmentation. Do not run two model instances to overlap synthesis unless measured. Watch `torch.cuda.max_memory_reserved()`.
- **Compile recompilation.** `torch.compile` with CUDA graphs recompiles or re-captures per shape. Pad or bucket the speech-token length, or keep it eager.
- **GIL.** Python sampling loop and watermark CPU work block other Python threads, including a Python audio callback. Keep the audio callback in C (e.g. a sink that reads from a ring buffer) or move synthesis out of process.
- **Laptop GPU noise and thermal throttling.** Medians drift up to 0.5 s between runs (repo README). Run many trials and report percentiles.
- **Reporting RTF.** Two conventions exist. Use synthesis time divided by audio duration (lower is faster). The repo's `fast_nano.py` uses this. State the convention in every report.
- **Time to first byte is not TTFA.** Measure to the first PCM sample handed to the audio device, not to the `generate()` return (the latter is the whole sentence).

## How to measure correctly

- **TTFA:** timer starts when the reply text is ready (after any LLM output that you include), stops when the first PCM sample is written to the audio device (or to a file-like sink in tests). Call `torch.cuda.synchronize()` before stopping the timer, otherwise asynchronous CUDA work is not counted.
- **Warm vs cold:** report the cold first request separately from warm requests. Discard 5 warm-up calls (sources suggest 5; the repo's `fast_nano.py` uses one warm-up). Run at least 20 warm trials per configuration; report p50 and p90, not best-of-N.
- **RTF:** per sentence, `synthesis_time / audio_duration`. Per reply, `total_synthesis_time / total_audio_duration`. The repo's measurement is per call with median of 3 runs; increase the count for decisions.
- **Pipeline health:** log, for each chunk, the time its synthesis finished versus when the previous chunk's playback ended. Any gap is an underrun (audible pause). This is the number that decides whether pipelining is enough.
- **Batching decisions:** measure per-decode-step time at batch 1, 2, 4 on the actual T3 path before deciding. Currently impossible without code changes.

---

## What is new or corrected versus `nano_official_setup.md`

This note does not repeat the official-setup facts (install pins, no streaming, no dtype knob, hard-coded `n_cfm_timesteps=2`, gated HF access, version conflict). New or corrected items:

1. **Corrected:** the official note lists "10x real-time on GPU" as unverified (search snippet only). The repo's own measurement (`nano_fast/README.md`) on this exact laptop GPU gives RTF about 0.27 to 0.35, roughly 3x faster than real time, not 10x. This is a measurement on one 6.8 s clip with a noisy GPU, not a general claim.
2. **New:** measured latency breakdown from the repo (decode step 5.3 to 7.6 ms; baseline 1.82 to 2.41 s for 6.8 s audio). Not in the official note.
3. **New:** `generate()` does not stream but also does not expose any per-sentence API, so chunking must be done by the wrapper. The official note said there is no batch API; this note confirms `inference_turbo` takes a single `text_tokens` tensor in the code path I read.
4. **New, pitfall:** `tokenizer(..., truncation=True)` silently truncates long text (L295). Not in the official note.
5. **New, pitfall:** `punc_norm` appends a full stop when text lacks terminal punctuation, so every chunk gets sentence-final cadence. Relevant to chunking; not in the official note.
6. **New:** watermarking (`perth`) runs on CPU inside every `generate()`, so it is paid per chunk.
7. **New:** `/workspace/chatnano/nano_fast/fast_nano.py` exists as a CUDA-graph decode path (about 15 to 20% end to end, noisy). Not in the official note. Its warm-up cost is "several seconds" per its README.
8. **Not verified, flagged:** `t3.py` has a `tqdm` progress loop around sampling (L338). I did not confirm it is on the `inference_turbo` path. If it is, progress output adds per-step CPU work; disable it in the serving build.

## Gaps summary

- No measured TTFA for sentence-chunked Nano. Numbers here are estimates from RTF.
- No measured batch scaling for Nano; batching path does not exist in the repo.
- No listening data on chunk-boundary artefacts for Nano; fade and pause ranges are heuristics.
- Sources are mostly vendor blogs and search snippets, not fetched in full. Specific figures (12.18 s to 580 ms, 15 to 20% from CUDA graphs) are repo-measured or snippet-reported as marked.
