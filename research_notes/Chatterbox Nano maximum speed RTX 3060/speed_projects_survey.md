# Chatterbox speed projects survey (Nano focus, RTX 3060 Laptop target)

Research date: 2026-10-23. Target: RTX 3060 Laptop (6 GB VRAM, sm_86), 12 cores, 14 GB RAM.
Scope: projects not already covered in `Chatterbox Nano low latency setup/speed_projects.md`. Anything repeated from that file is marked "(already in prior notes)". Corrections and new items are labelled.

Method: 5 standard WebSearch queries, 4 WebFetch checks on model cards and repos. Last-commit dates were NOT verified for most repos (GitHub pages fetched without dates); treat maintenance status as unknown unless stated.

## Corrections to prior notes

- CORRECTION: prior notes say "no ONNX export for Nano has been published." Officially true (Resemble has none), but a third-party Nano ONNX exists: `KitsuMate/chatterbox-nano-onnx` (see below). It is not official and has no speed numbers.
- CORRECTION: prior notes list "Nano is not in any third-party server." Still true for servers, but Nano is in third-party ONNX and CoreML packages (new).
- CORRECTION: prior notes say Turbo "lower compute and VRAM" with no numbers. NVIDIA's plugin card gives Turbo TTS-only VRAM of about 2.3 GB (new, see Section 3).
- NOTE: the prior "Nano CoreML T3 package 120 MB at 8-bit" item is from iliasaz; a second Nano CoreML repo exists (`seanll95/chatterbox-nano-coreml`, seen in search, not fetched).

## 1. Projects, ranked by likely benefit on RTX 3060 Laptop

Ranking reflects expected real-world benefit on this card, not claimed speedup.

### 1a. Nano ONNX (third-party): KitsuMate/chatterbox-nano-onnx (NEW)
- URL: https://huggingface.co/KitsuMate/chatterbox-nano-onnx
- Converter: `owensong`, from official Nano checkpoint. Mirror of a conversion.
- License: MIT (weights Copyright Resemble AI, 2025).
- Dtypes: text/speech embeddings FP16; reference encoder Q4F16; AR LM Q4F16; conditional decoder Q4.
- Runtime targets: Unity and ONNX Runtime. Example CPU smoke test only.
- Claimed speed: none. Quality: "no representative speech-quality or device-compatibility evaluation."
- Streaming: no. English only.
- Nano support: yes (it is Nano).
- Maintenance: unknown; no date on page. Explicitly "not an official Resemble AI ONNX release."
- Relevance: a ready Nano ONNX graph to test on CUDA EP (ORT GPU build) if you do not want to export yourself. Needs its own quality check (voice identity) against the PyTorch baseline.

### 1b. Official Turbo ONNX + community mirror (already in prior notes, plus mirror)
- Official: https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX (already in prior notes; Turbo only).
- Mirror: `KitsuMate/chatterbox-turbo-onnx` (Q4F16 profile for Unity/ORT). Mirror of official; no new speed data.
- Turbo only (not Nano).

### 1c. Nano CoreML ports (Apple only, not usable on RTX 3060)
- `iliasaz/chatterbox-nano-coreml` (already in prior notes; numbers Mac M5 and iPhone, not NVIDIA).
- `seanll95/chatterbox-nano-coreml` (NEW, seen in search only; not fetched; no numbers).
- Relevance on this machine: none, Apple-only.

### 1d. ResembleAI Chatterbox-Flash (NEW; separate model, not Nano/Turbo)
- URL: https://huggingface.co/ResembleAI/chatterbox-flash
- Architecture: block-diffusion T3 (Llama-520M class, "25 Hz, 0.5B"), S3Gen vocoder, FlashInfer paged KV cache, CUDA-graph capture. vLLM not mentioned.
- License: MIT.
- Streaming: yes, native block-by-block; emits audio per committed block.
- Claimed numbers (concurrency 1, 50 utterances, GPU NOT named): TTFP 118 ms, RTF 0.107 (D=16, alpha 0.5); TTFP 103 ms, RTF 0.076 (D=32, alpha 0.75).
- Apple M4 MLX: RTF 0.778; MLX 4-bit RTF 0.665.
- Quality (Seed-TTS test-en): WER 1.96 to 2.04 vs 2.20 for Chatterbox AR; SIM-o 0.704; UTMOS about 4.08. Vendor-reported.
- Nano: not mentioned. Maintenance: a maintainer demo Space exists; no update date.
- Relevance: the only official, streaming, low-TTFP option found. Not a drop-in for Nano; a different checkpoint and a different voice-identity path. Hardware for the RTF claim is unknown, so it cannot be transferred to the 3060 without a test.

### 1e. Cornball-ai R port (NEW; allocator tuning)
- URL: https://github.com/cornball-ai/chatterbox
- R port of resemble-ai/chatterbox, parity with chatterbox-tts 0.1.7. Supports standard English and Turbo; not multilingual.
- Claim (as written): `tune_gc = TRUE` default gives "roughly 6x speedup"; "~85%" of inference time in R garbage collection without tuning. Hardware not named. The vignette with numbers was not visible.
- Note: the search summary mentioned a "reserved rate" setting; the repo page does NOT mention it. Do not rely on that detail.
- License: not stated on page. Last commit: not shown.
- Relevance to Python: the underlying lesson (allocator fragmentation or GC pressure on a small VRAM card) may transfer. In PyTorch the analogue is `PYTORCH_CUDA_ALLOC_CONF` (e.g. `expandable_segments:True`) and avoiding repeated alloc/free in the AR loop. This is an inference, not a tested result. The R GC cause does not apply to Python.

### 1f. NVIDIA ACE / NVIGI Chatterbox plugin (NEW)
- URL: https://docs.nvidia.com/ace-for-games/chatterbox-tts/nvigi-chatterbox-model-card.html
- Version 1.0.0 (Jan 2026), NVIDIA proprietary plugin license; base models MIT (Turbo).
- Turbo TTS-only VRAM about 2.3 GB; Multilingual about 3.5 GB; full pipeline 8.5 GB+. Requires CC 7.0+, driver r580+. 16 GB system RAM minimum.
- Latency: no measured figures; "designed for sub-200 ms" is a design target. Turbo "roughly 2x faster than Multilingual" on same backend, unquantified.
- Multilingual LLaMA backbone Q4_0; S3Gen-Multilingual F16. Turbo precision not stated.
- Relevance: the only vendor VRAM figure for Turbo. Suggests Turbo fits 6 GB with headroom. Not Nano. Not a Python path.

### 1g. Other items (search-only, not verified; low priority)
- `MaciejBratko/Chatterbox-TTS-Server-OpenaiAPI-Compatible`, `travisvn/chatterbox-tts-api`, `jianchang512/chatterbox-api`: OpenAI-compatible wrappers. Streaming not confirmed for any. Nano support not checked. Last commit unknown.
- `BrunBrand/Chatterbox-TTS-Server-mt`: community server fork; no details fetched.
- `soniqo/Chatterbox-LiteRT` (HF): Chatterbox T3 LLM graphs in int8 weights with fp32 compute; audio path kept fp32 for precision. No GPU timings. Mobile/LiteRT target. Low relevance.
- `Dvirile/chatterbox-turbo-lora` (HF): LoRA-aware Turbo ONNX, q4f16, for Android/iOS via ORT. No speed figures. Turbo only.
- `Jimmi42/chatterbox-tts-apple-silicon` (HF space): Apple adaptation. Not relevant to NVIDIA.

## 2. Quantization evidence (for T3 and S3Gen)

- A CoreML port (not NVIDIA; iliasaz repo, already in prior notes) reports: int8 on T3 halves weight size but runs at 0.95x to 0.99x speed (a memory lever, not a speed lever). int8 on S3Gen raised log-spectral distance vs fp32 from 0.034 to 0.354, so it was rejected. Source: https://huggingface.co/iliasaz/chatterbox-nano-coreml (figures came from a search summary in this session, not a direct fetch; re-check before citing).
- LiteRT and ONNX Q4 Nano both quantize the decoder/LM; neither publishes quality or speed comparisons.
- No CUDA int8/int4 speed measurement on RTX 3060 found.
- Inference: on a 6 GB card, quantization is more likely to help with VRAM headroom than tokens/s. Keep the S3Gen decoder in fp16 or fp32 and check speaker similarity before any int8 decoder.

## 3. Techniques checked (general)

- torch.compile / CUDA graphs: no Chatterbox-specific result found. Cache-DiT docs warn that CUDA graphs with `dynamic=True` can break and that captured output buffers may be overwritten by later replays. The AR loop has varying shapes, so graph capture needs static KV-cache buffers (inference; unverified for Chatterbox). Chatterbox-Flash uses CUDA-graph capture with a paged KV cache, which is the closest working precedent (vendor claim).
- randombk vLLM port (already in prior notes): CUDA graphs disabled there due to correctness issues.
- BF16 on sm_86 (devnen/LuckyKo, already in prior notes): Ampere supports bf16; no independent measurement found.

## 4. Quality and regression notes (new)

- KitsuMate Nano ONNX: no quality evaluation; explicitly not device-tested. Treat as unvalidated.
- Chatterbox-Flash: vendor-reported WER and SIM-o with no speaker similarity on a Nano-equivalent baseline. Voice identity drift has to be tested on this project's reference clip.
- Quantized T3 (int8) on CoreML: speed no better; S3Gen int8 degraded spectra (see Section 2).
- No reports of voice-identity drift from a speed-up were found for Nano specifically.

## 5. Gaps

- No Nano-specific CUDA speed project found (no TensorRT, no vLLM, no torch.compile, no server with Nano).
- No RTF measured on RTX 3060 for any Chatterbox variant.
- Last-commit dates unverified for: cornball-ai, KitsuMate, Flash, devnen, MaciejBratko, travisvn, jianchang512.
- Chatterbox-Flash hardware unknown; Flash is a different model and does not run Nano weights.
- WebSearch returned some results only as summaries; the Flash "concurrency 1" figures are from the model card as fetched.

## 6. Recommended next steps (additions to prior notes)

1. Test `KitsuMate/chatterbox-nano-onnx` on ORT CUDA EP (fp16 decoder variant if available). Compare RTF and speaker similarity with the PyTorch Nano baseline on the same reference clip.
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the Nano PyTorch baseline and measure VRAM/RTF before any other change (cheap test, inspired by the cornball allocator finding).
3. Evaluate Chatterbox-Flash only as a separate streaming candidate for TTFP; do not expect Nano weights to run in it.
4. Keep the S3Gen decoder out of int8 until a speaker-similarity check passes.
