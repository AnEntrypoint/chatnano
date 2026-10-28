# Speed-up projects for Chatterbox (Nano / Turbo focus)

Research date: 2026-10-22. Local target: RTX 3060 Laptop (6 GB VRAM, sm_86, CUDA driver 580), 12 cores, 14 GB RAM.

## Key question 1: Which projects support Nano or Turbo?

### Takeaway
Official support: Nano and Turbo are both in the official repo and model cards. Nano is not in any third-party server we checked. Most community speed projects target only the original 500M model or the multilingual model. The only Nano-specific ports found are CoreML/Apple (iliasaz) and a Chatterbox-TTS-Server request (#162) that is still open.

### Cited Findings
- Official Nano: 110M, English, same architecture as Turbo, loaded via `ChatterboxTurboTTS` with `nano=True`, decoder distilled to one step; MIT — [HF model card](https://huggingface.co/ResembleAI/chatterbox-nano) (fetched via summary; page has no last-updated date)
- Official model table in resemble-ai/chatterbox README: Turbo 350M English; Nano 110M English; Multilingual V3 500M; original 500M English; no ONNX/TensorRT/compile/vLLM/streaming mention; 42 commits on master; no release date shown — [GitHub README](https://github.com/resemble-ai/chatterbox)
- Chatterbox-Turbo-ONNX (ResembleAI) covers Turbo only (350M), not Nano; uses onnxruntime with default CPU provider in the example, dtype options fp32/fp16/q8/q4/q4f16; no streaming; MIT — [HF ONNX card](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX)
- iliasaz/chatterbox-nano-coreml: CoreML conversion of Nano (T3 217M params, GPT-2 small width 768); iOS 18+/macOS 15+; ANE prefill; MIT (Swift runtime also MIT); no date — [HF](https://huggingface.co/iliasaz/chatterbox-nano-coreml)
- devnen/Chatterbox-TTS-Server issue #162 "req: support Chatterbox-Nano" — opened Aug 3 2026, still open; official server does not support Nano; a user ran a Docker build-time patch (Aug 6 2026), no PR — [Issue #162](https://github.com/devnen/Chatterbox-TTS-Server/issues/162)
- randombk/chatterbox-vllm: original 500M English + multilingual v2; no Turbo/Nano mention — [repo](https://github.com/randombk/chatterbox-vllm)
- hans00/Chatterbox-TTS-GGUF: converts original Chatterbox only (not Turbo/Nano); MIT — [HF](https://huggingface.co/hans00/Chatterbox-TTS-GGUF)
- Chatterbox-MLX fork (Apple Silicon): MLX-optimized fork, not Nano-specific — [GitHub fork](https://github.com/michaelcreatesstuff/chatterbox)
- mlx-community/chatterbox-turbo-fp16 (via mlx-audio): Turbo — [search summary](https://rapidmlx.com/docs/models/families/audio)
- LuckyKo and devnen Chatterbox-TTS-Server forks: original and Turbo; opt-in BF16 for T3 (`TTS_BF16=on|auto`), claims ~40% throughput on bf16-capable GPUs (devnen) — [devnen](https://github.com/devnen/Chatterbox-TTS-Server), [LuckyKo](https://github.com/LuckyKo/Chatterbox-TTS-Server) (search summaries; not verified in repo)

### Inferences
- The most mature "fast" path for Turbo is still the stock PyTorch code plus ONNX Runtime export by Resemble; no ONNX export for Nano has been published by Resemble.
- A Nano ONNX/TensorRT export would be a straightforward port of the Turbo ONNX pipeline, since the architecture is shared, but this is an inference, not a found project.

### Gaps
- No community repo found that explicitly supports Nano on CUDA or in a server.
- No maintainer-dated last-commit data for most repos (GitHub pages fetched without dates).

## Key question 2: Measured latency / RTF, and hardware

### Takeaway
Official Nano figure is "3x realtime on 8 CPU cores" (CPU, no model named precisely) and a vendor blog's "10x realtime on GPU" (GPU not named). No primary measured TTFA for Nano or Turbo exists. Community RTF numbers for the 500M model range from about 0.5 to 2.7 on consumer and datacenter GPUs. None are measured on RTX 3060 6 GB.

### Cited Findings
- Nano: "3x faster than realtime on 8 cores" — [HF card](https://huggingface.co/ResembleAI/chatterbox-nano), [GitHub README](https://github.com/resemble-ai/chatterbox)
- Resemble blog (published Jul 6 2026): Nano "10x realtime on GPU, 3x realtime on CPU using 8 threads"; GPU and CPU model not named; RTF about 0.1 GPU, 0.33 CPU derived; no streaming or TTFA reported; also Flash (vLLM "2x" vs AR baseline, no absolute numbers) — [Resemble blog](https://www.resemble.ai/resources/chatterbox-nano-and-flash-speed-at-the-edge-throughput-at-scale)
- Nano on Apple: iPhone 17 Pro Max prefill 22 ms (Nano) vs 64 ms (Turbo); Mac M5: Nano 4.16 s audio in 0.65 s (~6.4x realtime), Turbo 4.00 s in 1.02 s (~3.9x) — [iliasaz CoreML](https://huggingface.co/iliasaz/chatterbox-nano-coreml) (rough, nondeterministic)
- Turbo, Mac/MLX: sglang-omni PR #2243 benchmark MLX RTF 0.87 vs Torch/MPS 2.63 — [PR #2243](https://github.com/sgl-project/sglang-omni/pull/2243) (open, needs changes)
- chatterbox-mlx fork: "up to 2.4x faster" on Apple M4 32 GB, macOS 15.4 — [fork README](https://github.com/michaelcreatesstuff/chatterbox)
- vLLM port (original 500M): "~4x generation tokens/s without batching, >10x with batching," GPU not named in claim; benchmarks listed on RTX 3090 and RTX 3060 Ti but not tied to the claim; self-reported — [randombk/chatterbox-vllm](https://github.com/randombk/chatterbox-vllm)
- Original 500M, streaming fork (davidbrowne17), issue thread on resemble-ai/chatterbox:
  - Jul 17 2025 T4: first chunk 4.31 s, RTF 2.05 — [Issue #193](https://github.com/resemble-ai/chatterbox/issues/193)
  - Oct 22 2025 RTX 5090: first chunk 0.519 s, RTF 0.814, 2.996 s for a 63-char text (user log) — [Issue #193](https://github.com/resemble-ai/chatterbox/issues/193)
  - Jan 11 2026 Jetson Thor: RTF ~0.9, first chunk ~900 ms — [Issue #193](https://github.com/resemble-ai/chatterbox/issues/193)
  - Sep 2025 A100: 13.32 s per sentence (non-streaming) — [Issue #193](https://github.com/resemble-ai/chatterbox/issues/193)
  - Community "0.499 RTF / 0.472 s to first chunk on a 4090" for davidbrowne17/chatterbox-streaming — [search summary](https://github.com/davidbrowne17/chatterbox-streaming)
- Issue #71 (3090): RTF 2.75 "too slow"; dynamic quantization did not help — [Issue #71](https://github.com/resemble-ai/chatterbox/issues/71) (search summary)
- Issue #244 (RTX 5090): multiple workers slow each other; chunking/streaming near 1:1 with playback for one user — [Issue #244](https://github.com/resemble-ai/chatterbox/issues/244) (search summary)
- Picovoice on-device benchmark: Chatterbox-TTS-Turbo 48,281 ms first-token-to-speech on CPU, non-streaming — [Picovoice](https://picovoice.ai/blog/on-device-tts/) (search summary; likely PyTorch path)
- "sub 200 ms" latency: vendor claim for Resemble's hosted service, not the open model — [HF card](https://huggingface.co/ResembleAI/chatterbox-nano), [ONNX card](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX)
- Maintainer reply (Aug 14 2025): model released as from research team, denied intentional de-optimization — [Issue #193](https://github.com/resemble-ai/chatterbox/issues/193)

### Inferences
- Realistic expectation for Turbo on this 3060 Laptop GPU is around RTF 0.8–1.0 for one stream (extrapolating from 5090 and 3090 numbers; no direct measurement). This is an estimate, not a source figure. A 3060 Laptop is slower than a 3090 or 5090.
- For a sub-second first-chunk target, the streaming fork numbers (~0.5 s on 4090/5090) are the best-documented evidence, but they are for the 500M model.

### Gaps
- No measured Nano or Turbo TTFA on any NVIDIA laptop GPU, and no RTX 3060 numbers found.
- Resemble's "10x realtime on GPU" does not name the GPU.

## Key question 3: Compatibility with a 6 GB laptop GPU

### Takeaway
Nano (110M) and Turbo (350M) should fit in 6 GB with room for the CUDA context, since the original 500M model already runs on consumer GPUs. No source reports 6 GB VRAM use for Turbo or Nano. Techniques that depend on large VRAM budgets (vLLM, multi-worker batching) are risky on 6 GB.

### Cited Findings
- Turbo "lower compute and VRAM" with no numbers — [HF card](https://huggingface.co/ResembleAI/chatterbox-nano)
- ONNX Turbo supports fp16 and q8/q4 weight dtypes — [ONNX card](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX)
- Nano CoreML T3 package: 120 MB at 8-bit palettization (Turbo 328 MB) — [iliasaz](https://huggingface.co/iliasaz/chatterbox-nano-coreml)
- devnen server BF16 mode for T3 (`TTS_BF16`) — [devnen](https://github.com/devnen/Chatterbox-TTS-Server) (search summary)
- randombk vLLM port: CUDA graphs disabled in benchmarks, Linux/WSL2 Nvidia only, relies on vLLM internal APIs, likely pinned to vLLM 0.9.2/0.10.0 — [repo](https://github.com/randombk/chatterbox-vllm)
- hans00 GGUF: T3 and S3G in F16, Q8_0, Q5_K_M, Q4_K_M; needs stock llama.cpp b8095+ plus codec.cpp — [HF](https://huggingface.co/hans00/Chatterbox-TTS-GGUF) (original 500M only)

### Inferences
- Likely compatible on 6 GB: fp16/bf16 PyTorch (RTX 3060 supports bf16 on sm_86), ONNX Runtime CUDA EP fp16 for Turbo, chunked streaming with the streaming fork.
- Risky on 6 GB: vLLM (preallocates KV cache, needs `gpu_memory_utilization` tuning), multi-stream batching.
- Not applicable: TensorRT FP8 paths (Ampere lacks FP8 tensor cores).

### Gaps
- No measured VRAM figures for Turbo or Nano.

## Key question 4: Streaming and time-to-first-audio

### Takeaway
Official Nano/Turbo releases have no documented streaming mode. Streaming exists only in community forks for the original 500M model, with first-chunk latency of about 0.5 s on 4090/5090-class GPUs and about 0.9 s on Jetson Thor. A token-level streaming request for Turbo is open with no answer (Oct 10 2026).

### Cited Findings
- Streaming forks for the original 500M model (see Key Q2 figures) — [Issue #193](https://github.com/resemble-ai/chatterbox/issues/193), [davidbrowne17/chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming)
- grace-v2 issue #24 (Oct 10 2026, open, no reply): non-streaming PyTorch 2–3 s per sentence; asks for token-level streaming or Turbo ONNX export; RTF 0.44 reported without hardware — [Issue #24](https://github.com/MikkelBlom/grace-v2/issues/24)
- Issue #193 maintainer answer: none on streaming
- HF ONNX card: no streaming; generates all tokens then decodes — [ONNX card](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX)
- randombk vLLM: no server or streaming API; S3Gen is the bottleneck — [repo](https://github.com/randombk/chatterbox-vllm)
- Community "sub-200 ms" claims come from non-open or unmeasured sources

### Inferences
- Time-to-first-audio on Turbo with a chunked streaming wrapper is probably 0.5–1.5 s on this laptop GPU. This is an estimate.

### Gaps
- No measured TTFA for Nano on any GPU.

## Other categories checked

- TensorRT: no Chatterbox TensorRT build found. The only TensorRT TTS hit was EmotiVoice (different model). — search only
- CTranslate2: no port found — search only
- torch.compile / CUDA graphs: CUDA graphs disabled in randombk vLLM port due to correctness issues; no torch.compile results found.
- Core ML / MLX: iliasaz (Nano CoreML), mlx-community turbo-fp16, Jimmi42 HF repos (PyTorch/MPS, not MLX), sglang-omni PR #2243 (open).
- llama.cpp: hans00 GGUF (original only), NVIDIA NVIGI plugin (GGML; model variant not stated).
- Servers: devnen and LuckyKo Chatterbox-TTS-Server (OpenAI-compatible; original and Turbo; Nano open request #162).

## Caveats and quality
- Nano and Turbo support paralinguistic tags per HF card ([cough], [laugh], [chuckle]). Voice cloning from a short clip is claimed in a third-party blog (oakgen review), not verified in a primary source.
- Quality: vLLM port documents artifacts from rudimentary chunking and missing positional embeddings in multilingual mode; GGUF requires a host tokenizer.
- Watermark: Perth watermark is applied by upstream Chatterbox (CoreML card).

## Source reliability notes
- Resemble blog date Jul 6 2026 is the only dated official speed page found; the HF and GitHub pages show no last-updated dates.
- Many numbers come from GitHub issue users with unstated setups; treat as indicative.
- Aggregator and review sites (oakgen, pinggy, codersera, picovoice) were used only for cross-checks.
- Anything older than 2026 (issue #71, #193, #244 entries from 2025) should be flagged as possibly outdated.

## Recommended next steps for this machine
1. Benchmark Turbo bf16 and fp16 on the 3060 Laptop, first-chunk and RTF, with the streaming fork as the chunking reference.
2. Test ONNX Turbo (ResembleAI/chatterbox-turbo-ONNX) with ONNX Runtime CUDA EP for comparison.
3. Export Nano to ONNX yourself, since no published Nano ONNX was found.
4. Skip vLLM on 6 GB unless headroom is measured first.
