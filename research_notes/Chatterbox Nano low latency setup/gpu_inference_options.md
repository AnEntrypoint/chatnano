# GPU Inference Acceleration Options for Chatterbox-Nano (110M) on RTX 3060 Laptop

Scope: read-only research. Nothing was installed and /workspace/chatnano was not modified.
Hardware: RTX 3060 Laptop GPU (6 GB, sm_86, driver 580.178.04), 12 CPU cores, 14 GB RAM, Linux, Python 3.11.17.
Note: no benchmark of Chatterbox-Nano on a laptop 3060 was found. All ranking below is inference from the cited sources plus first-principles reasoning, and must be confirmed with a local benchmark.

## Model structure (what the sources say)

- Resemble's page gives 110M params, trained on the same data as Chatterbox Turbo, zero-shot cloning from 10 s+ reference audio, and a PerTh watermark on every clip. It reports "10x real-time on GPU" and "3x real-time on CPU (8 threads)" with no hardware, batch, precision, or method given. [Resemble Nano page](https://www.resemble.ai/learn/models/chatterbox-nano)
- A community ONNX conversion splits the model into four graphs: embed_tokens (FP16), speech_encoder (Q4F16), language_model (Q4F16, autoregressive), and conditional_decoder (Q4, non-autoregressive acoustic decoder). It is non-streaming, English-only, and was only CPU smoke-tested. [KitsuMate ONNX card](https://huggingface.co/KitsuMate/chatterbox-nano-onnx)
- A community GGUF card reports a GPT-2-small-style backbone. [HF search result](https://huggingface.co/cstr/chatterbox-nano-GGUF)

Implication: the autoregressive language-model loop, which runs once per speech token at batch 1, is the part that determines latency. The decoder runs once per chunk.

## Key question answers

### Q1. Which 2-3 options give the biggest real win at batch 1 on a 3060 laptop?

1. **CUDA graphs (via torch.compile mode="reduce-overhead", or manual torch.cuda.graph capture of the decode step).** Biggest expected win for the AR loop. PyTorch describes `reduce-overhead` as using CUDA graphs to reduce Python and launch overhead, "useful for small batches." [PyTorch torch.compile docs (2.8)](https://docs.pytorch.org/docs/2.8/generated/torch.compile.html); [Torch-TensorRT CUDA graphs note](https://docs.pytorch.org/TensorRT/contributors/cuda_graphs.html) says CUDA graphs help most for batch-size-1 latency. A PyTorch blog on Seamless M4T-v2 found its text decoder and vocoder CPU-bound at batch size 1, and used torch.compile with CUDA Graphs to fix this. [PyTorch blog: accelerating generative AI 4](https://pytorch.org/blog/accelerating-generative-ai-4/) (the search snippet described this; the page itself was not fetched).
2. **Static KV cache plus compile.** Needed to make the AR loop graph-capturable, because a growing dynamic cache blocks kernel fusion. Transformers documents up to ~4x on a 7B model forward on A100, with about 3x end-to-end, and gains are smaller for smaller models. Model support is limited to some architectures in the versions reviewed. [HF llm_optims docs (4.41-4.44 era)](https://huggingface.co/docs/transformers/v4.44.2/en/llm_optims). This is only useful if Nano's LM can be run through Transformers-style generate; a custom loop needs a hand-written preallocated KV cache.
3. **fp16 autocast or weights in fp16.** Cheap to try and low risk. Ampere supports bf16 and fp16. Expect a moderate gain on the decoder and on any compute-heavy step, and little gain on the tiny per-token matmuls, where launch overhead dominates. NVIDIA forum evidence is mixed: one thread reports no fp16 vs fp32 difference for a TensorRT model. [NVIDIA forum](https://forums.developer.nvidia.com/t/no-performance-difference-between-float16-and-float32-optimized-tensorrt-models/185140)

### Q2. Is the bottleneck per-token Python/launch overhead or raw compute?

Most likely per-token overhead. This is an inference from the sources and arithmetic, not a measured result:
- At 110M params, fp16 weights are about 220 MB. A 3060 laptop has roughly 336 GB/s of memory bandwidth, so reading all weights once per token is about 0.7 ms. Actual per-token times are typically much higher than this in eager PyTorch, which points to launch and Python overhead (many small kernels per layer) rather than arithmetic or bandwidth.
- The NVIDIA CUDA graph guidance says the strongest candidates are workloads with many small kernels and under about 80% GPU utilization. [NVIDIA CUDA graphs troubleshooting](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html) (seen in search snippet)
- The Seamless M4T-v2 evidence above is the closest analogue found: a small decoder at batch 1 was CPU-bound.

To confirm locally: profile one decode step with torch.profiler or Nsight Systems. If GPU kernels occupy well under ~80% of the step wall time, CUDA graphs will pay off. If the GPU is saturated, switch to fp16 and check kernel efficiency.

### Q3. Recommended order and how to measure

Recommended order:
1. Baseline: eager fp32 and eager fp16, with a correct timing harness (below). Confirm the GPU is actually used.
2. Check graph breaks: `TORCH_LOGS=graph_breaks` (or `torch._dynamo.explain`). Remove `.item()`, data-dependent branches, and dynamic-length slicing in the AR loop. Graph breaks are the main pitfall for small models. [Search: torch.compile graph breaks](https://docs.pytorch.org/docs/2.8/generated/torch.compile.html)
3. torch.compile with mode="reduce-overhead" on the LM decode step, with a static KV cache and padded/bucketed lengths. Expect a one-time warmup/recompile cost; the Spheron blog claims 30-90 s for the first compile, which is a vendor claim and must be measured here. [Spheron blog](https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/)
4. Only if still needed: fp16 weights for the decoder and the rest of the pipeline, and possibly TensorRT or ONNX Runtime for the decoder (see below).

Measurement:
- RTF = total wall time to generate audio / duration of audio produced. Use `torch.cuda.synchronize()` before each timer read. Report the median over at least 10 runs after warmup, plus p90.
- Time-to-first-audio (TTFA) = time from request start to the first audio chunk leaving the pipeline. For a non-streaming model like the ONNX card, TTFA equals the full generation time. Count text preprocessing and reference-audio encoding separately, and cache reference-speaker embeddings if the voice is fixed.
- Use a fixed set of sentences of varying length. Measure cold start (first call after process start) separately from warm calls, because compile cost lands on cold start.
- Watch for shape-driven recompiles; a shape sweep should show whether the graph count stays flat.
- Track output quality (e.g., ASR round-trip WER or a listening check) for every change; fp16 and quantization can drift.

### Q4. Options not worth it at this model size (inferences, with evidence noted)

- **TensorRT / torch-TensorRT:** High setup cost (version-locked engines, separate build per GPU/TRT/CUDA version, the engine-portability caveat). Gains on a 110M AR model at batch 1 are uncertain because the bottleneck is launch overhead, which TRT does not remove unless the whole loop is captured. Best reserved for the conditional decoder if it is the largest chunk. No laptop-3060 small-TTS benchmark was found; a T4 batch-1 FP16 YOLO number exists but does not transfer. [HF model card, T4 FP16 latency](https://huggingface.co/thangkt/PCB-Prune-YOLO-TensorRT-FP16); [TensorRT forum: trtexec batch 1](https://forums.developer.nvidia.com/t/questions-about-using-tensorrt-batch-size/115338)
- **ONNX Runtime CUDA EP:** Viable for the decoder, but the Nano ONNX conversion tested CPU only, has no GPU numbers, and uses Q4 quantization whose accuracy has not been evaluated. Package pitfalls: `onnxruntime-gpu` must match the CUDA major version (CUDA 11 builds silently fall back to CPU on a CUDA 12 machine). Official docs say GPU packages from 1.27 are built with CUDA 13.0 by default and older ones with 12.8, so pin the version. cuDNN 8 and 9 builds are not interchangeable. [ORT CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html); [ORT install](https://onnxruntime.ai/docs/install/)
- **CTranslate2:** Supports encoder-decoder and decoder-only Transformers, mainly for ASR/MT (Whisper, T5). No TTS architecture was found in the sources. Not recommended unless a Nano-compatible port exists. [CTranslate2 README](https://github.com/OpenNMT/CTranslate2/)
- **int8 weight-only (bitsandbytes or torchao):** Only helps when decode is weight-bandwidth-bound, which a 110M model in fp16 (~220 MB) is not. A torchao int8wo benchmark shows 155 tokens/s, but its bf16 baseline was not clearly visible in the excerpt. A user report shows torchao Int8WeightOnlyConfig in eager mode was 2.5x slower and produced garbage output. Low priority. [torchao quantized inference docs](https://docs.pytorch.org/ao/main/workflows/inference.html); [torchao changelog](https://data.safetycli.com/changelogs/torchao)
- **Triton custom kernels:** Large effort, and no evidence found that they help a model this small; the gain is in fusion, which compile already provides. Not recommended.
- **CPU fallback with ONNX/OpenVINO:** Not needed for the GPU path. Resemble's own CPU figure is 3x real-time with 8 threads, which is a claim without hardware detail. Keep it only as a fallback for machines without a GPU.
- **Chatterbox-specific CUDA graph fork:** A GitHub fork (langfod/chatterbox) claims "advanced CUDA graph optimizations" with no measured numbers. Treat as unverified. [langfod/chatterbox releases](https://github.com/langfod/chatterbox/releases)

## Summary ranking (expected latency gain vs setup cost, for this model on this laptop)

| Rank | Option | Expected gain (unverified) | Setup cost | Main pitfalls |
|---|---|---|---|---|
| 1 | CUDA graphs / compile reduce-overhead on AR loop | Largest, if launch-bound | Medium | Graph breaks, recompiles per shape, cold-start compile time, memory use |
| 2 | Static KV cache | Enables #1; ~3x end-to-end reported for a 7B model, less for small | Medium | Model-specific support; custom loop may need manual cache |
| 3 | fp16 (or bf16) weights and autocast | Moderate on compute-heavy parts, low on tiny matmuls | Low | Accuracy drift; check audio quality |
| 4 | ONNX Runtime CUDA EP (decoder only) | Uncertain | Medium | CUDA-version pinning, silent CPU fallback, Q4 accuracy |
| 5 | TensorRT on decoder | Uncertain | High | Per-GPU engine builds, version lock |
| 6 | int8 weight-only | Low at this size | Medium | Eager-mode slowdowns reported; accuracy |
| 7 | CTranslate2 | Not applicable (no TTS arch found) | High | Architecture support |
| 8 | Triton kernels | Low | Very high | Maintenance |
| 9 | CPU ONNX/OpenVINO | Fallback only | Medium | Not a GPU speedup |

Install footprint: torch.compile and CUDA graphs need no new packages beyond the existing PyTorch (check that the installed torch build supports the CUDA version on the driver; driver 580 supports CUDA 13.0). ONNX Runtime GPU and TensorRT add several hundred MB to GB each. bitsandbytes and torchao are small-to-medium. CTranslate2 is small. Exact sizes were not checked in this session.

## Platform notes

- torch.compile, CUDA graphs and fp16 work on Linux, which is this machine. Windows support for Triton-based compile has historically been weaker, which does not affect this Linux setup.
- Nothing in the sources reviewed flagged a Linux-specific blocker for this hardware.

## Gaps

- No measured Chatterbox-Nano latency on any GPU, and no Nano-specific torch.compile or CUDA graph result. The Resemble figures are unverified headlines.
- No benchmark of TensorRT, ORT CUDA EP, or int8 for a small TTS model on a laptop RTX 3060. The TRT numbers found were for T4 or older GPUs and for vision models.
- Architecture of Nano's LM (exact layer count, whether it uses a standard Transformers class) was not confirmed. This decides whether the Transformers static-cache route applies. Check the official repo config.
- Current (2026) torch.compile docs for mode behavior were only seen through the 2.8 docs page; check the installed torch version's docs.
- The Seamless blog and the Transformers static-cache numbers come from other models and hardware.
- Bitsandbytes was not researched in detail; it is the expected int8 route but adds little here.

## Recommended next steps (no installs made)

1. Read the Chatterbox-Nano repo config and model code to determine whether the LM is a standard Transformers model and whether generate uses a KV cache.
2. Write a benchmark harness in a separate directory (not in /workspace/chatnano) measuring RTF and TTFA with synchronized timers, cold and warm, with a fixed sentence set.
3. Profile one AR decode step to confirm launch-bound vs compute-bound.
4. Only then try compile reduce-overhead with a static cache, then fp16.
</content>
</invoke>
