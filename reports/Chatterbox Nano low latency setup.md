# Benchmark eager fp32 first, then compile

The fastest safe path for Chatterbox-Nano on this RTX 3060 Laptop is the official stack run as-is: Python 3.11, the pinned `torch==2.6.0` / `transformers==5.2.0` set from the repo, `device="cuda"`, and fp32 weights, which is the only dtype the official code uses. **No source in the research notes contains a measured Nano latency on any GPU**, so every speed number for Nano is unknown until this machine measures it. The obvious levers (CUDA graphs via `torch.compile(mode="reduce-overhead")`, a static KV cache, fp16) are generic PyTorch techniques that the sources only support by analogy from other models. The community speed projects are almost all built for the 500M original model or Turbo, not Nano. Nano-specific acceleration in the sources is limited to a CoreML port for Apple hardware and an unverified ONNX conversion. The first job is a benchmark harness, not an install of a speed project.

## Recommended setup for this machine

Create the environment from the local checkout (HEAD `5de7a54`, 2026-07-21, `pyproject.toml` version 0.1.7, which I confirmed on disk). The pins are `torch==2.6.0` and `torchaudio==2.6.0` for Python below 3.14, `transformers==5.2.0`, `diffusers==0.29.0`, `numpy>=1.24,<2.0`, `safetensors==0.5.3`, and `librosa==0.11.0` ([pyproject](https://github.com/resemble-ai/chatterbox)). The HF card says Nano was "Tested on Python 3.11 and Debian 11" ([HF model card](https://huggingface.co/ResembleAI/chatterbox-nano)), which matches this machine.

```bash
python3.11 -m venv /workspace/chatnano/.venv-nano
source /workspace/chatnano/.venv-nano/bin/activate
# Optional: pin the CUDA build of torch explicitly (notes did not verify the wheel index for sm_86)
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/resemble-ai/chatterbox.git && cd chatterbox && pip install -e .
pip show chatterbox-tts   # record the installed version
```

Nano is **gated on Hugging Face**. The model card requires agreeing to share contact information before access ([HF model card](https://huggingface.co/ResembleAI/chatterbox-nano)). Accept the form under the account that owns the token, then run `huggingface-cli login` before the first call. The coordinator will attempt the download; if the gate has not been cleared, nothing below runs. The code downloads `*.safetensors`, `*.json`, `*.txt`, `*.pt`, and `*.model` into the standard HF cache (`~/.cache/huggingface` unless `HF_HOME` is set).

The loading call is `ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)`. It takes only `device` and `nano`. There is no dtype argument, and the code never calls `.half()` or `.bfloat16()`, so the model runs in fp32 ([local code notes](/workspace/chatnano/research_notes/Chatterbox%20Nano%20low%20latency%20setup/nano_official_setup.md)). Run the official example first (`example_tts_nano.py` in the repo) to confirm the GPU is in use.

What to benchmark first, in this order:

| Step | What to measure | Why |
|---|---|---|
| 1. Eager fp32 baseline | Cold start (first call) and warm calls, separately; generation time per sentence; real-time factor (RTF = wall time / audio duration) | Nothing is measured yet, and cold start includes any compile cost later |
| 2. Timer discipline | `torch.cuda.synchronize()` before each read; median and p90 over at least 10 warm runs; a fixed set of sentences of varying length | Required for valid GPU timing |
| 3. Profile one step | `torch.profiler` or Nsight Systems on the autoregressive loop | Tells you whether the GPU is launch-bound (CUDA graphs help) or compute-bound (fp16 helps) |
| 4. Eager fp16 (optional) | Same harness, after `.half()` on the modules | Unsupported by Resemble; check audio quality (ASR round trip or listening test) |
| 5. Compile (only if launch-bound) | `torch.compile(mode="reduce-overhead")` on the LM decode step; watch `TORCH_LOGS=graph_breaks` and shape-driven recompiles | The notes rank this highest, but it is unmeasured |

TTFA is the metric to track, and it is the full generation time for Nano. The model has no streaming generator, so "first audio" equals "all audio" ([nano_official_setup.md](/workspace/chatnano/research_notes/Chatterbox%20Nano%20low%20latency%20setup/nano_official_setup.md)). Count text preprocessing and reference-audio encoding separately, and cache the voice conditionals if the voice is fixed. Keep a quality check for every change, since fp16 and quantization can drift.

## Community projects and what they mean for Nano

Nano has no speed project built for CUDA. The table below sorts the surveyed projects by whether they apply to Nano on this machine.

| Project | Models covered | Platform | Applies to Nano here? |
|---|---|---|---|
| Official repo (`nano=True`) | Nano, Turbo, 500M | CUDA, CPU | Yes, this is the baseline |
| ResembleAI Turbo ONNX | Turbo (350M) only | ONNX Runtime, CPU default in example; fp16/q8/q4 | No. No Nano ONNX is published by Resemble |
| KitsuMate Nano ONNX | Nano (community conversion) | ONNX, CPU smoke-tested only | Partly. Non-streaming, Q4 accuracy unevaluated; no GPU numbers |
| iliasaz Nano CoreML | Nano | Apple iOS 18+/macOS 15+ | No (Apple only) |
| Chatterbox-MLX forks, mlx-community Turbo fp16 | Turbo, original | Apple Silicon | No |
| randombk vLLM port | Original 500M | Linux/WSL2 CUDA | No. CUDA graphs disabled, relies on vLLM internals, risky on 6 GB |
| davidbrowne17 streaming fork | Original 500M | CUDA | Architecture reference only. Its chunked streaming is the only documented first-chunk path, and it is not for Nano |
| devnen / LuckyKo Chatterbox-TTS-Server | Original, Turbo | CUDA | Turbo only. Nano support request (#162) is open |
| langfod CUDA-graph fork | Not stated | CUDA | Unverified. Claims "advanced CUDA graph optimizations" with no numbers |
| hans00 GGUF | Original only | llama.cpp | No |

The sources show a few speed claims worth knowing, all unverified for Nano. The devnen and LuckyKo servers offer an opt-in BF16 mode for the T3 stage that claims about 40% throughput on bf16-capable GPUs, from search summaries not checked in the repo ([devnen](https://github.com/devnen/Chatterbox-TTS-Server)). The davidbrowne17 fork reports first-chunk times of 0.519 s at RTF 0.814 on an RTX 5090 and about 0.47 s on a 4090 for the 500M model, but those are user logs and summaries, and they do not transfer to a 6 GB laptop GPU ([issue #193](https://github.com/resemble-ai/chatterbox/issues/193); [davidbrowne17/chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming)). The grace-v2 report of RTF 0.44 has no hardware named ([grace-v2 issue #24](https://github.com/MikkelBlom/grace-v2/issues/24)).

## Optimization ranking for Nano on this laptop

Ranked by expected latency gain against setup cost. The gains are inference from the cited PyTorch and CUDA sources, not from Chatterbox measurements.

| Rank | Option | Applies to Nano? | Expected gain | Setup cost | Main risk |
|---|---|---|---|---|---|
| 1 | Benchmark baseline (eager fp32) | Yes | Zero, but it is the yardstick | Low | None |
| 2 | CUDA graphs via `torch.compile(mode="reduce-overhead")` on the LM loop | Likely, if launch-bound | Largest, if the step is launch-bound | Medium | Graph breaks (`.item()`, data-dependent branches), recompiles per shape, cold-start compile time. Spheron claims 30-90 s for a first compile, a vendor figure ([Spheron](https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/)) |
| 3 | Static KV cache | Only if the LM runs through Transformers-style `generate` | Enables rank 2. Transformers reports up to about 4x forward speedup on a 7B model, less for small models ([HF docs](https://huggingface.co/docs/transformers/v4.44.2/en/llm_optims)) | Medium | Needs a hand-written cache for a custom loop |
| 4 | fp16 or bf16 weights and autocast | Unverified | Moderate on compute-heavy parts, little on tiny matmuls | Low | Accuracy drift; NVIDIA forum evidence is mixed ([thread](https://forums.developer.nvidia.com/t/no-performance-difference-between-float16-and-float32-optimized-tensorrt-models/185140)) |
| 5 | ONNX Runtime CUDA EP for the decoder | Speculative (no Nano GPU test) | Uncertain | Medium | `onnxruntime-gpu` must match the CUDA major version; a mismatch can silently fall back to CPU |
| 6 | TensorRT for the decoder | Speculative | Uncertain | High | Per-GPU engine builds; no small-TTS laptop 3060 benchmark found |
| 7 | int8 weight-only | Unlikely to help | Low for a 110M model | Medium | One user report of 2.5x slower eager mode and garbage output |
| 8 | CTranslate2, Triton kernels, vLLM | No | Not applicable or low | High | No TTS architecture support found; 6 GB budget risks |

Two of the community tactics that matter most, a streaming wrapper and a server port, are not latency reducers for Nano as such. They change how the audio is delivered. A chunked wrapper would also need care, since the owensong ONNX author warns that concatenating independently decoded chunks causes pauses and prosody breaks ([nano_official_setup.md](/workspace/chatnano/research_notes/Chatterbox%20Nano%20low%20latency%20setup/nano_official_setup.md)).

## Facts, inferences, and unmeasured claims

**Verified from sources:** Nano is 110M parameters, English, MIT, the same class as Turbo with `nano=True`, using a `GPT2_small` backbone and `t3_nano_v1.safetensors`. Its speech-to-mel decoder is hard-coded to two CFM steps in `generate()`, and there is no step-count knob ([nano_official_setup.md](/workspace/chatnano/research_notes/Chatterbox%20Nano%20low%20latency%20setup/nano_official_setup.md)). Nano has no streaming generator and no `torch.compile` or CUDA-graph usage in the repo. The official pins above hold on master and in the local checkout.

**Inference:** Nano's autoregressive loop runs once per speech token at batch 1, so it likely dominates latency. The 3060 Laptop's roughly 336 GB/s bandwidth makes weight reads negligible, which points to launch overhead rather than compute. Both inferences need the profile in step 3.

**Unmeasured:** Every Nano latency, RTF, and VRAM figure on this or any GPU. Resemble's "10x realtime on GPU" for Nano names no GPU, batch size, or precision, and the official 3x figure is for 8 CPU cores ([Resemble blog](https://www.resemble.ai/resources/chatterbox-nano-and-flash-speed-at-the-edge-throughput-at-scale); [HF card](https://huggingface.co/ResembleAI/chatterbox-nano)). The only Nano timing on record is the iliasaz Mac M5 run (4.16 s of audio in 0.65 s), which is rough, nondeterministic, and Apple-only ([iliasaz CoreML](https://huggingface.co/iliasaz/chatterbox-nano-coreml)). Expect no VRAM figure either: at 110M parameters, memory should fit in 6 GB, but that is unmeasured.

## Conflicts and open questions

**Version conflict.** The local checkout and master `pyproject.toml` both say 0.1.7, but PR #559 "Bump version to 0.2.0" appears closed on Sep 10, 2026. Either it was never merged or the fetch is stale. Record `pip show chatterbox-tts` after install and pin that version.

**Resemble's GPU claim.** The "10x realtime on GPU" figure has no hardware named. Treat it as a headline until measured.

**ONNX availability.** The KitsuMate Nano ONNX card exists, but a separate note concluded no Nano ONNX was published. The KitsuMate card is community work and CPU-only, so the conflict is about what counts as "published," not about a verified GPU path.

**Parameter count.** The CoreML card lists the Nano T3 at 217M parameters, while Resemble and the local notes say the whole model is 110M. This is unreconciled, and it affects any memory estimate.

**Open questions for the benchmark.** Does the Nano LM use a standard Transformers class (which decides whether the static-cache route applies)? Does `t3.inference_turbo` expose a max-token setting? Does the installed torch build support sm_86 under driver 580 (the notes say driver 580 supports CUDA 13.0, and cu124 wheels include sm_86, but neither was tested)? Is the HF gate cleared for this account?

## Conclusion

The honest answer to "what is the lowest-latency setup" is that it is not yet known for Nano, and the sources cannot supply it. The practical path is a clean official install pinned to torch 2.6.0 and transformers 5.2.0, a synchronized benchmark in fp32 that separates cold and warm runs, and a profile that decides between CUDA graphs and fp16. Only after that should anyone reach for the community projects, and most of them (streaming forks, vLLM, GGUF, CoreML, MLX) target the 500M model, Turbo, or Apple silicon. The one Nano-specific tool that might matter here is the KitsuMate ONNX conversion, and only after a GPU test shows it beats eager PyTorch.

## Addendum: static cache and CUDA graphs (measured)

Implemented in `nano_fast/fast_nano.py`. Decode step on RTX 3060 Laptop, greedy,
same tokens as the dynamic cache: dynamic eager 7.6 ms, static eager 8.2 ms,
static with CUDA graphs 5.3 ms. End to end over 7 interleaved runs, median
2.41 s baseline vs 1.99 s fast; best-of-7 is equal (1.82 s vs 1.84 s), so the
gain is about 15-20% on median and within the laptop's run-to-run noise.
