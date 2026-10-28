# GPU-side inference options for Chatterbox-Nano on RTX 3060 Laptop (6 GB, sm_86)

Scope: read-only research. Nothing installed; /workspace/chatnano was not modified.
Companion to: `research_notes/Chatterbox Nano low latency setup/gpu_inference_options.md` (prior note, not repeated here).
Status: no local benchmark was run. All rankings are inference from the code read below plus the sources cited. Confirm with a benchmark before adopting any change.
Search coverage was thin: 3 standard searches and 3 page fetches, and several claims below remain unverified (marked UNVERIFIED).

## What is new or corrected versus the prior note

1. **Corrected: the LM backbone is confirmed.** The prior note listed it as unconfirmed. The code shows Nano uses `GPT2_small` (`n_layer: 12`, `n_embd: 768`, `n_head: 12`, `vocab_size: 50276`) via `llama_configs.py`, with the speech head over a 6563-token speech vocabulary (`tts_turbo.py:from_local`, `llama_configs.py:83-86`).
2. **Corrected: the generation loop does not go through `transformers.generate`.** Nano calls `ChatterboxTurboTTS.generate` -> `T3.inference_turbo` (`models/t3/t3.py`, around lines 392-450), which calls `self.tfmr(...)` directly with a hand-managed `past_key_values`. The prior note's "static cache via transformers generate" route therefore does not apply as-is. Using it means either rewriting this loop around a static cache or wrapping `tfmr.forward`.
3. **New finding: the Nano decode loop forces a GPU-to-CPU sync on every token**, which the prior note did not identify:
   - `if torch.all(processed_logits == -float("inf")):` (t3.py, in the loop) forces a host read each step.
   - `if torch.all(next_speech_token == self.hp.stop_speech_token): break` forces another.
   - `tqdm(range(max_gen_len))` wraps the loop (CPU-side overhead, minor).
   These syncs stop the CPU from queuing ahead of the GPU, so launch overhead and GPU idle time add up on every step. Removing them (or checking stop only every N steps) is a cheap win that is independent of CUDA graphs.
4. **New finding: the loop shape changes every step.** `input_ids = torch.cat(generated_speech_tokens, dim=1)` grows by one column per step and is passed to the logits processors (repetition penalty). The growing tensor is a dynamic shape, which is the main obstacle to `torch.compile` / CUDA graph capture of the step. Fix: keep a preallocated buffer and mask, so the processor sees a fixed shape.
5. **New finding: no autocast and no dtype is set in the Nano path.** `inference_turbo` runs under `torch.inference_mode()` only. The model loads in default fp32 (`from_local` does not call `.half()` or `.to(dtype)`). The prior note said "no dtype knob exposed", which is true for the API, but the code path is fp32 by default. Weights are about 0.44 GB fp32 for the 110M model, which fits 6 GB easily. Memory is not the constraint; launch overhead is the likely constraint.
6. **Clarified: the sampling step uses full-vocab top-k 1000 and top-p 0.95** (`TopKLogitsWarper`, `TopPLogitsWarper`, `RepetitionPenaltyLogitsProcessor`, `TemperatureLogitsWarper` in `inference_turbo`). Each step does a sort over 6563 logits and a multinomial draw. These are small kernels that add launch count per step. They are a candidate for fusion in a custom step function (or for compiling the whole step, including the sampler).

## Key question answers

### Q1. Which techniques reduce per-token latency at batch 1, and by how much?

Ranked by expected impact for this loop. Magnitudes are expectations, not measurements.

| Rank | Technique | Why it applies here | Expected per-token effect | Confidence |
|---|---|---|---|---|
| 1 | Remove the per-step host syncs and tqdm (items 3 above) | Each step blocks the CPU on the GPU twice | Removes idle gaps; likely a large fraction of step time if the GPU is short of work | High that it helps; size UNVERIFIED |
| 2 | CUDA graph capture of the whole decode step (model forward + sampler) with a static KV cache buffer and fixed-shape token buffer | 12 layers of GPT2 at batch 1 are many small kernels, so launch-bound | Biggest win if launch-bound; the Seamless M4T-v2 analogue (prior note) was CPU-bound at batch 1 | Medium; UNVERIFIED for Nano |
| 3 | torch.compile mode="reduce-overhead" on the step function, with fullgraph=True | Same mechanism as #2, but compiler-managed; needs no graph breaks | Similar to #2; adds compile time | Medium; compile-time cost UNVERIFIED |
| 4 | fp16 or bf16 weights and activations | 0.44 GB fp32 is small; at batch 1 the matmuls are tiny | Small (a few percent) unless the step is compute-bound, which it probably is not | Low-medium |
| 5 | SDPA backend (flash or mem-efficient) | Attention at batch 1 with a short cache is cheap; GPT2 head dim is 64 | Negligible at current lengths | Low; see Q3 |
| 6 | int8 / int4 weight-only quantization | Weights are not the bottleneck at this size | Likely negative or neutral; prior note's eager-mode slowdown report stands | Low |

Caveat: items 1 and 2/3 interact. Item 1 is a prerequisite for item 2/3, because a step that syncs to the host cannot be captured as a graph (the sync is a graph break or a capture error).

Note on quantization: the prior note said int8 weight-only is low priority. This still holds. Nothing found here changes that.

### Q2. KV cache and static cache

- The Nano path uses HF `GPT2Model` with `DynamicCache` tuples that grow every step. Dynamic growth changes shapes and prevents graph capture.
- A static (preallocated, fixed-length) KV cache with an attention mask over the valid positions is the standard fix. Transformers documents it (Transformers 4.4x-era `llm_optims` per the prior note; the 5.x `perf_torch_compile` page fetched here shows only the generic compile API, not the static cache details). The HF example is `torch.compile(model, mode="reduce-overhead", fullgraph=True)`. Transformers' own benchmark table is for vision models, not for this loop, so it does not transfer.
- For Nano, a static cache has to be hand-built because `inference_turbo` manages `past_key_values` itself. Max length is bounded by `max_gen_len=1000` (about 40 s of speech per the tokens-per-second assumption; UNVERIFIED), so one fixed buffer is cheap: 12 layers x 2 (K,V) x 12 heads x 64 dim x 2 bytes x 1000 positions is about 37 MB in fp16 (about 74 MB in fp32). This is a computed value, not a measurement.
- Pitfall: prefix length (text tokens plus conditioning prompt, `speech_cond_prompt_len=375`) varies per request. Bucket the prefix length (for example, to multiples of 64) or re-capture per bucket. Record the number of distinct shapes in a sweep.

### Q3. fp16, bf16, int8/int4 on sm_86; SDPA availability

- **bf16 on this GPU:** Ampere (sm_86) has bf16 tensor-core support. This is general NVIDIA Ampere knowledge, not sourced in this session (UNVERIFIED here, widely documented). Practical advice: prefer fp16 for speed on these kernels unless accuracy drifts; test both, since the audio quality is the real gate.
- **fp16 and bf16 for Nano:** neither is wired in the code. Adding it means `model.half()` or autocast on the T3 forward, with the S3Gen decoder left separate. The Chatterbox decoder code accepts `float32`, `bfloat16`, and `float16` (`s3gen/decoder.py:28`), but the T3 path does not expose a knob.
- **SDPA / flash / mem-efficient on sm_86:** PyTorch's SDPA dispatches to flash or memory-efficient kernels on CUDA when dtype, head dim, and mask allow. The PyTorch docs fetched here (SDPBackend page, 2.14) do not state compute-capability or head-dim limits, so this could not be confirmed from a source in this session. Community knowledge (UNVERIFIED): flash on sm_86 is available for fp16/bf16 with head dim up to 256, but consumer Ampere has reduced limits for some kernels. The fastest check is a 20-line test that forces each backend with `torch.nn.attention.sdpa_kernel` and reports the error on failure. For head dim 64 (GPT2-small), at least one fused backend should be available.
- **Quantization:** For 110M fp32 weights, int8/int4 saves little and adds dequant work at batch 1. Not recommended before the sync and graph work.

### Q4. TensorRT, TensorRT-LLM, Triton

- **TensorRT / torch-TensorRT:** Engine builds are per GPU, TensorRT, and CUDA version. Gains on the AR loop are unclear, since the loop is launch-bound and TRT does not remove Python-level syncs. It is a candidate only for the S3Gen decoder/vocoder if it is measured to be large, which was not done here. Not recommended as a first step.
- **TensorRT-LLM:** Built for large LLM serving. Not viable as a step-up for a 110M GPT2-small loop. No source was checked for its current sm_86 status. Not recommended.
- **Triton kernels:** Only if a specific fused sampler (top-k/top-p/repetition penalty) shows up as a hotspot in a profile. Otherwise compile covers it. Low priority.

### Q5. torch.compile and CUDA graph pitfalls

Sourced:
- CUDA graphs require static memory addresses and fixed shapes. If either changes between steps, the graph is invalid and must be recaptured. ([vLLM torch.compile write-up](https://vllm.ai/blog/torch-compile)); also [Spheron guide](https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/)
- `fullgraph=True` raises an error on any graph break, which helps find breaks early. ([Transformers torch.compile docs](https://huggingface.co/docs/transformers/v5.8.0/perf_torch_compile))
- `reduce-overhead` trades a little memory for less Python overhead. `max-autotune` takes longer to compile for the fastest kernels. ([same page](https://huggingface.co/docs/transformers/v5.8.0/perf_torch_compile))

From the code (not sourced externally, verified by reading):
- Host syncs (`torch.all(...)` in an `if`) are graph breaks.
- `tqdm` and `print` are graph breaks.
- A growing `torch.cat` of token ids changes shape each step (recompiles or capture failures).
- Repetition penalty over a growing `input_ids` tensor has the same problem as above.

Unverified (typical in practice, should be checked on this install): cold compile time for a 12-layer model is tens of seconds; one recompile per prefix bucket; memory pool fragmentation across captures with many shapes. Measure these; do not trust the Spheron 30-90 s figure.

### Q6. Thermal and power limits on a laptop 3060

- Notebookcheck lists the RTX 3060 Laptop at 60 W or 115 W TGP depending on the laptop, with boost clocks that differ by wattage (about 817-1282 MHz at 60 W, about 1387-1702 MHz at 115 W). These are ranges of possible clocks, not measured sustained values. [Notebookcheck comparison](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4060-Laptop-GPU-vs-GeForce-RTX-3060-Laptop-GPU_11455_10478.247598.0.html)
- Digital Trends reports that several laptop makers do not publish power and clock details, so the actual TGP must be measured. [Digital Trends](https://digitaltrends.com/?p=3078478)
- Gap: no source found on sustained clocks under a small-batch, low-utilization load. A batch-1 decode loop is light on the GPU, so it may not reach the clocks a gaming load does, but this is inference only.
- Practical measurement: run the benchmark for at least 5 minutes, log `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv -l 1`, and compare first-minute and last-minute RTF. Check the power plug (AC vs battery) and the performance profile before each run.

### Q7. CPU vs GPU, and splitting work

- Resemble's claim: "10x real-time on GPU" and "3x real-time on CPU (8 threads)" with no hardware or precision detail (prior note's source). Unverified.
- For a 12-layer, 768-dim AR loop at batch 1, the per-token work is about 85M parameters of matmul. On a 12-core CPU this is tens of milliseconds per token in fp32 (estimate, UNVERIFIED). On the GPU with graphs, it could be a few milliseconds per token if launch-bound. The GPU should win for the LM loop once the syncs are removed. Measure the CPU baseline too; the answer depends on the CPU's AVX-512 or AMX support, which was not checked.
- Splitting: keep the LM loop on GPU. Running the S3Gen decoder or vocoder on CPU is likely slower than the GPU, and would add a device transfer. Only consider it if the GPU is busy with the LM and the decoder is idle, which is not the case in a sequential pipeline. Measure per-stage time first.

## Ranked recommendation (try in this order)

1. **Baseline and profile.** Time the current fp32 path with synchronized timers (prior note's harness), cold and warm, with a fixed sentence set. Run `torch.profiler` on one `inference_turbo` step. Record GPU kernel occupancy vs step wall time. This decides whether item 2 is worth it.
2. **Remove per-step host syncs and tqdm.** Replace the `torch.all` checks with a stop check every N steps (or on a GPU tensor flag read once per chunk). Verify token-for-token output with a fixed seed (with the same sampler, the output should match exactly). Expected: moderate win, low risk.
3. **Fixed-shape buffers.** Preallocate the token buffer and the static KV cache. Use an attention mask over valid positions and a bucketed prefix length.
4. **CUDA graph capture of the step (manual `torch.cuda.graph`), then torch.compile mode="reduce-overhead" as the alternative.** Manual capture avoids compile time but needs fixed shapes everywhere. Count graphs in a shape sweep to confirm the count stays flat.
5. **fp16 for the T3 forward** (not the decoder yet), checked against ASR round-trip WER or a listening test. Ampere supports it; expected small gain.
6. **SDPA backend check.** Run the per-backend test on sm_86 before assuming flash or mem-efficient is used.
7. **Stop here unless the profile shows the S3Gen decoder as a significant share.** Only then look at fp16 or ONNX Runtime or TRT for that stage.

Skip: TensorRT-LLM, int8/int4 weight-only, Triton custom kernels, and CPU splitting, unless a measurement in steps 1-5 points at them.

## Install and compatibility risks (versions)

- torch.compile, CUDA graphs, SDPA, and fp16/bf16 are in the installed PyTorch. Check the local torch and CUDA build with `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"`, which should report `(8, 6)`. Driver 580 supports CUDA 13.0 (prior note).
- Triton ships with the PyTorch CUDA wheel on Linux. No extra install for compile.
- No version was pinned in the sources read here. The Transformers and PyTorch doc versions quoted are the ones fetched; behavior may differ on the installed versions.
- Mixing compile with `inference_mode` and changing module state (for example, `model.half()` after compile) causes recompiles. Do dtype changes before compile.

## Gaps

- No Nano-specific or GPT2-small-specific CUDA graph or torch.compile benchmark found.
- No source found for SDPA sm_86 head-dim or dtype limits (PyTorch SDPBackend docs fetched lacked the constraint text).
- No source found for sustained clocks on a 3060 Laptop under a small, batch-1 load.
- bf16 support on sm_86 is stated from general Ampere knowledge, not from a fetched source.
- No measurement of the watermark (PerTh) step or the S3Gen stage was made; they were not in scope of the decode loop, but they may be a meaningful share of end-to-end time.
- The Transformers static-cache numbers are for vision and 7B-class LLMs, so they do not transfer.

## Sources used

- Code: `/workspace/chatnano/chatterbox/src/chatterbox/tts_turbo.py` (`from_local`, `generate`), `/workspace/chatnano/chatterbox/src/chatterbox/models/t3/t3.py` (`inference_turbo`, around lines 392-450), `/workspace/chatnano/chatterbox/src/chatterbox/models/t3/llama_configs.py`, `/workspace/chatnano/chatterbox/src/chatterbox/models/s3gen/decoder.py:28`.
- [Transformers torch.compile docs (v5.8.0)](https://huggingface.co/docs/transformers/v5.8.0/perf_torch_compile)
- [vLLM torch.compile write-up](https://vllm.ai/blog/torch-compile)
- [Spheron torch.compile + CUDA graphs guide (PyTorch 2.6)](https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/)
- [PyTorch SDPBackend docs (2.14)](https://docs.pytorch.org/docs/2.14/generated/torch.nn.attention.SDPBackend.html) (no constraint text found)
- [Notebookcheck RTX 3060 Laptop comparison](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4060-Laptop-GPU-vs-GeForce-RTX-3060-Laptop-GPU_11455_10478.247598.0.html)
- [Digital Trends laptop GPU power disclosure](https://digitaltrends.com/?p=3078478)
