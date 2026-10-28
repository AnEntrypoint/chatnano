# Chatterbox-Nano: Official Setup and Inference Facts

Sources read (as of 2026-10-22):
- Local repo `/workspace/chatnano/chatterbox`, HEAD `5de7a54` (2026-07-21, "Update ChatterboxNano (#542)"). `pyproject.toml` version 0.1.7.
- HF model card: https://huggingface.co/ResembleAI/chatterbox-nano
- GitHub: https://github.com/resemble-ai/chatterbox (README, issues search, releases, raw `pyproject.toml` on master)

Local files read: `example_tts_nano.py`, `pyproject.toml`, `README.md` (Nano section, lines 32, 45, 87-105), `src/chatterbox/tts_turbo.py`.

## Key facts at a glance

- Nano = same class as Turbo (`ChatterboxTurboTTS`), selected with `nano=True`. Local code: `from_pretrained(device, nano=False)` (tts_turbo.py ~L192). Repo ID `ResembleAI/chatterbox-nano` (L27).
- Architecture differences in `from_local` (L136-189): `llama_config_name = "GPT2_small"` for Nano vs `"GPT2_medium"` for Turbo; T3 checkpoint `t3_nano_v1.safetensors` vs `t3_turbo_v1.safetensors`. S3Gen is `S3Gen(meanflow=True)` loaded from `s3gen_meanflow.safetensors` for both.
- English only. 110M params. MIT license (per HF page).
- Single-step decoder: the HF card says the speech-token-to-mel decoder was distilled from 10 steps to 1. In code, `generate()` calls `s3gen.inference(..., n_cfm_timesteps=2)` for both Turbo and Nano (tts_turbo.py ~L300). The code therefore does not expose a step-count knob, and the value 2 is hard-coded in `generate()`. I did not verify how `n_cfm_timesteps=2` maps to the "single-step" claim in the model card.
- Paralinguistic tags: supported via text, e.g. `[chuckle]` (example file). The HF card lists `[cough]`, `[laugh]`, `[chuckle]`, "and more" (no full list found).

## Install

Official commands:
- PyPI: `pip install chatterbox-tts` (HF card).
- Source: `git clone https://github.com/resemble-ai/chatterbox.git` then `pip install -e .` (HF card).

Pins from `pyproject.toml` (local 0.1.7, same on master per raw fetch):
- `requires-python = ">=3.10"`.
- `torch==2.6.0` for Python < 3.14 (comment: "torch 2.6.0 supports Python 3.10-3.13"); `torch>=2.9.0` for 3.14+.
- `torchaudio==2.6.0` (<3.14) / `>=2.9.0` (3.14+).
- `transformers==5.2.0`, `diffusers==0.29.0`, `numpy>=1.24,<2.0` (<3.13) / `>=2.0` (3.13+).
- `librosa==0.11.0`, `s3tokenizer` (unpinned), `safetensors==0.5.3`, `conformer==0.3.2`, `resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master`, `spacy-pkuseg`, `pykakasi==2.3.0`, `gradio==6.8.0`, `pyloudnorm`, `omegaconf`.
- No ONNX dependency. The S3 tokenizer is the `s3tokenizer` package plus the local `src/chatterbox/models/s3tokenizer`.
- The torch wheel is not pinned to a CUDA build. The user must install the matching CUDA wheel of torch 2.6.0 themselves (I did not find an official CUDA-specific command in the sources read).

Minimum versions for Nano specifically: none stated separately. The package pins above are the only requirement, so torch 2.6.0 and transformers 5.2.0 are the effective minimums (inferred from pins, not from a Nano-specific statement).

Conflict to note: the HF card says "Tested on Python 3.11 and Debian 11." A GitHub PR #559 "Bump version to 0.2.0, and document tags, voice conversion, and Gradio apps" is listed as closed on Sep 10, 2026, but the master `pyproject.toml` fetched still says 0.1.7. Either #559 was not merged to master, or the fetch is cached. I did not resolve this. Pin to whichever version you install and check `pip show chatterbox-tts`.

## Model download and caching

- `from_pretrained` calls `huggingface_hub.snapshot_download(repo_id=NANO_REPO_ID, token=HF_TOKEN or None, allow_patterns=["*.safetensors","*.json","*.txt","*.pt","*.model"])`. Standard HF cache applies (`~/.cache/huggingface` unless `HF_HOME` is set).
- Xet fallback: on Xet errors ("hex hash"), it sets `huggingface_hub.constants.HF_HUB_DISABLE_XET = True` and retries (tts_turbo.py ~L204-220).
- Files needed in the snapshot: `ve.safetensors`, `t3_nano_v1.safetensors`, `s3gen_meanflow.safetensors`, tokenizer files; optional `conds.pt` built-in voice (if present, `conds` is loaded automatically).
- The HF card warns (via search snippet) that the model is gated behind a contact-info form ("You need to agree to share your contact information to access this model"). Check access before relying on a headless install.

## Dtype, device, attention

- `from_pretrained(device, nano=False)` takes only `device` and `nano`. There is no `dtype` argument. Modules are created in default fp32 and moved with `.to(device)`. No `.half()`, `.bfloat16()`, or `torch_dtype` appears in `tts_turbo.py`.
- Device: `"cuda"` or `"cpu"` (also `"mps"`, which falls back to CPU if unavailable). Local code maps `cpu`/`mps` to CPU map_location for loading `conds.pt`.
- Attention implementation: no option is exposed by `from_pretrained` or `generate`. Not verified inside `T3` / GPT2 config (I did not read `models/t3/t3.py` in depth).
- Official fp16/bf16 statements: none found in the HF card, the GitHub README, or the releases page. I found no official "fp16/bf16 is fine" claim. Community fp16 conversions exist for MLX and Core ML (Apple), not for CUDA. Treat fp16/bf16 on CUDA as unsupported and unverified.

## Streaming

- No built-in streaming generator for Nano. The only inference entry point is `model.generate(text, ...)`, which returns one full tensor (`torch.from_numpy(watermarked_wav).unsqueeze(0)`). No generator, chunk callback, or `stream=` argument exists in `tts_turbo.py`.
- Consequently there is no official first-chunk latency for Nano. Do not assume any.
- Resemble's paid cloud API offers streaming (per search snippet). The self-hosted Nano path does not appear to.
- Community ONNX conversions (owensong, KitsuMate) state they are non-streaming. The owensong author warns that concatenating independently decoded chunks causes pauses and prosody discontinuities. Chatterbox-Flash is a different Resemble model with native streaming (time-to-first-packet from 103 ms), not Nano.

## torch.compile / CUDA graphs

- Not mentioned in the HF card, README, or local code. No `torch.compile`, `cuda.graph`, or `CUDAGraph` usage found in `tts_turbo.py`. Treat as unsupported. Not verified across the whole `src/` tree.

## Example script API

`example_tts_nano.py` (full file, 769 bytes):
```python
import torchaudio as ta
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
model = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
text = "Oh, that's hilarious! [chuckle] Um anyway, ... Would you like me to get some prices for you?"
wav = model.generate(text)                      # voice from built-in conds.pt, if present
# wav = model.generate(text, audio_prompt_path="your_10s_ref_clip.wav")
ta.save("test-nano.wav", wav, model.sr)
```

`generate()` signature (tts_turbo.py ~L272):
`generate(text, repetition_penalty=1.2, min_p=0.00, top_p=0.95, audio_prompt_path=None, exaggeration=0.0, cfg_weight=0.0, temperature=0.8, top_k=1000, norm_loudness=True)`

- `audio_prompt_path` triggers `prepare_conditionals`. It must be longer than 5 s (asserted); the code uses the first ~10 s (DEC_COND_LEN = 10 s, ENC_COND_LEN = 15 s).
- `exaggeration`, `cfg_weight`, `min_p` are ignored by Turbo/Nano with a logged warning (no CFG in this architecture).
- Output: a `torch.Tensor` of shape `[1, N]` at `model.sr` = `S3GEN_SR` (24 kHz per code constants; check `S3GEN_SR` in `models/s3gen/const.py` if needed; not read).
- Every output is watermarked with PerTh (`self.watermarker.apply_watermark`).
- Text goes through `punc_norm`, then `self.tokenizer` (max length truncation). The T3 call is `t3.inference_turbo(...)`, which uses `temperature`, `top_k`, `top_p`, `repetition_penalty`.
- Speech tokens >= 6561 are dropped (OOV), then 3 silence tokens are appended.

## Latency knobs (what exists)

- Sample rate: fixed by the model (`model.sr`). Not configurable.
- Steps: `n_cfm_timesteps=2` is hard-coded in `generate()`. Not exposed.
- Max tokens: not a `generate()` argument in the local code. The T3 inference loop limit is inside `t3.inference_turbo` (not read; a max-length parameter may exist there).
- Batch size: `generate()` processes one text. No batch API found.
- Speed reference: "3x faster than realtime on 8 CPU cores" (README and HF card). The GPU figure "10x real-time on GPU" appears only in a search-snippet summary and was not found on the official page text I fetched; treat as unverified.
- Memory: no official footprint stated. At 110M params the weights are small (fp32 roughly 0.44 GB for T3+S3Gen weights combined is an estimate, not an official figure; I did not measure it).

## RTX 3060 Laptop (6 GB, sm_86, driver 580) guidance

- No official recommendation for this GPU. Official example uses `device="cuda"`, fp32 default.
- Driver 580 supports CUDA 13.x runtime; torch 2.6.0 wheels are built for CUDA 12.4/12.6 (torch 2.6.0 cu124 is the typical build; I did not verify wheel availability for sm_86 from the sources read, but cu124 wheels include sm_86).
- Memory: 6 GB is likely sufficient for a 110M model in fp32 (inference), but this is an inference, not a measured number.

## Known issues

- No Nano-specific issues found in the GitHub issue search except:
  - #556 "LoRA language adaptation pipeline for Chatterbox Nano + training/inference contract findings" (opened 2026-09-04, status not shown).
  - PR #544 "Load models on demand and release them after 20 minutes idle" (closed 2026-08-05).
- Older platform issues: #216 "Experimental support for AMD 9070XT" (2025-08-10).
- Releases page showed only v0.1.2 (stale cache); its notes mention a CUDA fix (#114, "Fix CUDA error #72") and CPU/MPS support, not Nano.
- No official reports found on Windows, CUDA 12.x or 13.x specifically for Nano. Gap, not a clean bill of health.

## Gaps

- Could not confirm the exact `n_cfm_timesteps` / single-step mapping in code beyond the hard-coded 2.
- Could not read `models/t3/t3.py` inference loop for max-token, attention, or KV-cache details.
- No official fp16/bf16 or torch.compile statement.
- No official memory footprint or GPU RTF.
- Version conflict around #559 / 0.2.0 vs master pyproject 0.1.7.
- Gated HF access status unverified.
