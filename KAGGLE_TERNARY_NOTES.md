# Kaggle ternary training: what we learned (from fast-lfm-audio)

Written for the Chatterbox Nano ternary work. Everything below was verified on Kaggle or locally
unless marked **unverified**. Source of truth in the other project: `C:\dev\fast-lfm-audio\ternary\`
(`ternary_qat.py`, `train_ternary.py`, `eval_asr.py`, `interactive_ternary.py`) and `C:\dev\kaggle\`.

## 1. Kaggle CLI and account

- CLI: `C:\Users\user\AppData\Roaming\Python\Python312\scripts\kaggle` (v2.2.4). Already authenticated
  through `~/.kaggle/access_token`. Account: `heclgang`.
- Kernel metadata that works (`kernel-metadata.json`):
  - `id`: `heclgang/<slug>`, `code_file`, `language: python`
  - `kernel_type: "script"` (or `"notebook"`, see section 3)
  - `is_private: true`, `enable_gpu: true`, `enable_internet: true`
  - `machine_shape`: `NvidiaTeslaT4` (two T4s, 15 GB each), `NvidiaL4`, or `TpuV5E8` (TPU v5e-8).
    Only the T4 path runs CUDA code. The TPU needs a different stack (JAX or torch_xla).
  - `dataset_sources`: `["heclgang/<dataset-slug>"]`
- Commands: `kernels push -p <dir>`, `kernels status <slug>`, `kernels logs <slug>`,
  `kernels output <slug> -p <dir>`, `datasets create|version -p <dir> --dir-mode zip -m "<msg>"`.
  Each push is a new numbered version.
- **Dataset mount path.** Uploading with `--dir-mode zip` makes Kaggle extract the zips, so the files
  mount as plain folders at `/kaggle/input/datasets/heclgang/<slug>/`, not `/kaggle/input/<slug>/`.
  Find files with `glob.glob("/kaggle/input/**/<marker file>", recursive=True)` and copy from there.
  A hard-coded path failed on the first attempt.
- Wait for a dataset version to show `ready` (`datasets status`) before pushing a kernel that uses it.

## 2. Queueing and runtime

- A T4 kernel sat in `QUEUED` for 30 to 60+ minutes before starting. Nothing on our side speeds
  that up, so plan around it.
- Measured per-run cost: install and data setup about 2 minutes, then about 5.2 s per optimizer step
  for the ternary student (batch 2, grad checkpointing, FP32 teacher on the second GPU). 1,500 steps
  took about 2.2 hours of training, after the queue wait. The queue wait was not timed.
- `kernels output` downloads the whole working directory. A 1.2 GB checkpoint in `/kaggle/working`
  made the download terminate partway. Delete large artifacts before the script exits, and keep only
  the summary, the log, and small files in `/kaggle/working`.

## 3. Logs and progress

- A **script** kernel does not show its log while it runs. `kernels logs` returns nothing, and the
  browser logs page is empty, until the run finishes. Silence during a run is expected behaviour, not
  a hang.
- A **notebook** kernel (`kernel_type: "notebook"`, a `.ipynb`) streams cell output live. The balrog
  kernel in `C:\dev\kaggle\balrog-demo-train` is an example. If you need live progress, use a notebook,
  with one cell per stage. **Unverified** for our own training loop.
- Make a run observable from the outside anyway: write `summary.json` and a small log to
  `/kaggle/working` and print with `flush=True`.
- Pipe subprocess output through `tee` with `set -o pipefail`. Without `pipefail`, a failing trainer
  is hidden by `tee`'s exit code.

## 4. Browser checks on Kaggle

- The kernel page opens on an older version ("Version 4 of 5"). Select the latest version before
  reading logs or output. We read the wrong version once and drew a wrong conclusion.
- Private pages need the browser to be signed in as the account owner. A signed-out browser shows
  "We can't find that page" for private kernels, which looks like a missing kernel.
- Do not type API keys or passwords into Kaggle's web forms through automation. Add secrets yourself.

## 5. Secrets and tokens (important: projects can be public)

- Never put a Hugging Face token in code, a dataset, or a kernel file. The project was public.
- Add the token in the kernel's **Add-ons → Secrets** as `HF_TOKEN`, attach it to the kernel, and read
  it at runtime:
  ```python
  from kaggle_secrets import UserSecretsClient
  os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
  ```
- Without a token, Hugging Face downloads are unauthenticated and slow (the logs warn about it).
- Locally, store the token with `huggingface_hub.login(token=..., add_to_git_credential=False)`, which
  writes to `~/.cache/huggingface/token`, outside the repo.

## 6. Model loading on Kaggle

- Kaggle has no Hugging Face cache. Do not use `local_files_only=True`. Plain `snapshot_download(repo, revision=...)`
  uses the cache when it exists and downloads otherwise. Our first run failed because of that flag.
- Pin the revision and every git dependency to a commit. For the LFM project that was
  `transformers@843101f…` and `fast-mimi@f6825da…`, installed with `pip install git+…@<sha>`.
- Chatterbox Nano loads from `ResembleAI/chatterbox-nano` (see section 8). Its loader already
  tolerates a missing HF cache through `snapshot_download`.
- Hugging Face datasets: streaming `load_dataset(..., streaming=True)` worked on Kaggle. Locally the
  newer `datasets` needed `torchcodec` to decode audio. The fallback that worked everywhere was
  downloading one parquet shard with `hf_hub_download` and decoding the FLAC bytes with `soundfile`.

## 7. Ternary QAT: what to keep, what broke

Verified by the 35-test CPU suite in `ternary/test_ternary_qat.py` and by the runs above.

- **Quantizer**: absmean scale per output row (or per group of 128 columns), codes `clamp(round(W/s), -1, 1)`.
  Compute the statistics in **FP32** and cast back. In FP16, `clamp_min(1e-8)` underflows to 0, so an
  all-zero row gives NaN codes. This bug was real and is fixed.
- **STE**: `latent + alpha * (q - latent).detach()`. Gradient is the identity to the latent. Latents
  need `requires_grad` copied from the source `nn.Linear`, or frozen layers become trainable.
- **Alpha** is a `persistent=False` buffer, so FP and ternary checkpoints stay loadable with `strict=True`.
  Ramp `alpha` linearly from 0 to 1 over the first 25% of steps.
- **Latent clip**: `clip_latents` after each optimizer step (we used 1.0). **Unverified** whether it
  helps; it was on in the 1,500-step run.
- **Export**: store int8 codes plus **FP32** scales, and store `group_size`, `bias` and in/out sizes
  per layer (`export_metadata`). Our 1,500-step output was about 1.2 GB for 1.2 B weights, one byte
  per weight. Packing five base-3 digits per byte would give about 0.24 GB. Not implemented.
- **Reloading the packed model (dtype bug)**: when the model is FP16, build the replacement
  `nn.Linear` with `dtype=` the model's dtype. Our loader created FP32 layers inside an FP16 model and
  crashed with `mat1 and mat2 must have the same dtype`.
- **Teacher**: run the full-precision teacher in FP32 on the second GPU. It takes about 6 GB. Fine on a T4.
- **Student memory**: FP32 latents plus gradients plus Adafactor, with grad checkpointing, peaked at
  12.2 GB on a 15 GB T4. AdamW does not fit on a T4 for ~1.2 B latents. Use Adafactor.
- **T4 has no BF16.** Use FP16 autocast with a GradScaler.
- **Distillation is teacher-forced only.** The student never trains on its own free-running outputs
  (exposure bias). This is a known gap, not implemented.

### What the runs showed

- 30 steps: useless. Text loss went from 0.9 to about 10 when alpha reached 1.0 (step 7 to 8) and
  came back only to about 6. The student then generated endless text, 2,824% WER.
- 1,500 steps: text loss fell from about 2 to about 1.2 to 1.4 after alpha reached 1.0 at step 375.
  The run then failed at `verify_packed_reload`: max logit difference 0.078 against a tolerance of
  0.05. The check ran before the eval, so **no held-out WER was ever computed.** Three hours were lost.
- **Lesson:** a verification step must never kill the run before evaluation. Log the check result, then
  continue to the eval. Compare argmax agreement and WER rather than raw logits: the packed reload
  is FP16 and rounds logits. Use the eval for the decision.
- Evaluation: greedy generation with no stop condition runs to `max_new_tokens`. Cap it (we used 256).
  A degenerate student is what makes eval slow.

## 8. Chatterbox Nano specifics (from reading the code)

- Nano is `ChatterboxTurboTTS.from_pretrained(device, nano=True)` in
  `chatterbox/src/chatterbox/tts_turbo.py`.
- Hub repo: `ResembleAI/chatterbox-nano` (Turbo uses `ResembleAI/chatterbox-turbo`).
- Files: `t3_nano_v1.safetensors` (T3 text-to-speech-token model), `s3gen_meanflow.safetensors`
  (flow-matching vocoder, strict load), `ve.safetensors` (voice encoder), `conds.pt` (built-in voice,
  optional), plus the tokenizer.
- T3 backbone: `GPT2_small`, hidden size 768. Turbo uses `GPT2_medium` (hidden 1024, 30 layers).
  Config: `text_tokens_dict_size=50276`, `speech_tokens_dict_size=6563`, `speech_cond_prompt_len=375`,
  no perceiver resampler, no emotion adversarial head. `del t3.tfmr.wte` after load.
- `from_pretrained` has a workaround: if the first `snapshot_download` fails, it retries with
  `HF_HUB_DISABLE_XET` set. Keep that path on Kaggle.
- **Unverified:** exact `nn.Linear` names inside `t3.tfmr` and `s3gen`. Find them with
  `[n for n, m in model.named_modules() if isinstance(m, nn.Linear)]` before choosing candidates.

### Suggested candidate split (from the research in `ternary.md`, not yet checked against Nano)

- **Ternary:** the GPT-2 backbone linears in `t3.tfmr` (attention projections and MLP). This is most
  of the T3 parameters.
- **Keep full precision:** token embeddings and the speech-token output head, norms and biases,
  positional and conditioning layers, the voice encoder, and the whole S3Gen decoder and vocoder
  (the research found that quantizing the vocoder hurts quality most).

## 9. Checklist for the first Nano kernel

1. Build a local CPU smoke test with the tiny random config, then run the CPU unit tests.
2. Build a manifest of audio and transcripts, and a held-out eval manifest, with train and test
   separated. Cap eval generation length.
3. Put the code in a dataset (zips are fine), wait for `ready`, then push the kernel with
   `enable_gpu` and `enable_internet`. Attach the `HF_TOKEN` secret yourself.
4. Start with a 30-step probe to check memory, step time and the reload path. Then a pilot of a few
   hundred to a couple of thousand steps. Judge the result by held-out WER, not by loss.
5. Make the reload check non-fatal, and always reach the eval step.
6. Keep `/kaggle/working` small. Download only the summary and the log.
