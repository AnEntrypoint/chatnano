# Chatterbox-Nano speed benchmarks

Scripts used for the measurements in `reports/Chatterbox Nano maximum speed RTX 3060.md`.
Run with the `ml` env (torch 2.14.1+cu130, triton 3.8.0), e.g. `python -I tokens_bench.py <repo-root> out.json`.

- `tokens_bench.py <root> <out.json>`: seeded token streams for three texts (base and fast paths). Compare two runs with the same texts.
- `time_bench.py <root>`: median timing per text, base versus fast path.
- `newlen_bench.py <root>`: first-call time on unseen text lengths (cache bucketing).
- `step_bench.py <root>`: single decode-step timing (dynamic, static, compiled, cache padding).
- `fp16_wm_bench.py`, `s3gen_fp16_ac.py`, `seed_sim.py`: watermark CPU versus GPU, fp16 T3 and S3Gen against fp32, speaker similarity and log-spectral distance.
- `ref_test.py`, `e2e_check.py`: reference-conditionals cache and end-to-end generate with the GPU watermark.
- `ab_wavs.py`: writes fp32 and fp16-S3Gen WAV pairs for a listening check (`ab/`; WAVs are gitignored).
- `stream_bench.py` and `streaming_diag.py` (windowed streaming against the single-pass render) were removed with the streaming code. Their results are in the report, and the code is in git history before the removal.
- `length_probe.py`: the length at which Nano ends a reply early (sentence repeated k times). Sets the 72-word chunk limit in `serve.py`.
- `int4_bf16.py <default|bf16|int4> <out.json>`: the T3 token loop in fp16 (default), bf16 without quantisation (control) and bf16 with int4 weight-only. One mode per process; compare the tokens of the three JSON files.
- `listen_samples.py`: renders a two-paragraph text and a long paragraph through `serve.Server`'s own paragraph, chunk, seed and pause logic, and writes WAVs to `/tmp/chatnano-local-check/samples/` with the chunk-join times. Re-render with a fixed seed to compare.
- `asr_check.py <a.wav> [b.wav ...]`: Whisper-tiny transcripts. The intelligibility check for any audio change; run it on `listen_samples.py`'s output before shipping.
- `vocoder_variants.py NAME flow_half hift_half [autocast]`: the S3Gen flow and HiFT at each precision on the reference tokens. Found the fp16 flow fault (report: Scrambled speech).
- `decoder_same_noise.py`: the default against the fp32 decoder in one process with identical noise, plus the S3Gen memory cost.
- `decode_attn_bench.py`: 12-layer decode attention under one CUDA graph, eager against the experimental kernel in `nano_fast/decode_attn.py`, with the max abs difference per cache length.
- `decode_attn_e2e.py eager|triton OUT.json`: the compiled token loop with one attention mode per process.
- `decode_attn_tokens.py eager|triton OUT.json`: speech tokens for five seeds and three texts, per mode. Compare the two files, and run eager twice to measure the baseline's own variation.
- `serve_bench.py`: the resident server live: startup, cold and warm requests, concurrent requests on one and on two servers.
- `serve_equiv.py`: a served reply (`?seed=`) against the in-process single-pass render, with a different seed as the control.
- `natural_length_probe.py`: reply length on varied natural prose at four lengths (speech rate per word). Confirms the 72-word limit is inside the range that completes.
- `old_tokens.json`: token output of the code before the fast-path changes (`5f16c12` parent). `new_tokens.json`: output of the code at `5d71f59`.

The scripts hard-code `/workspace/chatnano` as the new tree. Comparisons against the pre-change code need a checkout of the older
revision (`git archive <rev> | tar -x -C <dir>`) passed as the root argument.
