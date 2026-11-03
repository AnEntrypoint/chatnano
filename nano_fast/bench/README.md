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
- `decode_step_logits_probe.py OUT.json`: per-step logits hashes for one seeded generation. Run warm and cold processes and compare the step lists to find where a run's numbers first change.
- `compile_mode_e2e.py max-autotune|autotune-off OUT.json`: token loop and tokens with the default compile and with autotuning off. Autotuning off is the bit-reproducible option across fresh compiles, at a measured cost.
- `serve_bench.py`: the resident server live: startup, cold and warm requests, concurrent requests on one and on two servers.
- `serve_equiv.py`: a served reply (`?seed=`) against the in-process single-pass render, with a different seed as the control.
- `pipeline_ceiling.py OUT.json [--repeats N]`: per unit, the T3 token loop and the vocoder stage (flow, HiFT, watermark) timed serially; the ceiling of overlapping unit k's vocoder with unit k+1's tokens is the sum of min(vocoder k, T3 k+1) over the reply's total.
- `pipeline_check.py ROOT OUT_DIR --tag T [--repeats N] [--seeds 0,1]`: served replies through `Server.pieces()` for three texts and two seeds, with time to first audio and total time, npy and WAV per reply. Run it on two trees (the working tree and a `git archive` copy whose `serve.py` paths are rewritten to the copy) and compare with `pipeline_check.py --compare DIR_A DIR_B` (max abs diff and LSD per reply), then `asr_check.py` on the WAVs.
- `natural_length_probe.py`: reply length on varied natural prose at four lengths (speech rate per word). Confirms the 72-word limit is inside the range that completes.
- `loudness_peaks.py OUT.json`: peak and RMS of reference `generate()` against the fast path over five seeds, the reference first (FastNano halves the T3 in place).
- `decoder_determinism.py OUT.json`: the flow's CFM decoder draws its noise for the prompt region from the generator, so two eager calls differ unless seeded; checks that the captured inputs are not changed by a call.
- `flow_graph_probe.py OUT.json`: the CFM decoder alone, eager against one CUDA graph over the same captured inputs, seeded so the replay can be compared exactly.
- `decoder_graph_check.py OUT.json`: the flow mel and the wav with the decoder graphs off and on, four texts and three seeds, one process; also the warm-up time and graph count.
- `decoder_graph_time.py OUT.json`: first graphs-on call per text (capture included), then five interleaved rounds of the flow and of generate.
- `decoder_graph_rng.py OUT.json`: the generator state after the full S3Gen call and the wav, graphs off and on.
- `decoder_graph_control.py OUT.json TEXT SEED`: generate twice eager and twice graphed, tokens and wav compared, to set the wav differences against run-to-run noise.
- `decoder_graph_hift.py OUT.json`: the mel that reaches HiFT inside generate(), graphs off and on, with the generator state at that point.
- `render_graph_wavs.py OUT_DIR`: WAVs of three texts at two seeds, graphs off and on, for `asr_check.py`.
- `hift_graph_probe.py OUT.json WAV_DIR`: HiFT alone at exact mel length, eager against one CUDA graph per text and per 64-frame length, with the wav and generator-state checks and the capture cost. Its graph body reimplements the ISTFT and the sine source (see the HiFT graphs subsection of the report).
- `../kaggle/t4_check.py`: the T4 check as a Kaggle script kernel: fresh-process token counts and the audio with whisper-tiny transcripts.
- `old_tokens.json`: token output of the code before the fast-path changes (`5f16c12` parent). `new_tokens.json`: output of the code at `5d71f59`.

The scripts hard-code `/workspace/chatnano` as the new tree. Comparisons against the pre-change code need a checkout of the older
revision (`git archive <rev> | tar -x -C <dir>`) passed as the root argument.
