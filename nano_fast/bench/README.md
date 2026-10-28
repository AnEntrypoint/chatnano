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
- `stream_bench.py`: mel-domain windowed streaming against the single-pass render (token identity, first-audio time, spectral distance). Streaming was removed from the server by decision; the benchmark stays as the record.
- `length_probe.py`: the length at which Nano ends a reply early (sentence repeated k times). Sets the 72-word limit in `serve.py`.
- `serve_bench.py`: the resident server live: startup, cold and warm requests, concurrent requests on one and on two servers.
- `serve_equiv.py`: a served reply (`?seed=`) against the in-process single-pass render, with a different seed as the control.
- `streaming_diag.py [tail] [context]`: where windowed streaming diverges from the single-pass render (committed mel against single-pass mel, then vocoded). Witness for the refutation recorded in the report.
- `natural_length_probe.py`: reply length on varied natural prose at four lengths (speech rate per word). Confirms the 72-word limit is inside the range that completes.
- `old_tokens.json`: token output of the code before the fast-path changes (`5f16c12` parent). `new_tokens.json`: output of the code at `5d71f59`.

The scripts hard-code `/workspace/chatnano` as the new tree. Comparisons against the pre-change code need a checkout of the older
revision (`git archive <rev> | tar -x -C <dir>`) passed as the root argument.
