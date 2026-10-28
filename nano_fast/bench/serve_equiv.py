# Acceptance check: a reply served by serve.py equals the single-pass render with the same seed.
# Phase 1 (subprocess): in-process generate() for two seeds, saved to /tmp, then the GPU is freed.
# Phase 2: serve.py with ?seed=, compared sample by sample. Control: the other seed must differ.
import http.client, os, subprocess, sys, urllib.parse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/data/cache/venvs/ml/bin/python"
TEXT = "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. Would you like me to get some prices for you?"
SEEDS = (0, 1)
REF = "/tmp/serve_equiv_ref_{}.npy"

if len(sys.argv) > 1 and sys.argv[1] == "ref":
    import torch
    sys.path.insert(0, "/workspace/chatnano/chatterbox/src"); sys.path.insert(0, "/workspace/chatnano/nano_fast")
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from fast_nano import FastNano
    base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
    fast = FastNano(base)
    for seed in SEEDS:
        torch.manual_seed(seed)
        with torch.inference_mode():
            audio = fast.generate(TEXT).squeeze(0).float().cpu().numpy()
        np.save(REF.format(seed), audio)
    print("ref done", flush=True)
    sys.exit(0)

sys.path.insert(0, HERE)
from serve_bench import start, stop  # same start/stop as the benchmark

out = subprocess.run([PY, "-I", __file__, "ref"], capture_output=True, text=True)
if out.returncode != 0:
    raise SystemExit(f"RESULT: FAIL reference render failed:\n{out.stderr[-800:]}")

PORT = 8794
p, _ = start(PORT, "/tmp/serve_equiv.log")
try:
    def fetch(seed):
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=900)
        conn.request("GET", "/tts?" + urllib.parse.urlencode({"text": TEXT, "seed": seed}))
        resp = conn.getresponse()
        data = resp.read()
        assert resp.status == 200 and resp.getheader("X-Sample-Rate") == "24000", resp.status
        return np.frombuffer(data, dtype="<f4")
    served = {s: fetch(s) for s in SEEDS}
finally:
    stop(p)

ok = True
for s in SEEDS:
    ref = np.load(REF.format(s))
    got = served[s]
    same_len = len(ref) == len(got)
    n = min(len(ref), len(got))
    mx = float(np.max(np.abs(ref[:n] - got[:n]))) if n else float("inf")
    print(f"seed {s}: served {len(got)} samples ({len(got)/24000:.2f} s), reference {len(ref)} samples; "
          f"same length {same_len}; max |diff| {mx:.5f}")
    ok = ok and same_len and mx < 0.01
ctrl = float(np.max(np.abs(np.load(REF.format(SEEDS[0])) - served[SEEDS[1]][:len(np.load(REF.format(SEEDS[0])))])))
print(f"control: seed {SEEDS[1]} served vs seed {SEEDS[0]} reference max |diff| {ctrl:.5f} (must be clearly larger)")
ok = ok and ctrl > 0.05
print("RESULT: PASS" if ok else "RESULT: FAIL")
sys.exit(0 if ok else 1)
