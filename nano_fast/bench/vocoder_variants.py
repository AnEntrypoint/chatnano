"""Found the fp16-flow fault. One variant per process: the reference tokens (fp32 T3, seed 0) through S3Gen with
the flow and HiFT each in their own precision. Usage:
  python3 -I vocoder_variants.py NAME flow_half(0|1) hift_half(0|1) [autocast(0|1)]
autocast=1 runs the flow under fp16 autocast with fp32 weights (the restored default). NOWRAP=1 runs HiFT under the
same autocast as the flow with no per-module cast, which is the restored default exactly. Writes
samples/var_NAME.wav and prints VOCODER_MS (median of 7). Compare transcripts with asr_check.py."""
import sys
import numpy as np
import torch
import wave

name, flow_half, hift_half = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1"
auto = len(sys.argv) > 4 and sys.argv[4] == "1"  # autocast over the flow with fp32 weights (the earlier design)
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
import fast_nano  # noqa: E402

TEXT = ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
        "Would you like me to get some prices for you?")
SIL = fast_nano.S3GEN_SIL
m = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)

# 1. speech tokens from the unmodified fp32 reference generate()
rec = {}
orig = m.s3gen.inference
def spy(*a, **k):
    rec["toks"] = k.get("speech_tokens", a[0] if a else None).flatten()
    return orig(*a, **k)
m.s3gen.inference = spy
with torch.inference_mode():
    torch.manual_seed(0)
    m.generate(TEXT)
m.s3gen.inference = orig
toks = rec["toks"][rec["toks"] < 6561]
toks = torch.cat([toks, torch.tensor([SIL, SIL, SIL], device=toks.device)])

# 2. the variant: flow and HiFT precisions, with conds in the flow's precision
g = m.s3gen
gen_orig = dict(m.conds.gen)
m.conds.gen = {k: (v.half() if flow_half and torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in gen_orig.items()}
if flow_half:
    g.flow.half()
if hift_half:
    g.mel2wav.half()
hift_orig = g.hift_inference

def hift_wrapped(speech_feat, cache_source=None):
    dt = torch.float16 if hift_half else torch.float32
    with torch.autocast("cuda", dtype=torch.float16, enabled=hift_half):
        return hift_orig(speech_feat.to(dt), cache_source) if cache_source is not None else \
            g.mel2wav.inference(speech_feat=speech_feat.to(dt), cache_source=torch.zeros(1, 1, 0, device="cuda", dtype=dt))

import os
if os.environ.get('NOWRAP') != '1':
    g.hift_inference = hift_wrapped  # NOWRAP=1: HiFT runs under the same autocast as the flow
def run_vocoder():
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=flow_half or auto):
        torch.manual_seed(0)
        return g.inference(speech_tokens=toks, ref_dict=m.conds.gen, n_cfm_timesteps=2)

import time, statistics
wav, _ = run_vocoder()  # warm-up
torch.cuda.synchronize()
ts = []
for _ in range(7):
    t0 = time.perf_counter(); run_vocoder(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
print("VOCODER_MS", name, round(statistics.median(ts) * 1000, 1), flush=True)
audio = wav.squeeze(0).float().cpu().numpy()
pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
path = f"/tmp/chatnano-local-check/samples/var_{name}.wav"
with wave.open(path, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm.tobytes())
print("VARIANT", name, "flow_half", flow_half, "hift_half", hift_half, "samples", len(audio),
      "peak", round(float(np.abs(audio).max()), 3), "nan", bool(np.isnan(audio).any()), flush=True)
