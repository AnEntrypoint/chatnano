"""Diagnostic: is the flow decoder deterministic for fixed captured inputs? Hashes the inputs before and after two
eager calls, and compares the two outputs under fp16 autocast and under fp32, with the location of the largest difference.
Usage: python3 -I decoder_determinism.py OUT.json
"""
import hashlib
import json
import sys

import torch

out_path = sys.argv[1]
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
import chatterbox.models.s3gen.decoder as cfm_decoder_mod  # noqa: E402
from chatterbox.tts_turbo import ChatterboxTurboTTS, S3GEN_SIL  # noqa: E402
from fast_nano import FastNano  # noqa: E402

TEXT = ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
        "Would you like me to get some prices for you?")
captured = {}


def pre_hook(module, args, kwargs):
    captured["kwargs"] = dict(kwargs)


def digest(t):
    return hashlib.md5(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:12]


base = ChatterboxTurboTTS.from_local("/workspace/chatsrc/chatnano-fp16", device="cuda", nano=True)
fast = FastNano(base)
s3 = base.s3gen
decoder = s3.flow.decoder
result = {}
with torch.inference_mode():
    torch.manual_seed(0)
    toks = fast.speech_tokens(TEXT).flatten()
    toks = toks[toks < 6561].to("cuda")
    toks = torch.cat([toks, torch.tensor([S3GEN_SIL] * 3, device="cuda").long()])
    handle = decoder.register_forward_pre_hook(pre_hook, with_kwargs=True)
    torch.manual_seed(1)
    with torch.autocast("cuda", dtype=torch.float16, enabled=fast.fp16_decoder, cache_enabled=False):
        s3.flow_inference(speech_tokens=toks, ref_dict=base.conds.gen, n_cfm_timesteps=2, finalize=True)
    handle.remove()
    kw = captured["kwargs"]
    tensors = {k: v for k, v in kw.items() if torch.is_tensor(v)}
    result["shapes"] = {k: list(v.shape) + [str(v.dtype)] for k, v in tensors.items()}
    result["n_timesteps"] = kw["n_timesteps"]
    result["meanflow"] = kw["meanflow"]
    before = {k: digest(v) for k, v in tensors.items()}

    def dec(enabled):
        with torch.autocast("cuda", dtype=torch.float16, enabled=enabled, cache_enabled=False):
            return decoder(**kw)[0].float()

    a = dec(True)
    b = dec(True)
    after = {k: digest(v) for k, v in tensors.items()}
    result["inputs_changed_by_call"] = [k for k in tensors if before[k] != after[k]]
    d = (a - b).abs()
    result["fp16_a_vs_b_max_abs"] = float(d.max())
    result["fp16_a_vs_b_mean_abs"] = float(d.mean())
    result["fp16_a_vs_b_frac_over_0p01"] = float((d > 0.01).float().mean())
    result["fp16_a_vs_b_argmax_channel_time"] = [int(x) for x in divmod(int(d.argmax()), d.shape[-1] * d.shape[-2]) ][:1] + [int(d.argmax() % (d.shape[-1] * d.shape[-2]) // d.shape[-1]), int(d.argmax() % d.shape[-1])]
    result["fp16_a_vs_a_max_abs_again"] = float((dec(True) - a).abs().max())
    c = dec(False)
    e = dec(False)
    result["fp32_c_vs_e_max_abs"] = float((c - e).abs().max())
    result["fp32_vs_fp16_max_abs"] = float((c - a).abs().max())
    result["output_abs_max"] = float(c.abs().max())
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print("RESULT", json.dumps(result), flush=True)
