"""int4 weight-only T3 on bf16, against the fp16 default (the open item in the report).

torchao's int4 path needs bf16 activations, so the default's fp16 T3 cannot take it. One mode per process, so the
models never share the GPU:
  default  the production FastNano: fp16 T3, fp32 speech head, fp16 S3Gen
  bf16     the control: the same T3 in bf16 (Conv1D as nn.Linear, speech head fp32), no quantisation
  int4     the bf16 T3 with Int4WeightOnlyConfig(group_size=128, TILE_PACKED_TO_4D). The default PLAIN packing
           needs the mslk package, which is not on the index here. S3Gen stays fp32 in both bf16 modes.
Only the T3 is timed (speech_tokens), so the comparison is the token loop. Writes tokens and timings to OUT.json.
Usage: python3 -I int4_bf16.py default|bf16|int4 OUT.json
"""
import json
import statistics
import sys
import time

import torch

mode, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, "/workspace/chatnano/chatterbox/src")
sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
from fast_nano import FastNano  # noqa: E402

WEIGHTS = "/workspace/chatsrc/chatnano-fp16"
# the same two texts as nano_fast/kaggle/suite.py
TEXTS = {
    "short": ("Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
              "Would you like me to get some prices for you?"),
    "long": ("This is a considerably longer sentence, written to push the decoder past a few dozen speech "
             "tokens, so that the cache bucket and the stop check both get exercised across several "
             "boundaries in one run."),
}


def conv1d_to_linear(model):
    # GPT-2 projections are transformers' Conv1D (weight stored as (in, out)); torchao quantises only nn.Linear
    from transformers.pytorch_utils import Conv1D
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, Conv1D):
                nin, nout = child.weight.shape
                lin = torch.nn.Linear(nin, nout, bias=child.bias is not None,
                                      device=child.weight.device, dtype=child.weight.dtype)
                lin.weight.data.copy_(child.weight.data.t())
                if child.bias is not None:
                    lin.bias.data.copy_(child.bias.data)
                setattr(mod, name, lin)
    return model


def build():
    base = ChatterboxTurboTTS.from_local(WEIGHTS, "cuda", nano=True)
    if mode == "default":
        return base, FastNano(base)
    if mode not in ("bf16", "int4"):
        raise SystemExit(f"unknown mode {mode!r}")
    # FastNano.__init__ casts T3 and S3Gen to fp16; skip that cast and build the bf16 path here
    FastNano._apply_fp16_t3 = lambda self: None
    fast = FastNano(base)
    t3 = base.t3
    conv1d_to_linear(t3.tfmr)
    t3.to(torch.bfloat16)
    t3.speech_head.float()  # the head stays fp32, as in the default path
    for k, v in list(vars(base.conds.t3).items()):
        if torch.is_tensor(v) and v.is_floating_point():
            setattr(base.conds.t3, k, v.bfloat16())
    if mode == "int4":
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
        from torchao.quantization.quant_api import Int4PackingFormat
        # the default PLAIN format needs the mslk package (unavailable here); TILE_PACKED_TO_4D is the tinygemm kernel
        quantize_(t3.tfmr, Int4WeightOnlyConfig(group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D))
    fast._head_weight_f32 = t3.speech_head.weight.detach().float().contiguous()
    fast._head_bias_f32 = (t3.speech_head.bias.detach().float().contiguous()
                           if t3.speech_head.bias is not None else None)
    # _fp16_step takes the trunk's dtype as given and casts the logits to fp32, so it serves bf16 unchanged
    fast._step = fast._fp16_step
    fast.step = torch.compile(fast._fp16_step, mode="max-autotune", fullgraph=False)
    return base, fast


def main():
    base, fast = build()
    n_quant = sum(type(m.weight.data).__name__ != "Tensor" for m in base.t3.tfmr.modules()
                  if isinstance(m, torch.nn.Linear))
    result = {"mode": mode, "quantized_layers": n_quant, "tokens": {}, "token_loop_ms": {}, "first_call_s": {}}

    def tokens(text):
        torch.manual_seed(0)
        return fast.speech_tokens(text).flatten().tolist()

    with torch.inference_mode():
        for name, text in TEXTS.items():
            t0 = time.perf_counter()
            result["tokens"][name] = tokens(text)  # the first call includes compilation
            result["first_call_s"][name] = round(time.perf_counter() - t0, 2)
            tokens(text)  # warm
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                t = time.perf_counter()
                tokens(text)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            result["token_loop_ms"][name] = round(statistics.median(ts) * 1000, 1)
    result["peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print("RESULT", json.dumps({k: v for k, v in result.items() if k != "tokens"}), flush=True)


try:
    main()
except Exception as e:  # a failure is a result too: record it rather than lose the run
    with open(out_path, "w") as f:
        json.dump({"mode": mode, "error": f"{type(e).__name__}: {str(e)[:400]}"}, f)
    print("RESULT", mode, "FAILED", type(e).__name__, str(e)[:400], flush=True)
    raise
