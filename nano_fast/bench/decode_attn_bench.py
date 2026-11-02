"""Decode attention, 12 layers under one CUDA graph: the eager path against nano_fast/decode_attn.py.

Each layer has its own K and V (the cache traffic is real), and the graph replays the 12 calls, so the timing
is GPU time with no Python in it. Correctness: max abs difference from eager on the same inputs, per cache
length. Usage: python3 -I decode_attn_bench.py
"""
import statistics
import sys

import torch

sys.path.insert(0, "/workspace/chatnano/nano_fast")
import decode_attn as DA  # noqa: E402
from transformers.models.gpt2 import modeling_gpt2  # noqa: E402

LAYERS, H, D = 12, 12, 64


class Mod(torch.nn.Module):
    scale_attn_weights = True
    scale_attn_by_inverse_layer_idx = False

    def __init__(self):
        super().__init__()
        self.attn_dropout = torch.nn.Identity()
        self.layer_idx = 0


def inputs(L, pos, dtype=torch.float16):
    g = torch.Generator(device="cuda").manual_seed(0)
    qs = [torch.randn(1, H, 1, D, device="cuda", dtype=dtype, generator=g) for _ in range(LAYERS)]
    ks = [torch.randn(1, H, L, D, device="cuda", dtype=dtype, generator=g) for _ in range(LAYERS)]
    vs = [torch.randn(1, H, L, D, device="cuda", dtype=dtype, generator=g) for _ in range(LAYERS)]
    mask = torch.zeros(1, 1, 1, L, device="cuda", dtype=dtype)
    mask[..., pos:] = torch.finfo(dtype).min  # the static cache's causal mask for one decode step
    return qs, ks, vs, mask


def graph_ms(fn, qs, ks, vs, mask, mod, n=200):
    def run():
        out = None
        for i in range(LAYERS):
            out = fn(mod, qs[i], ks[i], vs[i], mask)[0]
        return out
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / n)
    return statistics.median(ts)


def main():
    mod = Mod()
    for L in (256, 384, 768, 1024):
        pos = L // 2
        qs, ks, vs, mask = inputs(L, pos)
        with torch.inference_mode():
            ref = [modeling_gpt2.eager_attention_forward(mod, qs[i], ks[i], vs[i], mask)[0] for i in range(LAYERS)]
            got = [DA.decode_attention(mod, qs[i], ks[i], vs[i], mask)[0] for i in range(LAYERS)]
            err = max((a.float() - b.float()).abs().max().item() for a, b in zip(ref, got))
            scale = max(a.float().abs().max().item() for a in ref)
            t_eager = graph_ms(modeling_gpt2.eager_attention_forward, qs, ks, vs, mask, mod)
            t_triton = graph_ms(DA.decode_attention, qs, ks, vs, mask, mod)
        print(f"L={L} valid={pos}: max abs diff {err:.2e} (|out| max {scale:.2f}); "
              f"12-layer GPU time eager {t_eager * 1000:.0f} us, triton {t_triton * 1000:.0f} us, "
              f"saving {100 * (1 - t_triton / t_eager):.0f}%", flush=True)


if __name__ == "__main__":
    main()
