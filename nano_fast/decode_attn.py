"""Single-query decode attention for GPT-2 with a static cache: a split-KV Triton kernel and a combine kernel.

The eager path runs about ten small kernels per layer for one query row, and at this size their launch and
memory round trips, not the arithmetic, set the cost. This replaces those with two kernels per layer. Each
program covers one chunk of the cache for one head and keeps an online softmax (running max, sum and output).
The combine kernel merges the chunks with exact log-sum-exp rescaling, as in flash-decoding (Dao et al.) and
NVIDIA Model-Optimizer's attention_decode. Prefill (more than one query row) keeps the eager path.

Installed by replacing modeling_gpt2.eager_attention_forward, which GPT2Attention looks up when called. The
mask and the cache layout are unchanged, so the rest of the model is untouched.

Status: an experiment, not installed by default. Measured 2026-11-02 on the RTX 3060: the token loop is 2.4%
faster (310.7 to 303.4 ms short, 479.2 to 467.8 ms long), against 47-64% for attention alone under CUDA graphs.
Speech tokens differ from eager on 3 of 15 seed and text pairs. Eager differs from a second eager run on 2 of
15 pairs, so the default is not deterministic across processes either. Rejected for the default; see the
report, section "Decode attention kernel (rejected)".
"""
import math

import torch
import triton
import triton.language as tl
from transformers.models.gpt2 import modeling_gpt2

SPLIT = 64  # cache slots per program
BLOCK_N = 64  # slots per inner step
NUM_WARPS = 4


@triton.jit
def _partial(Q, K, V, MASK, M, LS, ACC, scale,
             sqh, sqd, skh, skn, skd, svh, svn, svd, smn,
             L, NS: tl.constexpr, D: tl.constexpr, SPLIT: tl.constexpr, BLOCK_N: tl.constexpr):
    h = tl.program_id(0)
    sp = tl.program_id(1)
    offs_d = tl.arange(0, D)
    # cast: torch.compile passes a Python float as fp64, which would widen every fp32 value below
    q = tl.load(Q + h * sqh + offs_d * sqd).to(tl.float32) * tl.cast(scale, tl.float32)
    m_i = tl.full((), float("-inf"), tl.float32)
    l_i = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    start = sp * SPLIT
    end = tl.minimum(start + SPLIT, L)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        valid = offs_n < end
        k = tl.load(K + h * skh + offs_n[:, None] * skn + offs_d[None, :] * skd,
                    mask=valid[:, None], other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1)
        s = s + tl.load(MASK + offs_n * smn, mask=valid, other=0.0).to(tl.float32)
        s = tl.where(valid, s, tl.full((BLOCK_N,), float("-inf"), tl.float32))  # fp32: a Python float widens to fp64
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        v = tl.load(V + h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd,
                    mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    # a chunk past the end of the cache keeps m = -inf, l = 0, acc = 0, which the combine ignores
    tl.store(M + h * NS + sp, m_i)
    tl.store(LS + h * NS + sp, l_i)
    tl.store(ACC + (h * NS + sp) * D + offs_d, acc)


@triton.jit
def _combine(M, LS, ACC, OUT, NS: tl.constexpr, D: tl.constexpr):
    h = tl.program_id(0)
    offs_s = tl.arange(0, NS)
    m = tl.load(M + h * NS + offs_s)
    l = tl.load(LS + h * NS + offs_s)
    m_all = tl.max(m, axis=0)
    w = tl.exp(m - m_all)
    denom = tl.sum(l * w, axis=0)
    offs_d = tl.arange(0, D)
    acc = tl.load(ACC + (h * NS + offs_s[:, None]) * D + offs_d[None, :])
    out = tl.sum(acc * w[:, None], axis=0) / denom
    tl.store(OUT + h * D + offs_d, out.to(OUT.dtype.element_ty))


_eager = modeling_gpt2.eager_attention_forward


def _launch(query, key, value, mask):
    H, D = query.shape[1], query.shape[3]
    L = key.shape[2]
    NS = max(2, triton.next_power_of_2(triton.cdiv(L, SPLIT)))
    dev = query.device
    m = torch.empty(H * NS, dtype=torch.float32, device=dev)
    ls = torch.empty(H * NS, dtype=torch.float32, device=dev)
    acc = torch.empty(H * NS, D, dtype=torch.float32, device=dev)
    out = torch.empty(H, D, dtype=query.dtype, device=dev)
    _partial[(H, NS)](query, key, value, mask, m, ls, acc, 1.0 / math.sqrt(D),
                      query.stride(1), query.stride(3),
                      key.stride(1), key.stride(2), key.stride(3),
                      value.stride(1), value.stride(2), value.stride(3),
                      mask.stride(0), L, NS=NS, D=D, SPLIT=SPLIT, BLOCK_N=BLOCK_N, num_warps=NUM_WARPS)
    _combine[(H,)](m, ls, acc, out, NS=NS, D=D, num_warps=NUM_WARPS)
    return out


# A custom op, so torch.compile treats the kernels as one opaque call: they are captured by the compiled step's
# CUDA graph instead of being traced into it. The fake implementation gives the output shape to the tracer.
@torch.library.custom_op("chatnano::decode_attn", mutates_args=())
def _decode_op(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _launch(query, key, value, mask)


@_decode_op.register_fake
def _decode_op_fake(query, key, value, mask):
    return query.new_empty(query.shape[1], query.shape[3])


def decode_attention(module, query, key, value, attention_mask, **kwargs):
    """GPT2Attention's attention function. One query row with a mask goes to the kernels; anything else is eager."""
    if (query.shape[2] != 1 or query.shape[0] != 1 or attention_mask is None
            or not getattr(module, "scale_attn_weights", True)
            or getattr(module, "scale_attn_by_inverse_layer_idx", False)):
        return _eager(module, query, key, value, attention_mask, **kwargs)
    H, D = query.shape[1], query.shape[3]
    out = torch.ops.chatnano.decode_attn(query, key, value, attention_mask[0, 0, 0])
    return out.view(1, 1, H, D), None


def install():
    """Route GPT-2 decode-step attention to the kernels. Process-wide: every GPT-2 in the process uses it."""
    modeling_gpt2.eager_attention_forward = decode_attention


def uninstall():
    modeling_gpt2.eager_attention_forward = _eager
