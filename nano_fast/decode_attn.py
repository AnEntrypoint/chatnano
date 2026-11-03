"""Single-query decode attention for GPT-2 with a static cache, as one Triton kernel per layer. Not installed by default."""
import math

import torch
import triton
import triton.language as tl
from transformers.models.gpt2 import modeling_gpt2

SPLIT = 64
BLOCK_N = 64
NUM_WARPS = 4

MAX_HEADS = 64
_count = None


@triton.jit
def _attn(Q, K, V, MASK, M, LS, ACC, COUNT, OUT, scale,
          sqh, sqd, skh, skn, skd, svh, svn, svd, smn, soh, sod,
          L, NS: tl.constexpr, D: tl.constexpr, SPLIT: tl.constexpr, BLOCK_N: tl.constexpr):
    h = tl.program_id(0)
    sp = tl.program_id(1)
    offs_d = tl.arange(0, D)
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
        s = tl.where(valid, s, tl.full((BLOCK_N,), float("-inf"), tl.float32))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        v = tl.load(V + h * svh + offs_n[:, None] * svn + offs_d[None, :] * svd,
                    mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    tl.store(M + h * NS + sp, m_i)
    tl.store(LS + h * NS + sp, l_i)
    tl.store(ACC + (h * NS + sp) * D + offs_d, acc)
    tl.debug_barrier()
    last = tl.atomic_add(COUNT + h, 1, sem="acq_rel")
    if last == NS - 1:
        offs_s = tl.arange(0, NS)
        m = tl.load(M + h * NS + offs_s, cache_modifier=".cg")
        l = tl.load(LS + h * NS + offs_s, cache_modifier=".cg")
        w = tl.exp(m - tl.max(m, axis=0))
        denom = tl.sum(l * w, axis=0)
        parts = tl.load(ACC + (h * NS + offs_s[:, None]) * D + offs_d[None, :], cache_modifier=".cg")
        out = tl.sum(parts * w[:, None], axis=0) / denom
        tl.store(OUT + h * soh + offs_d * sod, out.to(OUT.dtype.element_ty))
        tl.store(COUNT + h, 0)


_eager = modeling_gpt2.eager_attention_forward


def _counter():
    global _count
    if _count is None:
        _count = torch.zeros(MAX_HEADS, dtype=torch.int32, device="cuda")
    return _count


def _launch(query, key, value, mask):
    H, D = query.shape[1], query.shape[3]
    L = key.shape[2]
    NS = max(2, triton.next_power_of_2(triton.cdiv(L, SPLIT)))
    dev = query.device
    m = torch.empty(H * NS, dtype=torch.float32, device=dev)
    ls = torch.empty(H * NS, dtype=torch.float32, device=dev)
    acc = torch.empty(H * NS, D, dtype=torch.float32, device=dev)
    out = torch.empty(H, D, dtype=query.dtype, device=dev)
    _attn[(H, NS)](query, key, value, mask, m, ls, acc, _counter(), out, 1.0 / math.sqrt(D),
                   query.stride(1), query.stride(3),
                   key.stride(1), key.stride(2), key.stride(3),
                   value.stride(1), value.stride(2), value.stride(3),
                   mask.stride(0), out.stride(0), out.stride(1),
                   L, NS=NS, D=D, SPLIT=SPLIT, BLOCK_N=BLOCK_N, num_warps=NUM_WARPS)
    return out


@torch.library.custom_op("chatnano::decode_attn", mutates_args=())
def _decode_op(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _launch(query, key, value, mask)


@_decode_op.register_fake
def _decode_op_fake(query, key, value, mask):
    return query.new_empty(query.shape[1], query.shape[3])


def decode_attention(module, query, key, value, attention_mask, **kwargs):
    """GPT2Attention's attention function. One query row with a mask goes to the kernel; anything else is eager."""
    if (query.shape[2] != 1 or query.shape[0] != 1 or attention_mask is None
            or not getattr(module, "scale_attn_weights", True)
            or getattr(module, "scale_attn_by_inverse_layer_idx", False)):
        return _eager(module, query, key, value, attention_mask, **kwargs)
    H, D = query.shape[1], query.shape[3]
    out = torch.ops.chatnano.decode_attn(query, key, value, attention_mask[0, 0, 0])
    return out.view(1, 1, H, D), None


def install():
    """Route GPT-2 decode-step attention to the kernel. Process-wide: every GPT-2 in the process uses it."""
    _counter()
    modeling_gpt2.eager_attention_forward = decode_attention


def uninstall():
    modeling_gpt2.eager_attention_forward = _eager
