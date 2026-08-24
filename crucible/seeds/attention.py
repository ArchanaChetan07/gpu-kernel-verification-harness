"""Seed: a flash-attention-style blocked forward pass, written in pure torch.

The baseline is deliberately *not* a wrapper around
``F.scaled_dot_product_attention``. It carries the structure a real Triton
attention kernel carries -- a KV block loop over a padded KV cache, a boundary
mask, an fp32 accumulator, an online (running max / running sum) softmax
rescale, a per-lane staging buffer and one explicit store -- so that an AST
mutation lands on a site that exists in the production kernel too.

The cache is windowed rather than prefix-valid: only ``[k_start, k_start +
n_ctx)`` carries data, as in a sliding-window or paged KV cache. That is what
makes the empty-block guard load-bearing instead of merely thrifty -- a block
ahead of the window has an all -inf score tile, and taking a running max over
it leaves ``exp(m_prev - m_run)`` as ``-inf - -inf``.

The independent ground truth is ``F.scaled_dot_product_attention`` over the
window: a different code path, a different softmax formulation (one global max,
no rescale) and a different memory plan, so a bug shared by baseline and
reference cannot cancel out.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn.functional as F

from ..schema import ShapeSpec
from .registry import SeedSpec, register

_SOURCE_HEADER = '''"""Blocked attention forward: known-good baseline."""

from __future__ import annotations

import torch


'''


def blocked_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_ctx: int,
    k_start: int = 0,
    block_n: int = 64,
) -> torch.Tensor:
    """Attention forward over a window of a padded KV cache.

    ``q`` is ``(B, H, M, D)``; ``k`` and ``v`` are ``(B, H, N_pad, D)``. Only
    cache entries ``[k_start, k_start + n_ctx)`` carry data -- the rest is
    eviction slack ahead of the window and allocation slack behind it, and
    neither may influence the result.
    """
    head_dim = int(q.shape[-1])
    n_pad = int(k.shape[-2])
    scale = float(head_dim) ** -0.5
    lo = int(k_start)
    hi = lo + int(n_ctx)

    # Allocated once and written on every return path, so a caller can never
    # observe whatever happened to be in this buffer beforehand.
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if int(n_ctx) <= 0:
        out.zero_()
        return out

    lead = tuple(q.shape[:-1])  # (B, H, M)
    # The accumulator dtype is pinned to fp32 independently of the input dtype.
    # It is updated in place below, so this dtype is the dtype of every partial
    # sum over the whole context -- not just of the initial zeros.
    acc = torch.zeros(lead + (head_dim,), dtype=torch.float32, device=q.device)
    m_run = torch.full(lead, float("-inf"), dtype=torch.float32, device=q.device)
    l_run = torch.zeros(lead, dtype=torch.float32, device=q.device)

    q_f32 = q.to(torch.float32)
    n_blocks = (n_pad + int(block_n) - 1) // int(block_n)

    for blk in range(n_blocks):
        start = blk * int(block_n)
        offs = start + torch.arange(int(block_n), device=q.device)
        # Boundary mask: the cache is allocated to n_pad but the window is
        # [lo, hi), and neither edge is required to land on a block boundary.
        mask = (offs >= lo) & (offs < hi)
        n_valid = int(mask.sum())
        if n_valid == 0:
            # A block entirely outside the window contributes nothing. Running
            # it anyway takes the running max over an all -inf tile, and once
            # m_run is -inf the rescale factor exp(m_prev - m_run) is 0/0.
            continue

        gather = offs.clamp_max(n_pad - 1)  # keep the gather in range; masked lanes are discarded
        k_blk = k.index_select(-2, gather).to(torch.float32)
        v_blk = v.index_select(-2, gather).to(torch.float32)
        # The transposed K tile is materialised in row-major order: the view
        # produced by transpose() has the reduction dim on the outer stride,
        # which is the wrong access order for the tile product below.
        k_blk_t = k_blk.transpose(-1, -2).contiguous()

        s = (q_f32 @ k_blk_t) * scale
        s = torch.where(mask, s, torch.full_like(s, float("-inf")))
        blk_max = s.amax(dim=-1)

        # Snapshot before the in-place update: both rescale factors below must
        # read the *previous* running max, and m_run is written through here.
        m_prev = m_run.clone()
        torch.maximum(m_run, blk_max, out=m_run)
        alpha = torch.exp(m_prev - m_run)

        # Staging buffer for the block's probabilities. It is sized by the
        # block's lane count, not by n_valid, because it is addressed by lane id.
        lane = offs - start
        prob = torch.zeros(tuple(s.shape[:-1]) + (int(block_n),), dtype=torch.float32, device=q.device)
        prob[..., lane] = torch.where(mask, torch.exp(s - m_run.unsqueeze(-1)), torch.zeros_like(s))

        # Cross-block combine: the previous partial is *rescaled* by alpha before
        # the new block is folded in. This is not a sum of per-block partials --
        # a per-block partial is normalised by its own block max and is not a
        # term of the final result. Both updates are in place so the running
        # state keeps the accumulator dtype declared above.
        l_run.mul_(alpha)
        l_run.add_(prob.sum(dim=-1))
        acc.mul_(alpha.unsqueeze(-1))
        acc.add_(prob @ v_blk)

    denom = torch.where(l_run > 0, l_run, torch.ones_like(l_run))
    out.copy_((acc / denom.unsqueeze(-1)).to(out.dtype))  # the one store on this path
    return out


SOURCE = _SOURCE_HEADER + inspect.getsource(blocked_attention_fwd)


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_ctx: int,
    k_start: int = 0,
    block_n: int = 64,
) -> torch.Tensor:
    """Independent ground truth: torch's fused SDPA over the window.

    ``block_n`` is accepted and ignored -- the reference has no blocking, which
    is exactly why it is a usable oracle for a blocked implementation.
    """
    if int(n_ctx) <= 0:
        # Convention, not computation: an empty window contributes nothing, so
        # the result is the additive identity rather than 0/0.
        return torch.zeros(q.shape, dtype=q.dtype, device=q.device)
    lo = int(k_start)
    hi = lo + int(n_ctx)
    return F.scaled_dot_product_attention(
        q, k[..., lo:hi, :].contiguous(), v[..., lo:hi, :].contiguous()
    )


def _generator(shape: ShapeSpec, device: Any) -> torch.Generator:
    """A generator seeded from the shape itself: same shape, same tensors."""
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(int(shape.key(), 16) % (2**31 - 1))
    return g


def make_inputs(shape: ShapeSpec, device: Any = "cpu", generator: Any = None) -> dict[str, Any]:
    kw = shape.kwargs
    gen = generator if generator is not None else _generator(shape, device)
    dtype = getattr(torch, str(kw.get("dtype", "float32")))
    b = int(kw["batch"])
    h = int(kw["heads"])
    m = int(kw["q_len"])
    d = int(kw["head_dim"])
    n_pad = int(kw["n_pad"])

    def rand(*dims: int) -> torch.Tensor:
        return torch.randn(dims, generator=gen, device=device, dtype=torch.float32)

    if bool(kw.get("noncontig", False)):
        # Allocate with the head dim outermost and transpose, so q arrives with a
        # stride pattern no kernel may assume away.
        q = rand(b, h, d, m).transpose(-1, -2).to(dtype)
        k = rand(b, h, d, n_pad).transpose(-1, -2).to(dtype)
        v = rand(b, h, d, n_pad).transpose(-1, -2).to(dtype)
    else:
        q = rand(b, h, m, d).to(dtype)
        k = rand(b, h, n_pad, d).to(dtype)
        v = rand(b, h, n_pad, d).to(dtype)
    return {
        "q": q,
        "k": k,
        "v": v,
        "n_ctx": int(kw["n_ctx"]),
        "k_start": int(kw.get("k_start", 0)),
        "block_n": int(kw.get("block_n", 64)),
    }


def _shape(
    name: str,
    *,
    batch: int = 1,
    heads: int = 1,
    q_len: int = 4,
    head_dim: int = 32,
    n_ctx: int = 128,
    n_pad: int | None = None,
    k_start: int = 0,
    block_n: int = 64,
    dtype: str = "float32",
    noncontig: bool = False,
) -> ShapeSpec:
    pad = (k_start + n_ctx) if n_pad is None else n_pad
    if k_start + n_ctx > pad:
        raise ValueError(f"attention shape {name!r}: window [{k_start}, {k_start + n_ctx}) exceeds n_pad {pad}")
    return ShapeSpec(
        name=name,
        kwargs={
            "batch": batch,
            "heads": heads,
            "q_len": q_len,
            "head_dim": head_dim,
            "n_ctx": n_ctx,
            "n_pad": pad,
            "k_start": k_start,
            "block_n": block_n,
            "dtype": dtype,
            "noncontig": noncontig,
        },
    )


SHAPE_SWEEP: list[ShapeSpec] = [
    # seq_len sweep, including both sides of the block boundary
    _shape("ctx1_fp32", batch=1, heads=1, q_len=1, head_dim=16, n_ctx=1),
    _shape("ctx127_fp32", heads=2, q_len=8, n_ctx=127, n_pad=128),
    _shape("ctx128_fp32", batch=2, heads=2, q_len=8, n_ctx=128),
    _shape("ctx129_fp32", heads=2, q_len=8, n_ctx=129, n_pad=192),
    _shape("ctx1023_fp32", q_len=4, n_ctx=1023, n_pad=1024),
    # the 4096 case is kept narrow in every other dim so it stays seconds on CPU
    _shape("ctx4096_fp32", q_len=2, head_dim=16, n_ctx=4096),
    # a whole KV block that lies entirely in the padding behind the window
    _shape("empty_block_fp32", q_len=4, n_ctx=1, n_pad=128),
    # ... and one entirely ahead of it: the first block the loop visits is empty,
    # which is the case that turns a missing guard into a NaN rather than a no-op
    _shape("leading_empty_block_fp32", q_len=4, n_ctx=65, n_pad=192, k_start=64),
    # a window whose start is not on a block boundary
    _shape("unaligned_window_fp32", heads=2, q_len=8, n_ctx=127, n_pad=256, k_start=33),
    # zero-length context: the guarded store path
    _shape("zero_ctx_fp32", q_len=4, n_ctx=0, n_pad=64),
    # head_dim at, and above, the block size
    _shape("headdim_at_block_fp32", head_dim=64, n_ctx=129, n_pad=192),
    _shape("headdim_above_block_fp32", head_dim=128, n_ctx=65, n_pad=128),
    # non-contiguous q/k/v
    _shape("noncontig_fp32", heads=2, q_len=8, n_ctx=127, n_pad=128, noncontig=True),
    # reduced precision
    _shape("ctx129_bf16", heads=2, q_len=8, n_ctx=129, n_pad=192, dtype="bfloat16"),
    _shape("ctx129_fp16", heads=2, q_len=8, n_ctx=129, n_pad=192, dtype="float16"),
    _shape("ctx1023_bf16", q_len=4, n_ctx=1023, n_pad=1024, dtype="bfloat16"),
]


def accum_depth(shape: ShapeSpec) -> int:
    """Terms folded into ``acc`` before it is read: one per valid KV entry."""
    return max(1, int(shape.kwargs["n_ctx"]))


_ITEMSIZE = {"float32": 4, "bfloat16": 2, "float16": 2}


def bytes_moved(shape: ShapeSpec) -> int:
    """(q + out) + (k + v) elements, times the element size.

    bytes = itemsize * B*H*D * (2*M + 2*n_ctx)
    q is read once, out is written once, k and v are each streamed once over
    the valid prefix; the padding is skipped by the empty-block guard.
    """
    kw = shape.kwargs
    it = _ITEMSIZE[str(kw.get("dtype", "float32"))]
    lead = int(kw["batch"]) * int(kw["heads"]) * int(kw["head_dim"])
    return it * lead * (2 * int(kw["q_len"]) + 2 * int(kw["n_ctx"]))


def flops(shape: ShapeSpec) -> int:
    """QK^T plus PV, both dense over the valid context.

    flops = 2*B*H*M*n_ctx*D  (QK^T)  +  2*B*H*M*n_ctx*D  (P@V)
    The softmax itself is O(B*H*M*n_ctx) and is not counted: it is not the
    roofline-relevant term at any head_dim we sweep.
    """
    kw = shape.kwargs
    return (
        4
        * int(kw["batch"])
        * int(kw["heads"])
        * int(kw["q_len"])
        * int(kw["n_ctx"])
        * int(kw["head_dim"])
    )


DENYLIST: tuple[str, ...] = (
    "torch.nn.functional.scaled_dot_product_attention",
    "torch.scaled_dot_product_attention",
    "torch._C._nn.scaled_dot_product_attention",
    "torch.ops.aten.scaled_dot_product_attention",
    "torch.ops.aten._scaled_dot_product_flash_attention",
    "torch.nn.MultiheadAttention",
    "torch.nn.functional.multi_head_attention_forward",
    "torch.nn.functional.softmax",
    "torch.softmax",
    "torch.Tensor.softmax",
    "torch.nn.Softmax",
    "torch.logsumexp",
    "torch.einsum",
    "crucible.seeds.attention.attention_reference",
    "crucible.seeds.attention.blocked_attention_fwd",
)


SEED: SeedSpec = register(
    SeedSpec(
        id="attention.blocked_fwd",
        domain="triton",
        tiers=("T2", "T3", "T4", "T5"),
        description=(
            "Flash-attention-style blocked forward over a padded KV cache: KV block "
            "loop, boundary mask, fp32 accumulator, online softmax rescale, one store."
        ),
        entry="blocked_attention_fwd",
        source=SOURCE,
        make_inputs=make_inputs,
        reference=attention_reference,
        shape_sweep=list(SHAPE_SWEEP),
        accum_depth=accum_depth,
        bytes_moved=bytes_moved,
        flops=flops,
        denylist=DENYLIST,
        supports_cpu=True,
        extras={
            "call_style": "kwargs",
            "reference_op": "torch.nn.functional.scaled_dot_product_attention",
            "block_arg": "block_n",
        },
    )
)

__all__ = [
    "SEED",
    "SOURCE",
    "SHAPE_SWEEP",
    "DENYLIST",
    "blocked_attention_fwd",
    "attention_reference",
    "make_inputs",
    "accum_depth",
    "bytes_moved",
    "flops",
]
