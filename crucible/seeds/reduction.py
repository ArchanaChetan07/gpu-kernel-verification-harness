"""Seed: a blocked LayerNorm forward over a padded feature axis, in pure torch.

A normalization kernel is a reduction kernel wearing a hat: it is the canonical
place where a partial-vs-total confusion, a tail-mask off-by-one, or a low
precision accumulator produces an answer that is right on aligned shapes and
quietly wrong on the shapes that matter. The baseline therefore keeps the three
passes explicit (sum, centred sum of squares, normalise-and-store), each with
its own block loop, boundary mask, zero-length guard and fp32 accumulator.

The independent ground truth is ``F.layer_norm`` on the valid prefix: Welford's
online algorithm inside ATen, no blocking, no masks -- a genuinely different
route to the same number.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn.functional as F

from ..schema import ShapeSpec
from .registry import SeedSpec, register

_SOURCE_HEADER = '''"""Blocked LayerNorm forward: known-good baseline."""

from __future__ import annotations

import torch


'''


def blocked_layernorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    n_valid: int,
    block_n: int = 64,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm over ``x[:, :n_valid]``; the padded tail is written as zeros.

    ``x`` is ``(rows, N_pad)`` and ``weight`` / ``bias`` are ``(N_pad,)``. Only
    the first ``n_valid`` columns carry data; the rest is allocation slack.
    """
    rows = int(x.shape[0])
    n_pad = int(x.shape[-1])
    width_limit = int(block_n)
    n = int(n_valid)

    # One buffer, written on every return path.
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if n <= 0:
        out.zero_()
        return out

    # Both reduction accumulators are fp32 regardless of the input dtype: their
    # depth is the full feature width, not one block.
    sum_acc = torch.zeros(rows, dtype=torch.float32, device=x.device)
    var_acc = torch.zeros(rows, dtype=torch.float32, device=x.device)
    n_blocks = (n_pad + width_limit - 1) // width_limit

    # Pass 1 -- partial sums.
    for blk in range(n_blocks):
        start = blk * width_limit
        view = x[:, start : start + width_limit]
        width = int(view.shape[-1])
        lane = torch.arange(width, device=x.device)
        offs = start + lane
        mask = offs < n  # boundary mask: the tail block is rarely full
        if int(mask.sum()) == 0:
            # Entirely inside the padding: no lane here can pass the mask, so
            # the block can only cost a pass over allocation slack.
            continue
        # Staging buffer sized by the block's lane count -- not by the number of
        # valid lanes -- because it is addressed by lane id.
        tile = torch.zeros((rows, width_limit), dtype=torch.float32, device=x.device)
        tile[:, lane] = view.to(torch.float32) * mask.to(torch.float32)
        sum_acc.add_(tile.sum(dim=-1))

    # Cross-block combine: the mean is a property of the whole row and cannot be
    # formed from any single block's partial sum.
    mean = sum_acc / float(n)

    # Pass 2 -- centred sum of squares. Centring needs the mean, so it cannot be
    # fused into pass 1 without trading it for the sumsq - mean^2 identity, which
    # cancels catastrophically once the row mean is large relative to its spread.
    for blk in range(n_blocks):
        start = blk * width_limit
        view = x[:, start : start + width_limit]
        width = int(view.shape[-1])
        lane = torch.arange(width, device=x.device)
        offs = start + lane
        mask = offs < n
        if int(mask.sum()) == 0:
            continue
        # `.to()` hands back x's own tile unchanged when x is already fp32, so
        # the in-place centring below would write through into x -- which pass 3
        # reads again. This copy is the boundary between the two passes.
        centred = view.to(torch.float32).clone()
        centred.sub_(mean.unsqueeze(-1))
        centred.mul_(mask.to(torch.float32))
        var_acc.add_((centred * centred).sum(dim=-1))

    var = var_acc / float(n)
    rstd = torch.rsqrt(var + float(eps))

    # Pass 3 -- normalise and store.
    for blk in range(n_blocks):
        start = blk * width_limit
        view = x[:, start : start + width_limit]
        width = int(view.shape[-1])
        lane = torch.arange(width, device=x.device)
        offs_full = start + torch.arange(width_limit, device=x.device)
        mask_full = (offs_full < n).to(torch.float32)
        if int(mask_full.sum()) == 0:
            # Nothing valid here, but the output tile still has to be defined.
            out[:, start : start + width] = torch.zeros(
                (rows, width), dtype=out.dtype, device=out.device
            )
            continue
        stage = torch.zeros((rows, width_limit), dtype=torch.float32, device=x.device)
        stage[:, lane] = view.to(torch.float32)
        w_tile = torch.zeros(width_limit, dtype=torch.float32, device=x.device)
        b_tile = torch.zeros(width_limit, dtype=torch.float32, device=x.device)
        w_tile[lane] = weight[start : start + width_limit].to(torch.float32)
        b_tile[lane] = bias[start : start + width_limit].to(torch.float32)
        # One mean per row, broadcast across the lanes rather than replicated:
        # expand keeps the stride at zero, so the tile is read once from cache.
        mu = mean.unsqueeze(-1).expand(rows, width_limit)
        norm = (stage - mu) * rstd.unsqueeze(-1)
        norm = norm * w_tile + b_tile
        norm = norm * mask_full
        out[:, start : start + width] = norm[:, :width].to(out.dtype)  # explicit store
    return out


SOURCE = _SOURCE_HEADER + inspect.getsource(blocked_layernorm_fwd)


def layernorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    n_valid: int,
    block_n: int = 64,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Independent ground truth: ATen's fused LayerNorm on the valid prefix.

    ``block_n`` is accepted and ignored; the reference has no blocking at all.
    """
    out = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    n = int(n_valid)
    if n <= 0:
        return out
    out[:, :n] = F.layer_norm(
        x[:, :n].contiguous(),
        (n,),
        weight[:n].contiguous(),
        bias[:n].contiguous(),
        float(eps),
    )
    return out


def _generator(shape: ShapeSpec, device: Any) -> torch.Generator:
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(int(shape.key(), 16) % (2**31 - 1))
    return g


def make_inputs(shape: ShapeSpec, device: Any = "cpu", generator: Any = None) -> dict[str, Any]:
    kw = shape.kwargs
    gen = generator if generator is not None else _generator(shape, device)
    dtype = getattr(torch, str(kw.get("dtype", "float32")))
    rows = int(kw["rows"])
    n_pad = int(kw["n_pad"])

    def rand(*dims: int) -> torch.Tensor:
        return torch.randn(dims, generator=gen, device=device, dtype=torch.float32)

    if bool(kw.get("noncontig", False)):
        x = rand(n_pad, rows).transpose(0, 1).to(dtype)
    else:
        x = rand(rows, n_pad).to(dtype)
    weight = (1.0 + 0.1 * rand(n_pad)).to(dtype)
    bias = (0.1 * rand(n_pad)).to(dtype)
    return {
        "x": x,
        "weight": weight,
        "bias": bias,
        "n_valid": int(kw["n_valid"]),
        "block_n": int(kw.get("block_n", 64)),
        "eps": float(kw.get("eps", 1e-5)),
    }


def _shape(
    name: str,
    *,
    rows: int = 4,
    n_valid: int = 128,
    n_pad: int | None = None,
    block_n: int = 64,
    dtype: str = "float32",
    noncontig: bool = False,
) -> ShapeSpec:
    return ShapeSpec(
        name=name,
        kwargs={
            "rows": rows,
            "n_valid": n_valid,
            "n_pad": n_valid if n_pad is None else n_pad,
            "block_n": block_n,
            "eps": 1e-5,
            "dtype": dtype,
            "noncontig": noncontig,
        },
    )


SHAPE_SWEEP: list[ShapeSpec] = [
    # a single row with a single feature: variance is exactly zero
    _shape("n1_fp32", rows=1, n_valid=1),
    _shape("n127_fp32", n_valid=127, n_pad=128),
    _shape("n128_fp32", n_valid=128),
    _shape("n129_fp32", n_valid=129, n_pad=192),
    _shape("n1023_fp32", rows=2, n_valid=1023, n_pad=1024),
    # widest case, kept to two rows so it stays seconds on CPU
    _shape("n4096_fp32", rows=2, n_valid=4096),
    # a whole feature block inside the padding
    _shape("empty_block_fp32", rows=3, n_valid=3, n_pad=128),
    # zero-length: the guarded store path
    _shape("zero_len_fp32", rows=3, n_valid=0, n_pad=64),
    # feature width exactly at, and just above, the block size
    _shape("width_at_block_fp32", n_valid=64, block_n=64),
    _shape("width_above_block_fp32", n_valid=200, n_pad=256, block_n=64),
    # non-contiguous rows
    _shape("noncontig_fp32", n_valid=127, n_pad=128, noncontig=True),
    # reduced precision
    _shape("n129_bf16", n_valid=129, n_pad=192, dtype="bfloat16"),
    _shape("n129_fp16", n_valid=129, n_pad=192, dtype="float16"),
    _shape("n1023_bf16", rows=2, n_valid=1023, n_pad=1024, dtype="bfloat16"),
]


def accum_depth(shape: ShapeSpec) -> int:
    """Terms in the deepest reduction: one per valid feature, twice over."""
    return max(1, int(shape.kwargs["n_valid"]))


_ITEMSIZE = {"float32": 4, "bfloat16": 2, "float16": 2}


def bytes_moved(shape: ShapeSpec) -> int:
    """Three read passes over the valid prefix, one write over the whole output.

    bytes = itemsize * (3*rows*n_valid + rows*n_pad + 2*n_valid)
    The output is written across its full allocated width -- including the padded
    tail, which is stored as zeros -- so the count stays honest at n_valid = 0.
    A single-pass Welford kernel would move a third of the read traffic; that gap
    is exactly what the roofline check is asked to notice.
    """
    kw = shape.kwargs
    it = _ITEMSIZE[str(kw.get("dtype", "float32"))]
    rows = int(kw["rows"])
    n = int(kw["n_valid"])
    n_pad = int(kw["n_pad"])
    return it * (3 * rows * n + rows * n_pad + 2 * n)


def flops(shape: ShapeSpec) -> int:
    """Per element: 1 add (pass 1), 3 (sub, mul, add) (pass 2), 4 (pass 3).

    flops = 8 * rows * n_valid
    The per-row rsqrt is O(rows) and is not counted.
    """
    kw = shape.kwargs
    return 8 * int(kw["rows"]) * int(kw["n_valid"])


DENYLIST: tuple[str, ...] = (
    "torch.nn.functional.layer_norm",
    "torch.layer_norm",
    "torch.native_layer_norm",
    "torch.ops.aten.native_layer_norm",
    "torch.ops.aten.layer_norm",
    "torch.nn.LayerNorm",
    "torch.nn.functional.group_norm",
    "torch.nn.functional.rms_norm",
    "torch.nn.functional.normalize",
    "torch.nn.functional.batch_norm",
    "torch.var_mean",
    "torch.std_mean",
    "torch.Tensor.var",
    "torch.Tensor.std",
    "torch.einsum",
    "crucible.seeds.reduction.layernorm_reference",
    "crucible.seeds.reduction.blocked_layernorm_fwd",
)


SEED: SeedSpec = register(
    SeedSpec(
        id="reduction.blocked_layernorm",
        domain="triton",
        tiers=("T2", "T3", "T4", "T5"),
        description=(
            "Three-pass blocked LayerNorm over a padded feature axis: block loops, "
            "boundary masks, fp32 partial sums, an explicit mean/variance combine "
            "and one store per output tile."
        ),
        entry="blocked_layernorm_fwd",
        source=SOURCE,
        make_inputs=make_inputs,
        reference=layernorm_reference,
        shape_sweep=list(SHAPE_SWEEP),
        accum_depth=accum_depth,
        bytes_moved=bytes_moved,
        flops=flops,
        denylist=DENYLIST,
        supports_cpu=True,
        extras={
            "call_style": "kwargs",
            "reference_op": "torch.nn.functional.layer_norm",
            "block_arg": "block_n",
        },
    )
)

__all__ = [
    "SEED",
    "SOURCE",
    "SHAPE_SWEEP",
    "DENYLIST",
    "blocked_layernorm_fwd",
    "layernorm_reference",
    "make_inputs",
    "accum_depth",
    "bytes_moved",
    "flops",
]
