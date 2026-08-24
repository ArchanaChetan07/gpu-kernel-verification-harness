"""Seed: a three-level tiled GEMM over a padded K axis, written in pure torch.

This is the CUDA-domain baseline. It is written the way a CUDA tiled GEMM is
written -- an (M, N) tile grid, a K block loop, shared-memory-shaped staging
buffers addressed by lane id, a boundary mask on the K tail, an fp32
accumulator and one store per output tile -- rather than as a call to a fused
op, so that the mutation engine has the same sites a real kernel has.

The K accumulation is Kahan-compensated. That is not decoration: it makes the
running total structurally distinct from the per-block partial (the total needs
the compensation term the partial never sees), and it makes the snapshot before
the in-place update load-bearing rather than cosmetic.

The independent ground truth is ``torch.matmul`` over the valid K prefix: one
fused, unblocked, uncompensated call into ATen.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch

from ..schema import ShapeSpec
from .registry import SeedSpec, register

_SOURCE_HEADER = '''"""Tiled matmul over a padded K axis: known-good baseline."""

from __future__ import annotations

import torch


'''


def tiled_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    k_valid: int,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 64,
) -> torch.Tensor:
    """``a[:, :k_valid] @ b[:k_valid, :]`` computed tile by tile.

    ``a`` is ``(M, K_pad)`` and ``b`` is ``(K_pad, N)``. Rows of ``b`` and
    columns of ``a`` at or beyond ``k_valid`` are allocation slack and must not
    reach the accumulator.
    """
    m_size = int(a.shape[0])
    k_pad = int(a.shape[1])
    n_size = int(b.shape[1])
    k = int(k_valid)
    bm, bn, bk = int(block_m), int(block_n), int(block_k)

    out = torch.empty((m_size, n_size), dtype=a.dtype, device=a.device)
    if k <= 0 or m_size == 0 or n_size == 0:
        out.zero_()  # explicit store on the degenerate path
        return out

    n_kblocks = (k_pad + bk - 1) // bk
    for m0 in range(0, m_size, bm):
        m_hi = min(m0 + bm, m_size)
        for n0 in range(0, n_size, bn):
            n_hi = min(n0 + bn, n_size)
            # fp32 accumulator plus its Kahan compensation term, both pinned to
            # fp32 whatever the operand dtype is: this is where a K-deep
            # reduction either keeps its low bits or silently loses them.
            acc = torch.zeros((m_hi - m0, n_hi - n0), dtype=torch.float32, device=a.device)
            comp = torch.zeros((m_hi - m0, n_hi - n0), dtype=torch.float32, device=a.device)

            for kb in range(n_kblocks):
                k0 = kb * bk
                offs_k = k0 + torch.arange(bk, device=a.device)
                mask_k = offs_k < k  # boundary mask on the K tail
                if int(mask_k.sum()) == 0:
                    # This K block lies entirely in the padding: no lane in it
                    # can pass the mask, so it can only cost time. The guard is
                    # also what keeps a zero valid-lane count from reaching the
                    # staging and reduction code below.
                    continue

                a_view = a[m0:m_hi, k0 : k0 + bk]
                b_view = b[k0 : k0 + bk, n0:n_hi]
                width = int(a_view.shape[1])
                lane = torch.arange(width, device=a.device)
                lane_mask = mask_k.to(torch.float32)

                # Staging buffers sized by the block's lane count -- not by the
                # number of valid lanes -- because they are addressed by lane id.
                a_buf = torch.zeros((m_hi - m0, bk), dtype=torch.float32, device=a.device)
                b_buf = torch.zeros((bk, n_hi - n0), dtype=torch.float32, device=a.device)
                # `a` may arrive as a transposed view, so the K tile is
                # materialised row-major once instead of being re-strided on
                # every access inside the tile product.
                a_buf[:, lane] = a_view.to(torch.float32).contiguous()
                b_buf[lane, :] = b_view.to(torch.float32).contiguous()
                a_buf.mul_(lane_mask.unsqueeze(0))
                b_buf.mul_(lane_mask.unsqueeze(1))

                part = a_buf @ b_buf  # this block's partial, not the total

                # Cross-block combine, Kahan compensated: the running total
                # carries the low-order bits the previous add discarded, which
                # no single per-block partial contains. `acc` is updated in
                # place, so the pre-update value is snapshotted first -- without
                # the copy the compensation term collapses to -y and the
                # correction is applied to the wrong iteration.
                y = part - comp
                acc_prev = acc.clone()
                acc.add_(y)
                comp = (acc - acc_prev) - y

            out[m0:m_hi, n0:n_hi] = acc.to(out.dtype)  # explicit store for this tile
    return out


SOURCE = _SOURCE_HEADER + inspect.getsource(tiled_matmul)


def matmul_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    k_valid: int,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 64,
) -> torch.Tensor:
    """Independent ground truth: one fused ``torch.matmul`` on the valid prefix.

    The tile parameters are accepted and ignored; the reference is not tiled.
    """
    k = int(k_valid)
    if k <= 0:
        return torch.zeros((int(a.shape[0]), int(b.shape[1])), dtype=a.dtype, device=a.device)
    return torch.matmul(a[:, :k].contiguous(), b[:k, :].contiguous())


def _generator(shape: ShapeSpec, device: Any) -> torch.Generator:
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(int(shape.key(), 16) % (2**31 - 1))
    return g


def make_inputs(shape: ShapeSpec, device: Any = "cpu", generator: Any = None) -> dict[str, Any]:
    kw = shape.kwargs
    gen = generator if generator is not None else _generator(shape, device)
    dtype = getattr(torch, str(kw.get("dtype", "float32")))
    m = int(kw["m"])
    n = int(kw["n"])
    k_pad = int(kw["k_pad"])

    def rand(*dims: int) -> torch.Tensor:
        return torch.randn(dims, generator=gen, device=device, dtype=torch.float32)

    if bool(kw.get("noncontig", False)):
        a = rand(k_pad, m).transpose(0, 1).to(dtype)
        b = rand(n, k_pad).transpose(0, 1).to(dtype)
    else:
        a = rand(m, k_pad).to(dtype)
        b = rand(k_pad, n).to(dtype)
    return {
        "a": a,
        "b": b,
        "k_valid": int(kw["k_valid"]),
        "block_m": int(kw.get("block_m", 32)),
        "block_n": int(kw.get("block_n", 32)),
        "block_k": int(kw.get("block_k", 64)),
    }


def _shape(
    name: str,
    *,
    m: int = 8,
    n: int = 8,
    k_valid: int = 128,
    k_pad: int | None = None,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 64,
    dtype: str = "float32",
    noncontig: bool = False,
) -> ShapeSpec:
    return ShapeSpec(
        name=name,
        kwargs={
            "m": m,
            "n": n,
            "k_valid": k_valid,
            "k_pad": k_valid if k_pad is None else k_pad,
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "dtype": dtype,
            "noncontig": noncontig,
        },
    )


SHAPE_SWEEP: list[ShapeSpec] = [
    # single-element operands, and the K sweep across the block boundary
    _shape("k1_fp32", m=1, n=1, k_valid=1),
    _shape("k127_fp32", k_valid=127, k_pad=128),
    _shape("k128_fp32", k_valid=128),
    _shape("k129_fp32", k_valid=129, k_pad=192),
    _shape("k1023_fp32", m=4, n=4, k_valid=1023, k_pad=1024),
    # deepest reduction, narrow in M and N so it stays seconds on CPU
    _shape("k4096_fp32", m=4, n=4, k_valid=4096),
    # a whole K block inside the padding
    _shape("empty_block_fp32", k_valid=1, k_pad=128),
    # zero-length K: the guarded store path
    _shape("zero_k_fp32", k_valid=0, k_pad=64),
    # M and N exactly at, and just above, the tile size
    _shape("tile_at_block_fp32", m=32, n=32, k_valid=65, k_pad=128),
    _shape("tile_above_block_fp32", m=33, n=33, k_valid=65, k_pad=128),
    # non-contiguous operands
    _shape("noncontig_fp32", k_valid=127, k_pad=128, noncontig=True),
    # reduced precision
    _shape("k129_bf16", k_valid=129, k_pad=192, dtype="bfloat16"),
    _shape("k129_fp16", k_valid=129, k_pad=192, dtype="float16"),
    _shape("k1023_bf16", m=4, n=4, k_valid=1023, k_pad=1024, dtype="bfloat16"),
]


def accum_depth(shape: ShapeSpec) -> int:
    """Terms in each output element's dot product: one per valid K index."""
    return max(1, int(shape.kwargs["k_valid"]))


_ITEMSIZE = {"float32": 4, "bfloat16": 2, "float16": 2}


def bytes_moved(shape: ShapeSpec) -> int:
    """Compulsory traffic for one pass over each operand and the result.

    bytes = itemsize * (M*k_valid + k_valid*N + M*N)
    This is the lower bound with perfect tile reuse; a kernel that re-reads a
    tile per (m, n) pair moves ceil(N/BN) and ceil(M/BM) times more, which is
    what the roofline denominator is meant to expose.
    """
    kw = shape.kwargs
    it = _ITEMSIZE[str(kw.get("dtype", "float32"))]
    m, n, k = int(kw["m"]), int(kw["n"]), int(kw["k_valid"])
    return it * (m * k + k * n + m * n)


def flops(shape: ShapeSpec) -> int:
    """flops = 2 * M * N * k_valid  (one multiply and one add per term).

    The Kahan compensation adds 3 more flops per K *block*, not per term, and
    is not counted.
    """
    kw = shape.kwargs
    return 2 * int(kw["m"]) * int(kw["n"]) * int(kw["k_valid"])


DENYLIST: tuple[str, ...] = (
    "torch.matmul",
    "torch.mm",
    "torch.bmm",
    "torch.addmm",
    "torch.addbmm",
    "torch.baddbmm",
    "torch.einsum",
    "torch.tensordot",
    "torch.inner",
    "torch.Tensor.matmul",
    "torch.Tensor.mm",
    "torch.nn.functional.linear",
    "torch.nn.Linear",
    "torch.ops.aten.mm",
    "torch.ops.aten.addmm",
    "numpy.matmul",
    "numpy.dot",
    "crucible.seeds.matmul.matmul_reference",
    "crucible.seeds.matmul.tiled_matmul",
)


SEED: SeedSpec = register(
    SeedSpec(
        id="matmul.tiled",
        domain="cuda",
        tiers=("T2", "T3", "T4", "T5"),
        description=(
            "Three-level tiled GEMM over a padded K axis: (M, N) tile grid, K block "
            "loop with a tail mask, lane-indexed staging buffers, an fp32 "
            "Kahan-compensated accumulator and one store per output tile."
        ),
        entry="tiled_matmul",
        source=SOURCE,
        make_inputs=make_inputs,
        reference=matmul_reference,
        shape_sweep=list(SHAPE_SWEEP),
        accum_depth=accum_depth,
        bytes_moved=bytes_moved,
        flops=flops,
        denylist=DENYLIST,
        supports_cpu=True,
        extras={
            "call_style": "kwargs",
            "reference_op": "torch.matmul",
            "block_arg": "block_k",
        },
    )
)

__all__ = [
    "SEED",
    "SOURCE",
    "SHAPE_SWEEP",
    "DENYLIST",
    "tiled_matmul",
    "matmul_reference",
    "make_inputs",
    "accum_depth",
    "bytes_moved",
    "flops",
]
