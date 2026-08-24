"""Seed: a group-wise int4 weight-only dequant-and-matmul, in pure torch.

The weight matrix arrives packed two 4-bit values to a byte along N, with one
scale per (K group, N column) and one zero point per K group. The baseline
unpacks, dequantizes and accumulates inside a K block loop -- the shape a real
CUDA weight-only GEMM has -- so the nibble order, the group index arithmetic,
the tail mask and the accumulator dtype are all present as source, not hidden
inside a fused op.

The independent ground truth dequantizes the whole valid prefix in one
unblocked, unmasked pass and hands it to ``torch.matmul``.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch

from ..schema import ShapeSpec
from .registry import SeedSpec, register

_SOURCE_HEADER = '''"""int4 dequant + matmul over a padded K axis: known-good baseline."""

from __future__ import annotations

import torch


'''


def int4_dequant_matmul(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    k_valid: int,
    group_size: int = 32,
    block_m: int = 32,
    block_k: int = 64,
) -> torch.Tensor:
    """``a[:, :k_valid] @ dequant(b_packed)[:k_valid, :]``.

    ``a`` is ``(M, K_pad)``; ``b_packed`` is ``(K_pad, N // 2)`` uint8 holding
    output column ``2j`` in the low nibble and ``2j + 1`` in the high nibble;
    ``scales`` is ``(n_groups, N)`` and ``zero_points`` is ``(n_groups,)``, with
    group ``g`` covering K indices ``[g * group_size, (g + 1) * group_size)``.
    """
    m_size = int(a.shape[0])
    k_pad = int(a.shape[1])
    n_half = int(b_packed.shape[1])
    n_size = n_half * 2
    k = int(k_valid)
    bm, bk = int(block_m), int(block_k)
    n_groups = int(scales.shape[0])

    out = torch.empty((m_size, n_size), dtype=a.dtype, device=a.device)
    if k <= 0 or m_size == 0 or n_size == 0:
        out.zero_()  # explicit store on the degenerate path
        return out

    n_kblocks = (k_pad + bk - 1) // bk
    for m0 in range(0, m_size, bm):
        m_hi = min(m0 + bm, m_size)
        # fp32 accumulator and its Kahan compensation term. Dequantized weights
        # span several orders of magnitude across groups, so the low bits of the
        # running total are exactly what a K-deep quantized GEMM stands to lose.
        acc = torch.zeros((m_hi - m0, n_size), dtype=torch.float32, device=a.device)
        comp = torch.zeros((m_hi - m0, n_size), dtype=torch.float32, device=a.device)

        for kb in range(n_kblocks):
            k0 = kb * bk
            offs_k = k0 + torch.arange(bk, device=a.device)
            mask_k = offs_k < k  # boundary mask on the K tail
            if int(mask_k.sum()) == 0:
                # Entirely padding. The packed bytes there are allocation slack
                # and dequantize to a nonzero weight, so the lane mask below has
                # to zero them; skipping the block keeps that whole unpack,
                # gather and dequant sequence off the critical path.
                continue

            byte_view = b_packed[k0 : k0 + bk, :]
            width = int(byte_view.shape[0])
            lane = torch.arange(width, device=a.device)
            low = (byte_view & 0x0F).to(torch.int16)
            high = ((byte_view >> 4) & 0x0F).to(torch.int16)
            # Staging buffer sized by the block's lane count -- not by the number
            # of valid lanes -- because it is addressed by lane id.
            q_buf = torch.zeros((bk, n_half, 2), dtype=torch.int16, device=a.device)
            q_buf[lane, :, 0] = low  # even output columns live in the low nibble
            q_buf[lane, :, 1] = high  # odd output columns live in the high nibble
            q_flat = q_buf.reshape(bk, n_size).to(torch.float32)

            group = (offs_k // int(group_size)).clamp_max(n_groups - 1)
            scale_tile = scales.index_select(0, group).to(torch.float32)
            # One zero point per group, broadcast across the N columns instead of
            # replicated per column: the offset does not vary along N.
            zero_tile = (
                zero_points.index_select(0, group).to(torch.float32).unsqueeze(1).expand(bk, n_size)
            )
            lane_mask = mask_k.to(torch.float32)
            w_buf = (q_flat - zero_tile) * scale_tile
            w_buf.mul_(lane_mask.unsqueeze(1))

            a_view = a[m0:m_hi, k0 : k0 + bk]
            a_buf = torch.zeros((m_hi - m0, bk), dtype=torch.float32, device=a.device)
            # `a` may arrive as a transposed view; the K tile is materialised
            # row-major once rather than re-strided inside the tile product.
            a_buf[:, lane] = a_view.to(torch.float32).contiguous()
            a_buf.mul_(lane_mask.unsqueeze(0))

            part = a_buf @ w_buf  # this block's partial, not the total

            # Cross-block combine, Kahan compensated: the running total carries
            # the low-order bits the previous add dropped, which no per-block
            # partial contains. `acc` is updated in place, so its pre-update
            # value is snapshotted first -- without the copy the compensation
            # term collapses to -y and corrects the wrong iteration.
            y = part - comp
            acc_prev = acc.clone()
            acc.add_(y)
            comp = (acc - acc_prev) - y

        out[m0:m_hi, :] = acc.to(out.dtype)  # explicit store for this M tile
    return out


SOURCE = _SOURCE_HEADER + inspect.getsource(int4_dequant_matmul)


def quant_reference(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    k_valid: int,
    group_size: int = 32,
    block_m: int = 32,
    block_k: int = 64,
) -> torch.Tensor:
    """Independent ground truth: dequantize the valid prefix, then ``torch.matmul``.

    No blocking, no masks, no compensation; the tile parameters are ignored.
    """
    m_size = int(a.shape[0])
    n_half = int(b_packed.shape[1])
    n_size = n_half * 2
    k = int(k_valid)
    if k <= 0 or m_size == 0 or n_size == 0:
        return torch.zeros((m_size, n_size), dtype=a.dtype, device=a.device)
    packed = b_packed[:k, :]
    low = (packed & 0x0F).to(torch.int32)
    high = ((packed >> 4) & 0x0F).to(torch.int32)
    q = torch.stack((low, high), dim=-1).reshape(k, n_size).to(torch.float32)
    group = (torch.arange(k, device=a.device) // int(group_size)).clamp_max(int(scales.shape[0]) - 1)
    zero = zero_points.index_select(0, group).to(torch.float32).unsqueeze(1)
    weight = (q - zero) * scales.index_select(0, group).to(torch.float32)
    prod = torch.matmul(a[:, :k].to(torch.float32).contiguous(), weight)
    return prod.to(a.dtype)


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
    if n % 2 != 0:
        raise ValueError(f"quant shape {shape.name!r}: N must be even (two int4 per byte), got {n}")
    k_pad = int(kw["k_pad"])
    group_size = int(kw["group_size"])
    n_groups = max(1, (k_pad + group_size - 1) // group_size)

    if bool(kw.get("noncontig", False)):
        a = torch.randn((k_pad, m), generator=gen, device=device, dtype=torch.float32)
        a = a.transpose(0, 1).to(dtype)
    else:
        a = torch.randn((m, k_pad), generator=gen, device=device, dtype=torch.float32).to(dtype)
    b_packed = torch.randint(
        0, 256, (k_pad, n // 2), generator=gen, device=device, dtype=torch.uint8
    )
    # Positive, well-separated per-(group, column) scales, and an integer zero
    # point in the int4 range so the dequantized weights straddle zero.
    scales = 0.01 + 0.05 * torch.rand((n_groups, n), generator=gen, device=device, dtype=torch.float32)
    zero_points = torch.randint(
        0, 16, (n_groups,), generator=gen, device=device, dtype=torch.int64
    ).to(torch.float32)
    return {
        "a": a,
        "b_packed": b_packed,
        "scales": scales,
        "zero_points": zero_points,
        "k_valid": int(kw["k_valid"]),
        "group_size": group_size,
        "block_m": int(kw.get("block_m", 32)),
        "block_k": int(kw.get("block_k", 64)),
    }


def _shape(
    name: str,
    *,
    m: int = 8,
    n: int = 8,
    k_valid: int = 128,
    k_pad: int | None = None,
    group_size: int = 32,
    block_m: int = 32,
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
            "group_size": group_size,
            "block_m": block_m,
            "block_k": block_k,
            "dtype": dtype,
            "noncontig": noncontig,
        },
    )


SHAPE_SWEEP: list[ShapeSpec] = [
    # a single row, a single int4 pair, a single K index
    _shape("k1_fp32", m=1, n=2, k_valid=1, k_pad=32),
    _shape("k127_fp32", k_valid=127, k_pad=128),
    _shape("k128_fp32", k_valid=128),
    _shape("k129_fp32", k_valid=129, k_pad=192),
    _shape("k1023_fp32", m=4, n=8, k_valid=1023, k_pad=1024),
    # deepest reduction, narrow in M and N so it stays seconds on CPU
    _shape("k4096_fp32", m=2, n=8, k_valid=4096),
    # a whole K block inside the padding
    _shape("empty_block_fp32", k_valid=1, k_pad=128),
    # zero-length K: the guarded store path
    _shape("zero_k_fp32", k_valid=0, k_pad=64),
    # group size exactly at the K block size, and above it (a block then sits
    # strictly inside one group, so the group index never advances mid-block)
    _shape("group_at_block_fp32", k_valid=129, k_pad=192, group_size=64, block_k=64),
    _shape("group_above_block_fp32", k_valid=129, k_pad=192, group_size=128, block_k=64),
    # a K tail that ends part-way through a group
    _shape("group_tail_fp32", k_valid=100, k_pad=128, group_size=32),
    # non-contiguous activations
    _shape("noncontig_fp32", k_valid=127, k_pad=128, noncontig=True),
    # reduced precision activations
    _shape("k129_bf16", k_valid=129, k_pad=192, dtype="bfloat16"),
    _shape("k129_fp16", k_valid=129, k_pad=192, dtype="float16"),
    _shape("k1023_bf16", m=4, n=8, k_valid=1023, k_pad=1024, dtype="bfloat16"),
]


def accum_depth(shape: ShapeSpec) -> int:
    """Terms in each output element's dot product: one per valid K index."""
    return max(1, int(shape.kwargs["k_valid"]))


_ITEMSIZE = {"float32": 4, "bfloat16": 2, "float16": 2}


def bytes_moved(shape: ShapeSpec) -> int:
    """Activations and result at full width, weights at half a byte per element.

    bytes = itemsize*(M*k_valid + M*N) + k_valid*N/2 + 4*(n_groups*N + n_groups)
    The 4x saving on the weight stream is the entire point of int4; a candidate
    that dequantizes to fp16 up front moves 4x more and the roofline says so.
    """
    kw = shape.kwargs
    it = _ITEMSIZE[str(kw.get("dtype", "float32"))]
    m, n, k = int(kw["m"]), int(kw["n"]), int(kw["k_valid"])
    group_size = int(kw["group_size"])
    n_groups = max(1, (int(kw["k_pad"]) + group_size - 1) // group_size)
    return it * (m * k + m * n) + (k * n) // 2 + 4 * (n_groups * n + n_groups)


def flops(shape: ShapeSpec) -> int:
    """flops = 2*M*N*k_valid (the GEMM) + 2*N*k_valid (subtract, then scale).

    The nibble unpack is integer work and is not counted as floating point.
    """
    kw = shape.kwargs
    m, n, k = int(kw["m"]), int(kw["n"]), int(kw["k_valid"])
    return 2 * m * n * k + 2 * n * k


DENYLIST: tuple[str, ...] = (
    "torch.matmul",
    "torch.mm",
    "torch.bmm",
    "torch.addmm",
    "torch.einsum",
    "torch.tensordot",
    "torch.Tensor.matmul",
    "torch.nn.functional.linear",
    "torch.nn.Linear",
    "torch.ops.aten.mm",
    "torch.ops.aten.addmm",
    "torch._weight_int4pack_mm",
    "torch.ops.aten._weight_int4pack_mm",
    "torch.ops.aten._convert_weight_to_int4pack",
    "torch.ao.quantization",
    "torch.ao.nn.quantized",
    "torch.nn.quantized",
    "bitsandbytes.functional.gemv_4bit",
    "bitsandbytes.matmul_4bit",
    "bitsandbytes.nn.Linear4bit",
    "awq.modules.linear",
    "auto_gptq.nn_modules.qlinear",
    "crucible.seeds.quant.quant_reference",
    "crucible.seeds.quant.int4_dequant_matmul",
)


SEED: SeedSpec = register(
    SeedSpec(
        id="quant.int4_dequant_matmul",
        domain="cuda",
        tiers=("T2", "T3", "T4", "T5"),
        description=(
            "Group-wise int4 weight-only GEMM: nibble unpack in a K block loop, "
            "per-group scale gather with a broadcast zero point, a tail mask, an "
            "fp32 Kahan-compensated accumulator and one store per M tile."
        ),
        entry="int4_dequant_matmul",
        source=SOURCE,
        make_inputs=make_inputs,
        reference=quant_reference,
        shape_sweep=list(SHAPE_SWEEP),
        accum_depth=accum_depth,
        bytes_moved=bytes_moved,
        flops=flops,
        denylist=DENYLIST,
        supports_cpu=True,
        extras={
            "call_style": "kwargs",
            "reference_op": "dequantize + torch.matmul",
            "block_arg": "block_k",
            "packing": "two int4 per uint8 along N; low nibble is the even column",
        },
    )
)

__all__ = [
    "SEED",
    "SOURCE",
    "SHAPE_SWEEP",
    "DENYLIST",
    "int4_dequant_matmul",
    "quant_reference",
    "make_inputs",
    "accum_depth",
    "bytes_moved",
    "flops",
]
