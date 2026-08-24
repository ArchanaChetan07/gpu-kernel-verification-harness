"""Seed: an autotuning cache keyed by a shape class.

This is the T4 seed, and the reason it belongs in the bank is that its failure
mode produces **numerically identical output**. A stale tile config still
computes the right GEMM; it just computes it on the wrong tiling. Nothing in a
correctness test can see that, which is precisely why a corpus built from
correctness tests does not contain this failure and models are bad at it.

So the graded quantity is not only ``y``. It is also the config the tuner chose
for each problem and what that choice costs. The cost is a **model**, not a
measurement: an explicit function of padded work, re-read traffic, per-tile
launch overhead and per-k-step loop overhead. That is a deliberate choice -
a wall-clock timing on a shared CPU is not reproducible enough to grade on, and
inventing a measurement we did not take would be worse than modelling one we
did not claim to have taken. O2 is where real timings live; here the number is
declared to be a model and behaves like one.

The cache is keyed by ``_cache_key``, which buckets every dimension to a power
of two, and the tuner tunes on the **class representative** rather than on the
concrete problem. That is what makes the cache semantically invisible when it is
right: the chosen config is a pure function of the key, so a correct
implementation with a cache and a correct implementation without one agree
exactly. Drop a dimension from the key and they stop agreeing - on cost, never
on values.
"""

from __future__ import annotations

from typing import Any

from ..schema import ShapeSpec
from .registry import CompareResult, SeedSpec, register

#: The tile search space. Passed in as an input so the baseline and the ground
#: truth search the same space without sharing code.
DEFAULT_CONFIGS: tuple[tuple[int, int, int], ...] = (
    (16, 16, 16),
    (32, 32, 32),
    (32, 32, 64),
    (32, 64, 32),
    (64, 32, 64),
    (64, 64, 32),
    (64, 64, 64),
    (128, 64, 64),
)

# The modelled machine. Duplicated between the baseline and the reference on
# purpose: these constants are the cost model itself, so a mutation that edits
# one of them has to disagree with something.
NS_PER_MAC = 1.0 / 512.0
NS_PER_BYTE = 1.0 / 256.0
NS_PER_TILE_LAUNCH = 200.0
NS_PER_K_STEP = 60.0
MODEL_ITEMSIZE = 8

# --------------------------------------------------------------------------- #
# the known-good baseline
# --------------------------------------------------------------------------- #

SOURCE = '''"""Blocked GEMM behind a tile-config cache keyed by shape class."""
import torch

NS_PER_MAC = 1.0 / 512.0
NS_PER_BYTE = 1.0 / 256.0
NS_PER_TILE_LAUNCH = 200.0
NS_PER_K_STEP = 60.0
MODEL_ITEMSIZE = 8


def _bucket(v):
    """Round a dimension up to a power of two. Two problems that bucket to the
    same triple are the same shape class and are expected to want the same
    tile."""
    b = 1
    while b < v:
        b *= 2
    return b


def _cache_key(m, n, k):
    """The cache key. EVERY dimension that can change the optimal tile has to
    appear here. k does: with a short reduction a wide k-block pays for padding
    it never uses, and with a long one a narrow k-block pays per-iteration
    overhead it did not need. A key that omits k cannot tell those apart and
    will hand the second problem the first problem's config."""
    return (_bucket(m), _bucket(n), _bucket(k))


def _cost_ns(m, n, k, bm, bn, bk):
    """Modelled cost of running an m*n*k problem on a bm*bn*bk tiling.

    Four terms, each with a reason: work actually issued including the padding
    a tile forces, operand bytes re-read once per k-block per output tile, a
    fixed cost per tile launched, and a fixed cost per k-loop iteration.
    """
    tiles_m = (m + bm - 1) // bm
    tiles_n = (n + bn - 1) // bn
    tiles_k = (k + bk - 1) // bk
    padded_macs = (tiles_m * bm) * (tiles_n * bn) * (tiles_k * bk)
    traffic = tiles_m * tiles_n * tiles_k * (bm * bk + bk * bn) * MODEL_ITEMSIZE
    launches = tiles_m * tiles_n
    k_steps = tiles_m * tiles_n * tiles_k
    return (padded_macs * NS_PER_MAC + traffic * NS_PER_BYTE
            + launches * NS_PER_TILE_LAUNCH + k_steps * NS_PER_K_STEP)


def _autotune(key, configs):
    """Search the config space on the CLASS REPRESENTATIVE, not on the concrete
    problem. That is what makes the cache sound: the answer depends on nothing
    but the key, so caching it cannot change any result. Ties go to the first
    config in the given order, so the search is deterministic."""
    best = None
    best_cost = None
    for cfg in configs:
        bm, bn, bk = int(cfg[0]), int(cfg[1]), int(cfg[2])
        c = _cost_ns(key[0], key[1], key[2], bm, bn, bk)
        if best_cost is None or c < best_cost:
            best_cost = c
            best = (bm, bn, bk)
    return best


def _blocked_matmul(a, b, cfg, acc_dtype):
    """Tiled matmul with an explicit accumulator dtype, an explicit empty-tile
    guard and an explicit store."""
    bm, bn, bk = cfg
    m = int(a.shape[0])
    k = int(a.shape[1])
    n = int(b.shape[1])
    out = torch.zeros(m, n, dtype=acc_dtype, device=a.device)
    for i0 in range(0, m, bm):
        i1 = min(i0 + bm, m)
        if i1 <= i0:                                   # empty-tile guard
            continue
        for j0 in range(0, n, bn):
            j1 = min(j0 + bn, n)
            if j1 <= j0:                               # empty-tile guard
                continue
            acc = torch.zeros(i1 - i0, j1 - j0, dtype=acc_dtype, device=a.device)
            for p0 in range(0, k, bk):
                p1 = min(p0 + bk, k)
                acc = acc + (a[i0:i1, p0:p1].to(acc_dtype)
                             @ b[p0:p1, j0:j1].to(acc_dtype))
            out[i0:i1, j0:j1] = acc                    # explicit store
    return out


def autotuned_gemm(a_flat, b_flat, shapes, configs, accum_dtype="float64"):
    """Run a sequence of GEMMs, tuning each shape class once and reusing it."""
    acc = getattr(torch, accum_dtype)                  # explicit accumulator
    cache = {}
    misses = 0
    ys = []
    chosen = []
    costs = []
    a_off = 0
    b_off = 0
    for spec in shapes:
        m, n, k = int(spec[0]), int(spec[1]), int(spec[2])
        a = a_flat[a_off:a_off + m * k].reshape(m, k)
        a_off += m * k
        b = b_flat[b_off:b_off + k * n].reshape(k, n)
        b_off += k * n

        key = _cache_key(m, n, k)
        cfg = cache.get(key)
        if cfg is None:
            misses += 1
            cfg = _autotune(key, configs)
            cache[key] = cfg

        ys.append(_blocked_matmul(a, b, cfg, acc).reshape(-1))
        chosen.append([cfg[0], cfg[1], cfg[2]])
        costs.append(_cost_ns(m, n, k, cfg[0], cfg[1], cfg[2]))

    if ys:
        y = torch.cat(ys).to(torch.float64)
    else:
        y = torch.zeros(0, dtype=torch.float64, device=a_flat.device)
    if chosen:
        cfg_t = torch.tensor(chosen, dtype=torch.int64)
    else:
        cfg_t = torch.zeros((0, 3), dtype=torch.int64)
    cost_t = torch.tensor(costs, dtype=torch.float64)
    return {
        "y": y,
        "configs_used": cfg_t,
        "cost_ns": cost_t,
        "total_cost_ns": cost_t.sum().to(torch.float64),
        "cache_misses": torch.tensor(misses, dtype=torch.int64),
    }
'''


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def _shapes_of(shape: ShapeSpec) -> list[list[int]]:
    return [[int(m), int(n), int(k)] for m, n, k in shape.kwargs.get("shapes", [])]


def _configs_of(shape: ShapeSpec) -> list[list[int]]:
    raw = shape.kwargs.get("configs")
    if raw is None:
        return [list(c) for c in DEFAULT_CONFIGS]
    return [[int(a), int(b), int(c)] for a, b, c in raw]


def make_inputs(shape: ShapeSpec, device: Any = "cpu", generator: Any = None) -> dict[str, Any]:
    """Operands ride as two flat tensors plus the shape list, so a sequence of
    differently-shaped GEMMs crosses the sandbox boundary as arrays and JSON
    rather than as a pickled list of tensors."""
    import torch

    dtype = getattr(torch, str(shape.kwargs.get("dtype", "float32")))
    shapes = _shapes_of(shape)
    a_n = sum(m * k for m, _, k in shapes)
    b_n = sum(k * n for _, n, k in shapes)
    a_flat = torch.randn(a_n, generator=generator, device=device, dtype=torch.float32)
    b_flat = torch.randn(b_n, generator=generator, device=device, dtype=torch.float32)
    return {
        "a_flat": a_flat.to(dtype),
        "b_flat": b_flat.to(dtype),
        "shapes": shapes,
        "configs": _configs_of(shape),
        "accum_dtype": str(shape.kwargs.get("accum_dtype", "float64")),
    }


# --------------------------------------------------------------------------- #
# independent ground truth
# --------------------------------------------------------------------------- #


def _ref_bucket(v: int) -> int:
    b = 1
    while b < v:
        b *= 2
    return b


def _ref_cost_ns(m: int, n: int, k: int, bm: int, bn: int, bk: int) -> float:
    tiles_m = -(-m // bm)
    tiles_n = -(-n // bn)
    tiles_k = -(-k // bk)
    padded_macs = (tiles_m * bm) * (tiles_n * bn) * (tiles_k * bk)
    traffic = tiles_m * tiles_n * tiles_k * (bm * bk + bk * bn) * MODEL_ITEMSIZE
    return (
        padded_macs * NS_PER_MAC
        + traffic * NS_PER_BYTE
        + tiles_m * tiles_n * NS_PER_TILE_LAUNCH
        + tiles_m * tiles_n * tiles_k * NS_PER_K_STEP
    )


def reference(
    a_flat: Any,
    b_flat: Any,
    shapes: Any,
    configs: Any,
    accum_dtype: str = "float64",
) -> dict[str, Any]:
    """Ground truth: no cache at all, and one fused matmul per problem.

    Both differences are the point. The cache is the thing under test, so the
    reference must not have one - it re-derives the config for every problem
    from the shape class and never remembers anything. And the values come from
    ``torch.matmul`` in fp64, which shares no tiling logic with the baseline, so
    a broken accumulator or a dropped tail tile cannot cancel out.

    The cost model and the bucketing are re-implemented here rather than
    imported: they are the specification the candidate is being held to, so a
    mutation that quietly edits either one has to disagree with this copy.
    """
    import torch

    a = torch.as_tensor(a_flat)
    b = torch.as_tensor(b_flat)
    cfg_space = [(int(c[0]), int(c[1]), int(c[2])) for c in configs]

    ys: list[Any] = []
    chosen: list[list[int]] = []
    costs: list[float] = []
    a_off = 0
    b_off = 0
    for spec in shapes:
        m, n, k = int(spec[0]), int(spec[1]), int(spec[2])
        am = a[a_off : a_off + m * k].reshape(m, k).to(torch.float64)
        a_off += m * k
        bm_ = b[b_off : b_off + k * n].reshape(k, n).to(torch.float64)
        b_off += k * n

        key = (_ref_bucket(m), _ref_bucket(n), _ref_bucket(k))
        best = cfg_space[0]
        best_cost = _ref_cost_ns(key[0], key[1], key[2], *best)
        for cfg in cfg_space[1:]:
            c = _ref_cost_ns(key[0], key[1], key[2], *cfg)
            if c < best_cost:
                best_cost = c
                best = cfg

        ys.append(torch.matmul(am, bm_).reshape(-1))
        chosen.append([best[0], best[1], best[2]])
        costs.append(_ref_cost_ns(m, n, k, *best))

    y = torch.cat(ys).to(torch.float64) if ys else torch.zeros(0, dtype=torch.float64, device=a_flat.device)
    cfg_t = (
        torch.tensor(chosen, dtype=torch.int64)
        if chosen
        else torch.zeros((0, 3), dtype=torch.int64)
    )
    cost_t = torch.tensor(costs, dtype=torch.float64)
    return {
        "y": y,
        "configs_used": cfg_t,
        "cost_ns": cost_t,
        "total_cost_ns": cost_t.sum().to(torch.float64),
        # The reference has no cache, so it declares the miss count it would
        # have had rather than pretending to have measured one.
        "cache_misses": torch.tensor(-1, dtype=torch.int64),
    }


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #

#: fp64 blocked accumulation against a fused fp64 matmul. The only difference is
#: summation order over at most a few k-blocks.
_Y_RTOL = 1.0e-10
_Y_ATOL = 1.0e-12

#: The cost model is exact arithmetic on both sides, so any excess at all is a
#: real regression. The epsilon only absorbs float addition order in the sum.
_COST_SLACK = 1.0e-9


def compare(got: Any, want: Any) -> CompareResult:
    """Values within fp64 tolerance, tile choice exactly, cost never worse.

    ``cache_misses`` is reported but not graded: an implementation with no cache
    is slow, not wrong, and O2 is where that is someone's problem.
    """
    import torch

    if not isinstance(got, dict):
        return CompareResult(
            ok=False, detail=f"expected a dict of outputs, got {type(got).__name__}", kind="shape"
        )
    missing = [k for k in ("y", "configs_used", "cost_ns", "total_cost_ns") if k not in got]
    if missing:
        return CompareResult(
            ok=False, detail=f"output is missing key(s): {', '.join(missing)}", kind="shape"
        )

    gy = torch.as_tensor(got["y"]).to(torch.float64).reshape(-1)
    wy = torch.as_tensor(want["y"]).to(torch.float64).reshape(-1)
    if gy.shape != wy.shape:
        return CompareResult(
            ok=False,
            detail=f"y: shape {tuple(gy.shape)} != reference {tuple(wy.shape)}",
            kind="shape",
        )
    max_abs = 0.0
    max_rel = 0.0
    if gy.numel():
        abs_err = (gy - wy).abs()
        max_abs = float(abs_err.max())
        max_rel = float((abs_err / wy.abs().clamp_min(1e-300)).max())
        bad = abs_err > (_Y_ATOL + _Y_RTOL * wy.abs())
        if bool(bad.any()):
            i = int(torch.nonzero(bad)[0])
            return CompareResult(
                ok=False,
                max_abs_err=max_abs,
                max_rel_err=max_rel,
                detail=f"y[{i}] = {float(gy[i]):.17g} vs reference {float(wy[i]):.17g}",
                kind="numeric",
            )

    gc = torch.as_tensor(got["cost_ns"]).to(torch.float64).reshape(-1)
    wc = torch.as_tensor(want["cost_ns"]).to(torch.float64).reshape(-1)
    if gc.shape != wc.shape:
        return CompareResult(
            ok=False,
            detail=f"cost_ns: {tuple(gc.shape)} entries != reference {tuple(wc.shape)}",
            kind="shape",
        )

    gcfg = torch.as_tensor(got["configs_used"]).to(torch.int64).reshape(-1, 3)
    wcfg = torch.as_tensor(want["configs_used"]).to(torch.int64).reshape(-1, 3)
    if gcfg.shape != wcfg.shape:
        return CompareResult(
            ok=False,
            detail=f"configs_used: {tuple(gcfg.shape)} != reference {tuple(wcfg.shape)}",
            kind="shape",
        )

    if gc.numel():
        ratio = gc / wc.clamp_min(1e-300)
        excess = float(ratio.max()) - 1.0
        worst = int(torch.argmax(ratio))
        if float(ratio.max()) > 1.0 + _COST_SLACK:
            g_row = [int(v) for v in gcfg[worst]]
            w_row = [int(v) for v in wcfg[worst]]
            return CompareResult(
                ok=False,
                max_abs_err=float((gc - wc).abs().max()),
                max_rel_err=excess,
                detail=(
                    f"stale tile config on problem {worst}: used {tuple(g_row)} where the "
                    f"shape class wants {tuple(w_row)}; modelled cost "
                    f"{float(gc[worst]):.1f} ns vs {float(wc[worst]):.1f} ns "
                    f"({excess * 100.0:.1f}% worse). Values are identical - this is "
                    f"invisible to a correctness check."
                ),
                kind="numeric",
            )
        if not bool(torch.equal(gcfg, wcfg)):
            i = int(torch.nonzero((gcfg != wcfg).any(dim=1))[0])
            return CompareResult(
                ok=False,
                max_rel_err=excess,
                detail=(
                    f"problem {i} tuned to {tuple(int(v) for v in gcfg[i])} but the shape "
                    f"class resolves to {tuple(int(v) for v in wcfg[i])} at equal modelled "
                    f"cost; the tuner is not a function of the key"
                ),
                kind="numeric",
            )

    return CompareResult(
        ok=True,
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        detail="tile choice matches the shape class on every problem; no cost excess",
        kind="numeric",
    )


# --------------------------------------------------------------------------- #
# adversarial sweep
# --------------------------------------------------------------------------- #

SHAPES: list[ShapeSpec] = [
    # The reduction length changes class while m and n do not. A key that omits
    # k cannot see this and hands problem 2 problem 0's tile.
    ShapeSpec(
        name="kclass_change_64",
        kwargs={"shapes": [[64, 64, 32], [64, 64, 32], [64, 64, 256], [64, 64, 256]]},
    ),
    # Same, one class larger, and with the classes interleaved so a cache that
    # only ever holds one entry thrashes instead of going stale.
    ShapeSpec(
        name="kclass_interleaved_128",
        kwargs={"shapes": [[128, 128, 32], [128, 128, 256], [128, 128, 32], [128, 128, 256]]},
    ),
    # The M class changes while n and k hold: a key that omits m is caught here
    # and nowhere else in the sweep.
    ShapeSpec(
        name="mclass_change",
        kwargs={"shapes": [[64, 64, 64], [64, 64, 64], [128, 64, 64]]},
    ),
    # Dimensions that are not powers of two, so every tile pads.
    ShapeSpec(
        name="ragged_dims",
        kwargs={"shapes": [[33, 33, 33], [17, 129, 5], [100, 96, 48]]},
    ),
    # Two problems in one class whose CONCRETE optima differ from the class
    # representative's. A tuner that benchmarks the arriving problem instead of
    # the class caches the small one's answer and charges the large one for it.
    ShapeSpec(
        name="class_repr_vs_concrete",
        kwargs={"shapes": [[65, 65, 65], [128, 128, 128]]},
    ),
    # One problem: nothing is ever looked up twice, so no reuse can go stale.
    # A decoy for staleness, though not for a wrong key function.
    ShapeSpec(
        name="single_problem",
        kwargs={"shapes": [[64, 64, 64]]},
    ),
    # Every problem in one class: the cached config is always the right one, so
    # a key missing a dimension is unobservable. Decoy.
    ShapeSpec(
        name="one_class_decoy",
        kwargs={"shapes": [[64, 64, 32], [64, 64, 32], [60, 61, 30]]},
    ),
    # Zero-length reduction: the k loop never runs and the output is exactly
    # zero, which is a real answer and not an error.
    ShapeSpec(
        name="zero_k",
        kwargs={"shapes": [[16, 16, 0], [16, 16, 16]]},
    ),
    # Zero rows: an empty output, and the empty-tile guard is the only reason
    # the tile loop terminates cleanly.
    ShapeSpec(
        name="zero_m",
        kwargs={"shapes": [[0, 16, 16], [16, 16, 16]]},
    ),
    # Nothing to do at all.
    ShapeSpec(
        name="empty_sequence",
        kwargs={"shapes": []},
    ),
    # bf16 accumulation: values, not cost, is what breaks here.
    ShapeSpec(
        name="bf16_accum_probe",
        kwargs={"shapes": [[64, 64, 256]], "accum_dtype": "float64", "dtype": "float32"},
    ),
]


def accum_depth(shape: ShapeSpec) -> int:
    """The longest dependent chain of adds is the reduction over k of the
    largest problem; blocking changes the order, not the depth."""
    shapes = _shapes_of(shape)
    if not shapes:
        return 1
    return max(max(int(k) for _, _, k in shapes), 1)


def bytes_moved(shape: ShapeSpec) -> int:
    """Compulsory traffic only: each operand read once and each output written
    once. The re-read traffic a tiling forces is candidate-dependent and is
    modelled inside ``_cost_ns``; counting it here would be claiming to know a
    tiling that has not been chosen yet."""
    itemsize = 4 if str(shape.kwargs.get("dtype", "float32")) == "float32" else 8
    total = 0
    for m, n, k in _shapes_of(shape):
        total += (m * k + k * n) * itemsize + m * n * 8
    return total


def flops(shape: ShapeSpec) -> int:
    """2*m*n*k per problem: one multiply and one add per MAC."""
    return sum(2 * m * n * k for m, n, k in _shapes_of(shape))


SEED: SeedSpec = register(
    SeedSpec(
        id="autotune.cached_gemm",
        domain="pytorch",
        tiers=("T4", "T5"),
        description=(
            "Blocked GEMM behind a tile-config cache keyed by a power-of-two shape "
            "class, with a modelled cost that regresses when a stale config "
            "survives a class change. Values stay identical either way."
        ),
        entry="autotuned_gemm",
        source=SOURCE,
        make_inputs=make_inputs,
        reference=reference,
        shape_sweep=list(SHAPES),
        accum_depth=accum_depth,
        bytes_moved=bytes_moved,
        flops=flops,
        denylist=(
            # Whole-problem fused matmul: the seed is about how the problem is
            # tiled, so calling the untiled op answers a different question.
            "torch.matmul",
            "torch.mm",
            "torch.bmm",
            "torch.einsum",
            "torch.addmm",
            "torch.nn.functional.linear",
            # A decorator cache hides the key function, which is the artefact
            # under test.
            "functools.lru_cache",
            "functools.cache",
            "crucible.seeds.autotune",
        ),
        compare=compare,
        supports_cpu=True,
        module=__name__,
        extras={
            "configs": DEFAULT_CONFIGS,
            "cost_is_modelled": True,
            "silent_failures": (
                "autotune_staleness: cache key drops a dimension, values identical",
                "tune_on_concrete_shape: cache stops being a pure memo",
                "accum_dtype: bf16 accumulator inside the k loop",
            ),
        },
    )
)

__all__ = [
    "SEED",
    "SOURCE",
    "DEFAULT_CONFIGS",
    "make_inputs",
    "reference",
    "compare",
    "SHAPES",
    "accum_depth",
    "bytes_moved",
    "flops",
]
