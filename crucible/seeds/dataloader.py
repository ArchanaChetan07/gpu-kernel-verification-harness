"""Seed: a deterministic sharded input pipeline.

The ground truth here is not a number, it is a **stream**: the exact multiset of
samples emitted and the exact order they come out in. That makes the failures
sharp to grade and invisible to watch - a sampler that hands every rank the same
shard, or drops the ragged tail on one rank and pads it on another, produces a
loss curve that descends perfectly normally while the model has seen a third of
the data twice and another third never.

Three sites carry the failure modes:

* an explicit empty-shard guard - with more ranks than samples, or ``drop_last``
  on a batch that is smaller than the world, a rank's shard is legitimately
  empty and must emit nothing rather than borrow from a neighbour;
* an explicit drop-or-pad decision, made once, before sharding, so that every
  rank ends up with the same number of batches (ranks with different batch
  counts desynchronise the collectives in the training loop downstream);
* a prefetch depth that determines whether the consumer stalls, modelled by an
  explicit bounded-queue simulation so the throughput consequence is a number
  and not a vibe.

The shuffle is an index-space bijection ``i -> (a*i + b) mod n`` rather than an
RNG permutation. That is a design choice with a reason: no RNG state has to
cross a process boundary for every rank to agree, a restart reproduces the order
exactly, and the ground truth can be re-derived independently - which it could
not be if the order were whatever ``torch.randperm`` happened to produce.
"""

from __future__ import annotations

import math
from typing import Any

from ..schema import ShapeSpec
from .registry import CompareResult, SeedSpec, register

# --------------------------------------------------------------------------- #
# the known-good baseline
# --------------------------------------------------------------------------- #

SOURCE = '''"""Deterministic sharded batch stream with an explicit prefetch model."""
import math

import torch


def _coprime_multiplier(n, s):
    """Smallest multiplier at or after a seeded start that is coprime to n, so
    that ``i -> a*i`` is a bijection on ``range(n)``."""
    if n <= 1:
        return 1
    a = 1 + (int(s) * 2246822519) % n
    while math.gcd(a, n) != 1:
        a += 1
        if a > n:
            a = 1
    return a


def _permutation(n, seed, epoch):
    """Index-space shuffle derived from (seed, epoch) alone.

    Every rank computes this identically without exchanging RNG state, and a
    resumed run reproduces it. Deriving it from the rank instead is how shards
    silently overlap.
    """
    if n <= 0:
        return torch.zeros(0, dtype=torch.int64)
    a = _coprime_multiplier(n, int(seed) + int(epoch))
    b = (int(seed) * 2654435761 + int(epoch) * 40503) % n
    idx = torch.arange(n, dtype=torch.int64)
    return (a * idx + b) % n


def _pad_or_drop(order, world_size, drop_last):
    """The decision, made once and globally: drop the ragged tail, or wrap the
    head of the order around to fill it. Both branches must leave every rank
    with the same shard length. Returns (order, n_padded)."""
    n = int(order.shape[0])
    if n == 0:
        return order, 0
    if drop_last:
        usable = (n // world_size) * world_size
        return order[:usable], 0
    per = (n + world_size - 1) // world_size
    total = per * world_size
    reps = (total + n - 1) // n
    return order.repeat(reps)[:total], total - n


def _batch_bounds(shard_len, batch_size, drop_last):
    """Batch boundaries within one shard. The tail batch is dropped or emitted
    short; it is never padded a second time.

    The guard is load-bearing, not decoration: ``width`` below is the divisor,
    and a shard with no samples has no width to divide by.
    """
    if shard_len <= 0:                       # empty-shard guard
        return []
    width = min(batch_size, shard_len)       # a batch is never wider than its shard
    if drop_last:
        n_batches = shard_len // batch_size
    else:
        n_batches = (shard_len + width - 1) // width
    bounds = []
    for bi in range(n_batches):
        lo = bi * batch_size
        hi = min(lo + batch_size, shard_len)
        bounds.append((lo, hi))
    return bounds


def _simulate(n_batches, prefetch_depth, producer_ms, consumer_ms):
    """Bounded-queue producer/consumer over one rank's batches.

    With depth d the producer may run d batches ahead of the consumer; batch i
    cannot start until batch i-d has been consumed and its queue slot freed.
    Depth 0 is no buffering at all: every batch is produced on demand, so the
    consumer stalls for the full producer latency every single time.

    Both sides of the depth are reported. ``stall_ms`` is the consumer waiting
    on an empty queue (depth too shallow); ``producer_idle_ms`` is the producer
    blocked on a full one (depth binding). A pipeline that ignores the depth
    entirely shows zero producer idle time and is otherwise indistinguishable.
    """
    depth = int(prefetch_depth)
    if depth < 0:
        depth = 0
    ready = [0] * n_batches
    done = [0] * n_batches
    prod_free = 0
    cons_free = 0
    stalls = 0
    stall_ms = 0
    idle_ms = 0
    for i in range(n_batches):
        gate = prod_free
        if depth == 0:
            gate = max(gate, cons_free)
        elif i >= depth:
            gate = max(gate, done[i - depth])
        idle_ms += gate - prod_free
        ready[i] = gate + producer_ms
        prod_free = ready[i]
        if ready[i] > cons_free:
            stalls += 1
            stall_ms += ready[i] - cons_free
        done[i] = max(cons_free, ready[i]) + consumer_ms
        cons_free = done[i]
    return stalls, stall_ms, cons_free, idle_ms


def _cat(parts):
    if not parts:
        return torch.zeros(0, dtype=torch.int64)
    return torch.cat(parts)


def batch_stream(n_samples, world_size, batch_size, seed=0, epochs=1,
                 drop_last=False, prefetch_depth=2, producer_ms=3,
                 consumer_ms=2):
    """Emit every rank's stream for every epoch, in (epoch, rank, batch) order."""
    n = int(n_samples)
    ws = int(world_size)
    bs = int(batch_size)
    if ws <= 0:
        raise ValueError("world_size must be positive")
    if bs <= 0:
        raise ValueError("batch_size must be positive")
    if n < 0:
        raise ValueError("n_samples must not be negative")

    order_parts = []
    rank_parts = []
    epoch_parts = []
    batch_parts = []
    counts = torch.zeros(n, dtype=torch.int64)
    n_batches = []
    stalls = []
    stall_ms = []
    makespan = []
    idle = []
    pad_total = 0

    for epoch in range(int(epochs)):
        perm = _permutation(n, seed, epoch)
        padded, padded_n = _pad_or_drop(perm, ws, bool(drop_last))
        pad_total += padded_n
        for rank in range(ws):
            shard = padded[rank::ws]                  # strided, disjoint shards
            bounds = _batch_bounds(int(shard.shape[0]), bs, bool(drop_last))
            n_batches.append(len(bounds))
            s, sm, mk, idl = _simulate(len(bounds), prefetch_depth,
                                       int(producer_ms), int(consumer_ms))
            stalls.append(s)
            stall_ms.append(sm)
            makespan.append(mk)
            idle.append(idl)
            for bi in range(len(bounds)):
                lo, hi = bounds[bi]
                chunk = shard[lo:hi]
                width = hi - lo
                order_parts.append(chunk)
                rank_parts.append(torch.full((width,), rank, dtype=torch.int64))
                epoch_parts.append(torch.full((width,), epoch, dtype=torch.int64))
                batch_parts.append(torch.full((width,), bi, dtype=torch.int64))
                counts.index_add_(0, chunk, torch.ones(width, dtype=torch.int64))

    return {
        "order": _cat(order_parts),
        "rank_of": _cat(rank_parts),
        "epoch_of": _cat(epoch_parts),
        "batch_of": _cat(batch_parts),
        "counts": counts,
        "n_batches": torch.tensor(n_batches, dtype=torch.int64),
        "stalls": torch.tensor(stalls, dtype=torch.int64),
        "stall_ms": torch.tensor(stall_ms, dtype=torch.int64),
        "makespan_ms": torch.tensor(makespan, dtype=torch.int64),
        "producer_idle_ms": torch.tensor(idle, dtype=torch.int64),
        "pad_count": torch.tensor(pad_total, dtype=torch.int64),
    }
'''


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def make_inputs(shape: ShapeSpec, device: Any = "cpu", generator: Any = None) -> dict[str, Any]:
    """Every input is a plain integer or bool: the pipeline is indexed, not
    sampled, so ``device`` and ``generator`` are accepted and unused. That is
    also why this seed is fully reproducible on any machine."""
    kw = shape.kwargs
    return {
        "n_samples": int(kw["n_samples"]),
        "world_size": int(kw["world_size"]),
        "batch_size": int(kw["batch_size"]),
        "seed": int(kw.get("seed", 0)),
        "epochs": int(kw.get("epochs", 1)),
        "drop_last": bool(kw.get("drop_last", False)),
        "prefetch_depth": int(kw.get("prefetch_depth", 2)),
        "producer_ms": int(kw.get("producer_ms", 3)),
        "consumer_ms": int(kw.get("consumer_ms", 2)),
    }


# --------------------------------------------------------------------------- #
# independent ground truth
# --------------------------------------------------------------------------- #


def _ref_permutation(n: int, seed: int, epoch: int) -> list[int]:
    if n <= 0:
        return []
    if n == 1:
        a = 1
    else:
        a = 1 + ((seed + epoch) * 2246822519) % n
        while math.gcd(a, n) != 1:
            a += 1
            if a > n:
                a = 1
    b = (seed * 2654435761 + epoch * 40503) % n
    return [(a * i + b) % n for i in range(n)]


def _ref_stalls(
    n_batches: int, depth: int, producer_ms: int, consumer_ms: int
) -> tuple[int, int, int, int]:
    """Event-driven form of the same queue model: two clocks and a map of
    consumption times, no arrays indexed by batch position."""
    depth = max(int(depth), 0)
    prod_clock = 0
    cons_clock = 0
    consumed_at: dict[int, int] = {}
    stalls = 0
    stall_ms = 0
    idle_ms = 0
    for i in range(n_batches):
        if depth == 0:
            gate = cons_clock
        elif i - depth >= 0:
            gate = consumed_at[i - depth]
        else:
            gate = 0
        idle_ms += max(prod_clock, gate) - prod_clock
        prod_clock = max(prod_clock, gate) + producer_ms
        if prod_clock > cons_clock:
            stalls += 1
            stall_ms += prod_clock - cons_clock
            cons_clock = prod_clock
        cons_clock += consumer_ms
        consumed_at[i] = cons_clock
    return stalls, stall_ms, cons_clock, idle_ms


def reference(
    n_samples: int,
    world_size: int,
    batch_size: int,
    seed: int = 0,
    epochs: int = 1,
    drop_last: bool = False,
    prefetch_depth: int = 2,
    producer_ms: int = 3,
    consumer_ms: int = 2,
) -> dict[str, Any]:
    """Ground truth for the emitted stream.

    A flat walk over the whole stream in plain python: no tensors, no strided
    views, no shared helper with the baseline. There is no fused library call
    that emits this stream, so an independent reference means an independently
    written one; the two agree only if both implement the specification.
    """
    import torch

    n = int(n_samples)
    ws = int(world_size)
    bs = int(batch_size)

    order: list[int] = []
    rank_of: list[int] = []
    epoch_of: list[int] = []
    batch_of: list[int] = []
    counts = [0] * n
    n_batches: list[int] = []
    stalls: list[int] = []
    stall_ms: list[int] = []
    makespan: list[int] = []
    idle: list[int] = []
    pad_total = 0

    for epoch in range(int(epochs)):
        perm = _ref_permutation(n, int(seed), epoch)
        if n == 0:
            padded: list[int] = []
        elif drop_last:
            padded = perm[: (n // ws) * ws]
        else:
            per = -(-n // ws)
            total = per * ws
            grown: list[int] = []
            while len(grown) < total:
                grown.extend(perm)
            padded = grown[:total]
            pad_total += total - n

        for rank in range(ws):
            shard = [padded[j] for j in range(rank, len(padded), ws)]
            if len(shard) == 0:
                n_batches.append(0)
                stalls.append(0)
                stall_ms.append(0)
                makespan.append(0)
                idle.append(0)
                continue
            if drop_last:
                nb = len(shard) // bs
            else:
                nb = -(-len(shard) // bs)
            n_batches.append(nb)
            s, sm, mk, idl = _ref_stalls(
                nb, int(prefetch_depth), int(producer_ms), int(consumer_ms)
            )
            stalls.append(s)
            stall_ms.append(sm)
            makespan.append(mk)
            idle.append(idl)
            emitted = 0
            bi = 0
            while bi < nb:
                take = min(bs, len(shard) - emitted)
                for _ in range(take):
                    sample = shard[emitted]
                    order.append(sample)
                    rank_of.append(rank)
                    epoch_of.append(epoch)
                    batch_of.append(bi)
                    counts[sample] += 1
                    emitted += 1
                bi += 1

    def ln(v: list[int]) -> Any:
        return torch.tensor(v, dtype=torch.int64)

    return {
        "order": ln(order),
        "rank_of": ln(rank_of),
        "epoch_of": ln(epoch_of),
        "batch_of": ln(batch_of),
        "counts": ln(counts),
        "n_batches": ln(n_batches),
        "stalls": ln(stalls),
        "stall_ms": ln(stall_ms),
        "makespan_ms": ln(makespan),
        "producer_idle_ms": ln(idle),
        "pad_count": torch.tensor(pad_total, dtype=torch.int64),
    }


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #

_KEYS = (
    "order",
    "rank_of",
    "epoch_of",
    "batch_of",
    "counts",
    "n_batches",
    "stalls",
    "stall_ms",
    "makespan_ms",
    "producer_idle_ms",
    "pad_count",
)


def compare(got: Any, want: Any) -> CompareResult:
    """Exact equality. There is no tolerance on a stream: a sample is either
    emitted at position i or it is not, and ``counts`` makes the multiset
    property (every sample exactly once per epoch, modulo declared padding)
    checkable independently of the order."""
    import torch

    if not isinstance(got, dict):
        return CompareResult(
            ok=False, detail=f"expected a dict of outputs, got {type(got).__name__}", kind="shape"
        )
    missing = [k for k in _KEYS if k not in got]
    if missing:
        return CompareResult(
            ok=False, detail=f"output is missing key(s): {', '.join(missing)}", kind="shape"
        )

    worst = 0.0
    for key in _KEYS:
        g = torch.as_tensor(got[key]).to(torch.int64).reshape(-1)
        w = torch.as_tensor(want[key]).to(torch.int64).reshape(-1)
        if g.shape != w.shape:
            return CompareResult(
                ok=False,
                max_abs_err=float(abs(g.numel() - w.numel())),
                detail=(
                    f"{key}: emitted {g.numel()} entries, reference emits {w.numel()}"
                ),
                kind="shape",
            )
        if g.numel() == 0:
            continue
        diff = (g - w).abs()
        worst = max(worst, float(diff.max()))
        bad = torch.nonzero(diff)
        if bad.numel():
            i = int(bad[0])
            return CompareResult(
                ok=False,
                max_abs_err=float(diff.max()),
                max_rel_err=float(diff.max()),
                detail=(
                    f"{key}[{i}] = {int(g[i])}, reference says {int(w[i])} "
                    f"({int(torch.count_nonzero(diff))} of {g.numel()} entries differ)"
                ),
                kind="numeric",
            )
    return CompareResult(ok=True, max_abs_err=worst, detail="stream matches exactly", kind="numeric")


# --------------------------------------------------------------------------- #
# adversarial sweep
# --------------------------------------------------------------------------- #

SHAPES: list[ShapeSpec] = [
    # Exactly divisible everywhere: the happy path, and therefore a decoy - the
    # pad/drop decision is unobservable when there is no ragged tail.
    ShapeSpec(
        name="even_64x4x16",
        kwargs={"n_samples": 64, "world_size": 4, "batch_size": 16, "drop_last": True},
    ),
    # Sample count not divisible by the world size; tail is padded by wrapping.
    ShapeSpec(
        name="ragged_100x3_pad",
        kwargs={"n_samples": 100, "world_size": 3, "batch_size": 8, "drop_last": False},
    ),
    # Same shape, dropped instead of padded: the two must differ, and both must
    # leave every rank with the same batch count.
    ShapeSpec(
        name="ragged_100x3_drop",
        kwargs={"n_samples": 100, "world_size": 3, "batch_size": 8, "drop_last": True},
    ),
    # Batch count per rank not divisible by the world size, short tail batch.
    ShapeSpec(
        name="ragged_45x4_shorttail",
        kwargs={"n_samples": 45, "world_size": 4, "batch_size": 4, "drop_last": False},
    ),
    # More ranks than samples: with drop_last every shard is legitimately empty.
    ShapeSpec(
        name="empty_shards_2x4",
        kwargs={"n_samples": 2, "world_size": 4, "batch_size": 2, "drop_last": True},
    ),
    # Same, padded: every rank gets one duplicated sample.
    ShapeSpec(
        name="pad_from_two_2x4",
        kwargs={"n_samples": 2, "world_size": 4, "batch_size": 2, "drop_last": False},
    ),
    # Nothing at all to emit.
    ShapeSpec(
        name="zero_samples",
        kwargs={"n_samples": 0, "world_size": 3, "batch_size": 4, "drop_last": False},
    ),
    # No prefetch: the consumer stalls on every single batch.
    ShapeSpec(
        name="prefetch0_stall_every",
        kwargs={
            "n_samples": 48,
            "world_size": 2,
            "batch_size": 4,
            "drop_last": True,
            "prefetch_depth": 0,
            "producer_ms": 7,
            "consumer_ms": 2,
        },
    ),
    # Deep queue and a fast producer: the consumer never waits after the first,
    # and the depth never binds, so the queue bound is invisible here.
    ShapeSpec(
        name="prefetch8_no_stall",
        kwargs={
            "n_samples": 48,
            "world_size": 2,
            "batch_size": 4,
            "drop_last": True,
            "prefetch_depth": 8,
            "producer_ms": 1,
            "consumer_ms": 5,
        },
    ),
    # Same fast producer, shallow queue: now the depth binds and the producer
    # blocks on backpressure. This is the only shape where ignoring the queue
    # bound is observable at all.
    ShapeSpec(
        name="prefetch2_backpressure",
        kwargs={
            "n_samples": 48,
            "world_size": 2,
            "batch_size": 4,
            "drop_last": True,
            "prefetch_depth": 2,
            "producer_ms": 1,
            "consumer_ms": 5,
        },
    ),
    # Two epochs: the permutation must change and the multiset must not.
    ShapeSpec(
        name="two_epochs_35x3",
        kwargs={
            "n_samples": 35,
            "world_size": 3,
            "batch_size": 6,
            "epochs": 2,
            "seed": 17,
            "drop_last": False,
        },
    ),
    # Batch larger than the whole shard.
    ShapeSpec(
        name="batch_gt_shard",
        kwargs={"n_samples": 9, "world_size": 2, "batch_size": 32, "drop_last": False},
    ),
]


def _emitted_per_rank(shape: ShapeSpec) -> int:
    kw = shape.kwargs
    n = int(kw["n_samples"])
    ws = int(kw["world_size"])
    bs = int(kw["batch_size"])
    if n == 0:
        return 0
    if bool(kw.get("drop_last", False)):
        shard = ((n // ws) * ws) // ws
        return (shard // bs) * bs
    return -(-n // ws)


def accum_depth(shape: ShapeSpec) -> int:
    """One. Every value in this pipeline is an exact int64 index; there is no
    floating-point accumulation to lose precision in, so a tolerance derived
    from an accumulation depth would be meaningless here and the comparison is
    exact instead."""
    return 1


def bytes_moved(shape: ShapeSpec) -> int:
    """int64 indices, read once from the permuted order and written once into a
    batch, per rank per epoch. Sample payloads are not modelled: this seed moves
    indices, and claiming payload bytes it never touches would be a fabricated
    number."""
    kw = shape.kwargs
    emitted = _emitted_per_rank(shape) * int(kw["world_size"]) * int(kw.get("epochs", 1))
    return emitted * 8 * 2


def flops(shape: ShapeSpec) -> int:
    """Integer index arithmetic only: one multiply, one add and one modulo per
    sample per epoch to build the permutation."""
    kw = shape.kwargs
    return 3 * int(kw["n_samples"]) * int(kw.get("epochs", 1))


SEED: SeedSpec = register(
    SeedSpec(
        id="dataloader.sharded_batch_stream",
        domain="data_pipeline",
        tiers=("T2", "T3", "T4"),
        description=(
            "Deterministic sharded sampler with an explicit empty-shard guard, an "
            "explicit drop_last/pad decision and a prefetch depth that determines "
            "whether the consumer stalls."
        ),
        entry="batch_stream",
        source=SOURCE,
        make_inputs=make_inputs,
        reference=reference,
        shape_sweep=list(SHAPES),
        accum_depth=accum_depth,
        bytes_moved=bytes_moved,
        flops=flops,
        denylist=(
            # Ready-made samplers implement (a different) specification for you.
            "torch.utils.data.DistributedSampler",
            "torch.utils.data.DataLoader",
            "torch.utils.data.RandomSampler",
            # An RNG permutation cannot be re-derived by a peer rank or a resume.
            "torch.randperm",
            "random.shuffle",
            "random.sample",
            "numpy.random",
            "crucible.seeds.dataloader",
        ),
        compare=compare,
        supports_cpu=True,
        module=__name__,
        extras={
            "exact": True,
            "silent_failures": (
                "shard overlap: every rank sees the same samples, loss still descends",
                "pad/drop split-brain: ranks end an epoch with different batch counts",
                "prefetch depth ignored: correct stream, stalled consumer",
            ),
        },
    )
)

__all__ = [
    "SEED",
    "SOURCE",
    "make_inputs",
    "reference",
    "compare",
    "SHAPES",
    "accum_depth",
    "bytes_moved",
    "flops",
]
