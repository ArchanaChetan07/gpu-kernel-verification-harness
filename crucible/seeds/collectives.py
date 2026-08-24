"""Seed: a data-parallel training step with explicit collectives.

The three failure modes this seed exists to express are all *silent*: the loss
curve still descends, the gradients still have plausible magnitudes, and nothing
raises. They are only visible against an independent ground truth.

* ``grad_norm_scope`` (T6) - under FSDP each rank owns a slice of the gradient,
  so the clipping coefficient must come from ``sqrt`` of the **all-reduced** sum
  of squares. Taking ``sqrt`` per shard gives every shard a different, larger
  coefficient; the model still trains, just not the model you asked for.
* ``collective_ordering`` (T3) - the all-reduce is a statement at function-body
  level, deliberately outside every conditional. A rank that reaches a
  collective its peers do not is a hang in a real process group; the simulator
  here reports it as a desync instead of quietly summing the wrong subset.
* the data-parallel loss reduction is a sum of per-rank **sums** over the global
  sample count. A mean of per-rank means is identical when the shards are even
  and wrong the moment they are not - which is why the sweep contains uneven
  shards, and why the happy-path shape is a decoy.

The baseline runs single-process over a list of shards so O1 can differential
test on CPU with no process group at all. ``extras["gloo_source"]`` carries the
same arithmetic written against ``torch.distributed`` for O5 to launch under
gloo; both are checked against the same reference.
"""

from __future__ import annotations

from typing import Any

from ..schema import ShapeSpec
from .registry import CompareResult, SeedSpec, register

# --------------------------------------------------------------------------- #
# the known-good baseline (text: this is what gets mutated and shipped)
# --------------------------------------------------------------------------- #

SOURCE = '''"""Single-process simulation of one data-parallel + FSDP training step."""
import torch


def _all_reduce_sum(parts, world_size):
    """SUM all-reduce over ``world_size`` ranks.

    Every rank must contribute exactly once. A rank that skips the collective
    deadlocks a real process group, so a participant count that does not match
    the world size is raised here rather than summed over whoever showed up.
    """
    if len(parts) != world_size:
        raise RuntimeError(
            "all-reduce desync: %d participants for world_size %d"
            % (len(parts), world_size)
        )
    total = parts[0].clone()
    for p in parts[1:]:
        total = total + p
    return total


def _split_shards(x, y, shard_sizes):
    """Contiguous data-parallel split. A shard may legitimately be empty."""
    shards = []
    off = 0
    for n in shard_sizes:
        n = int(n)
        shards.append((x[off:off + n], y[off:off + n]))
        off += n
    return shards


def _shard_flat(flat, n_shards):
    """Contiguous parameter shards, FSDP style. The tail shard may be short and
    a shard may be empty when there are more ranks than elements."""
    numel = int(flat.shape[0])
    per = (numel + n_shards - 1) // n_shards
    out = []
    for r in range(n_shards):
        lo = min(r * per, numel)
        hi = min(lo + per, numel)
        out.append(flat[lo:hi])
    return out


def _local_grad_and_loss(xs, ys, w):
    """Cross-entropy gradient and loss SUM for one shard, computed by hand."""
    logits = xs @ w
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp = torch.exp(shifted)
    probs = exp / exp.sum(dim=1, keepdim=True)
    picked = probs.gather(1, ys.view(-1, 1)).clamp_min(1e-30)
    loss_sum = -torch.log(picked).sum()
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, ys.view(-1, 1), 1.0)
    grad_sum = xs.transpose(0, 1) @ (probs - onehot)
    return grad_sum, loss_sum


def train_step(x, y, w0, shard_sizes, steps=8, lr=0.5, max_grad_norm=0.05,
               n_param_shards=2, accum_dtype="float32"):
    """Run ``steps`` DDP steps over a simulated ``len(shard_sizes)``-rank group."""
    acc = getattr(torch, accum_dtype)          # explicit accumulator dtype
    world_size = len(shard_sizes)
    total_n = 0
    for n in shard_sizes:
        total_n += int(n)
    if total_n <= 0:
        raise ValueError("global batch is empty; every shard has zero samples")
    xf = x.to(acc)
    w = w0.to(acc).clone()

    losses = []
    norms = []
    for _ in range(steps):
        grad_parts = []
        loss_parts = []
        for xs, ys in _split_shards(xf, y, shard_sizes):
            if int(xs.shape[0]) == 0:                    # empty-shard guard
                g = torch.zeros_like(w)
                l = torch.zeros((), dtype=acc)
            else:
                g, l = _local_grad_and_loss(xs, ys, w)
            # Outside the guard on purpose: an empty rank still participates.
            grad_parts.append(g)
            loss_parts.append(l)

        # Sums of per-rank SUMS over the GLOBAL count. A mean of per-rank means
        # silently reweights the shards as soon as they are uneven.
        grad = _all_reduce_sum(grad_parts, world_size) / total_n
        loss = _all_reduce_sum(loss_parts, world_size) / total_n

        # GLOBAL grad norm: each parameter shard contributes only its local sum
        # of squares; the all-reduce happens BEFORE the sqrt and the sqrt is
        # taken exactly once, over the total.
        param_shards = _shard_flat(grad.reshape(-1), n_param_shards)
        local_sq = []
        for s in param_shards:
            local_sq.append((s * s).sum())
        total_sq = _all_reduce_sum(local_sq, n_param_shards)
        global_norm = torch.sqrt(total_sq)

        if global_norm > max_grad_norm:
            grad = grad * (max_grad_norm / (global_norm + 1e-6))

        w = w - lr * grad
        losses.append(loss)
        norms.append(global_norm)

    return {
        "losses": torch.stack(losses).to(torch.float32),
        "grad_norms": torch.stack(norms).to(torch.float32),
        "w": w.to(torch.float32),
    }
'''

# --------------------------------------------------------------------------- #
# the same arithmetic under a real gloo process group (O5 launches this)
# --------------------------------------------------------------------------- #

GLOO_SOURCE = '''"""Multi-process form of the collectives seed. gloo only: NCCL is not
available on Windows and this must run identically on both machines.

``run_rank`` is importable at module scope so torch.multiprocessing.spawn can
reach it on a platform without fork.
"""

import os
from pathlib import Path

import torch
import torch.distributed as dist


def _local_grad_and_loss(xs, ys, w):
    logits = xs @ w
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp = torch.exp(shifted)
    probs = exp / exp.sum(dim=1, keepdim=True)
    picked = probs.gather(1, ys.view(-1, 1)).clamp_min(1e-30)
    loss_sum = -torch.log(picked).sum()
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, ys.view(-1, 1), 1.0)
    return xs.transpose(0, 1) @ (probs - onehot), loss_sum


def _shard_bounds(shard_sizes, rank):
    off = 0
    for r in range(rank):
        off += int(shard_sizes[r])
    return off, off + int(shard_sizes[rank])


def _flat_bounds(numel, n_shards, rank):
    per = (numel + n_shards - 1) // n_shards
    lo = min(rank * per, numel)
    return lo, min(lo + per, numel)


def train_step_dist(x, y, w0, shard_sizes, steps=8, lr=0.5, max_grad_norm=0.05,
                    accum_dtype="float32"):
    """One rank's view of the step. The process group must already be built."""
    acc = getattr(torch, accum_dtype)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total_n = 0
    for n in shard_sizes:
        total_n += int(n)
    lo, hi = _shard_bounds(shard_sizes, rank)
    xs_all = x.to(acc)[lo:hi]
    ys_all = y[lo:hi]
    w = w0.to(acc).clone()

    losses = []
    norms = []
    for _ in range(steps):
        if int(xs_all.shape[0]) == 0:                     # empty-shard guard
            g = torch.zeros_like(w)
            l = torch.zeros((), dtype=acc)
        else:
            g, l = _local_grad_and_loss(xs_all, ys_all, w)
        # Unconditional: an empty rank still enters both collectives.
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        dist.all_reduce(l, op=dist.ReduceOp.SUM)
        grad = g / total_n
        loss = l / total_n

        flat = grad.reshape(-1)
        plo, phi = _flat_bounds(int(flat.shape[0]), world_size, rank)
        my_shard = flat[plo:phi]
        local_sq = (my_shard * my_shard).sum()
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)   # BEFORE the sqrt
        global_norm = torch.sqrt(local_sq)

        if global_norm > max_grad_norm:
            grad = grad * (max_grad_norm / (global_norm + 1e-6))
        w = w - lr * grad
        losses.append(loss)
        norms.append(global_norm)

    return {
        "losses": torch.stack(losses).to(torch.float32),
        "grad_norms": torch.stack(norms).to(torch.float32),
        "w": w.to(torch.float32),
    }


def run_rank(rank, world_size, store_dir, payload):
    """Entry point for torch.multiprocessing.spawn. Returns nothing; writes a
    per-rank ``.pt``-free JSON summary so the parent never unpickles a tensor."""
    import json

    store_dir = Path(store_dir)
    store = dist.FileStore(str(store_dir / "store"), world_size)
    dist.init_process_group(backend="gloo", store=store, rank=rank,
                            world_size=world_size)
    try:
        x = torch.tensor(payload["x"], dtype=torch.float32)
        y = torch.tensor(payload["y"], dtype=torch.int64)
        w0 = torch.tensor(payload["w0"], dtype=torch.float32)
        out = train_step_dist(
            x, y, w0, payload["shard_sizes"],
            steps=int(payload.get("steps", 8)),
            lr=float(payload.get("lr", 0.5)),
            max_grad_norm=float(payload.get("max_grad_norm", 0.05)),
            accum_dtype=str(payload.get("accum_dtype", "float32")),
        )
        summary = {
            "rank": rank,
            "losses": [float(v) for v in out["losses"]],
            "grad_norms": [float(v) for v in out["grad_norms"]],
            "w": [float(v) for v in out["w"].reshape(-1)],
        }
        (store_dir / ("rank%d.json" % rank)).write_text(
            json.dumps(summary), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run_rank(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]),
             os.environ["CRUCIBLE_STORE_DIR"], {})
'''


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def make_inputs(shape: ShapeSpec, device: Any = "cpu", generator: Any = None) -> dict[str, Any]:
    """Pure function of (shape, device, generator); no global RNG is touched."""
    import torch

    kw = shape.kwargs
    shard_sizes = [int(v) for v in kw["shard_sizes"]]
    n = sum(shard_sizes)
    d = int(kw["d"])
    k = int(kw["k"])
    x = torch.randn(n, d, generator=generator, device=device, dtype=torch.float32)
    y = torch.randint(0, k, (n,), generator=generator, device=device, dtype=torch.int64)
    w0 = 0.1 * torch.randn(d, k, generator=generator, device=device, dtype=torch.float32)
    return {
        "x": x,
        "y": y,
        "w0": w0,
        "shard_sizes": shard_sizes,
        "steps": int(kw.get("steps", 8)),
        "lr": float(kw.get("lr", 0.5)),
        "max_grad_norm": float(kw.get("max_grad_norm", 0.05)),
        "n_param_shards": int(kw.get("n_param_shards", 2)),
        "accum_dtype": str(kw.get("accum_dtype", "float32")),
    }


# --------------------------------------------------------------------------- #
# independent ground truth
# --------------------------------------------------------------------------- #


def reference(
    x: Any,
    y: Any,
    w0: Any,
    shard_sizes: Any,
    steps: int = 8,
    lr: float = 0.5,
    max_grad_norm: float = 0.05,
    n_param_shards: int = 2,
    accum_dtype: str = "float32",
) -> dict[str, Any]:
    """Ground truth: autograd on the whole global batch, clipped by torch's own
    ``clip_grad_norm_``.

    Nothing here is sharded, and that is the point: a correct sharded step must
    be indistinguishable from the unsharded one. ``shard_sizes``,
    ``n_param_shards`` and ``accum_dtype`` are accepted and ignored - topology
    is exactly the thing the result must not depend on. fp64 throughout so the
    reference itself is not a source of drift.
    """
    import torch
    import torch.nn.functional as F

    xf = torch.as_tensor(x).to(torch.float64)
    yl = torch.as_tensor(y).to(torch.int64)
    w = torch.as_tensor(w0).to(torch.float64).clone()

    losses: list[float] = []
    norms: list[float] = []
    for _ in range(int(steps)):
        p = torch.nn.Parameter(w.clone())
        loss = F.cross_entropy(xf @ p, yl)          # mean over the GLOBAL batch
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_([p], float(max_grad_norm))
        w = w - float(lr) * p.grad.detach()
        losses.append(float(loss.detach()))
        norms.append(float(total_norm))

    return {
        "losses": torch.tensor(losses, dtype=torch.float32),
        "grad_norms": torch.tensor(norms, dtype=torch.float32),
        "w": w.to(torch.float32),
    }


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #

_KEYS = ("losses", "grad_norms", "w")
_RTOL = 2.0e-4
_ATOL = 2.0e-5


def compare(got: Any, want: Any) -> CompareResult:
    """Elementwise on the loss curve, the norm curve and the final weights.

    The tolerance is ~100x above the fp32-vs-fp64 noise floor measured on this
    seed and ~100x below the drift a per-shard grad norm produces, so a pass is
    a real pass and a fail names the step it first went wrong.
    """
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

    worst_abs = 0.0
    worst_rel = 0.0
    detail = ""
    ok = True
    for key in _KEYS:
        g = torch.as_tensor(got[key]).to(torch.float64).reshape(-1)
        w = torch.as_tensor(want[key]).to(torch.float64).reshape(-1)
        if g.shape != w.shape:
            return CompareResult(
                ok=False,
                detail=f"{key}: shape {tuple(g.shape)} != reference {tuple(w.shape)}",
                kind="shape",
            )
        if g.numel() == 0:
            continue
        abs_err = (g - w).abs()
        rel_err = abs_err / w.abs().clamp_min(1e-12)
        budget = _ATOL + _RTOL * w.abs()
        bad = abs_err > budget
        worst_abs = max(worst_abs, float(abs_err.max()))
        worst_rel = max(worst_rel, float(rel_err.max()))
        if bool(bad.any()) and ok:
            ok = False
            i = int(torch.nonzero(bad)[0])
            detail = (
                f"{key}[{i}] = {float(g[i]):.9g} vs reference {float(w[i]):.9g} "
                f"(abs {float(abs_err[i]):.3g} > budget {float(budget[i]):.3g})"
            )
    if ok:
        detail = f"max_abs={worst_abs:.3g} max_rel={worst_rel:.3g} within rtol={_RTOL:g}"
    return CompareResult(
        ok=ok, max_abs_err=worst_abs, max_rel_err=worst_rel, detail=detail, kind="loss_curve"
    )


# --------------------------------------------------------------------------- #
# adversarial sweep
# --------------------------------------------------------------------------- #

SHAPES: list[ShapeSpec] = [
    # Even shards, clipping active: the honest happy path, and a decoy for the
    # loss-reduction bug (mean-of-means == global mean when shards are even).
    ShapeSpec(
        name="dp2_even_clip",
        kwargs={"shard_sizes": [16, 16], "d": 8, "k": 4, "steps": 8, "n_param_shards": 2},
    ),
    # Uneven shards: the loss reduction and the gradient reduction now disagree
    # with any per-rank averaging.
    ShapeSpec(
        name="dp3_uneven",
        kwargs={"shard_sizes": [21, 7, 4], "d": 8, "k": 4, "steps": 8, "n_param_shards": 2},
    ),
    # A zero-length data shard sitting between two populated ones.
    ShapeSpec(
        name="dp3_empty_middle",
        kwargs={"shard_sizes": [13, 0, 9], "d": 8, "k": 4, "steps": 8, "n_param_shards": 2},
    ),
    # More parameter shards than the grad norm reduction is comfortable with,
    # and a param count not divisible by the shard count.
    ShapeSpec(
        name="fsdp5_ragged_params",
        kwargs={"shard_sizes": [10, 6, 6], "d": 7, "k": 3, "steps": 8, "n_param_shards": 5},
    ),
    # More parameter shards than parameters: some shards are empty.
    ShapeSpec(
        name="fsdp_more_ranks_than_params",
        kwargs={"shard_sizes": [8, 8], "d": 2, "k": 2, "steps": 6, "n_param_shards": 9},
    ),
    # Clipping never fires, so a wrong grad norm cannot change the trajectory.
    # The scope bug survives here in the *reported* norm only - a much weaker
    # signal, and the reason a task graded on the loss curve alone would miss it.
    ShapeSpec(
        name="dp2_noclip_normonly",
        kwargs={
            "shard_sizes": [12, 12],
            "d": 8,
            "k": 4,
            "steps": 6,
            "n_param_shards": 2,
            "max_grad_norm": 1000.0,
        },
    ),
    # Single rank: every collective is the identity, so nothing distributed can
    # be detected here at all.
    ShapeSpec(
        name="dp1_single_decoy",
        kwargs={"shard_sizes": [24], "d": 8, "k": 4, "steps": 6, "n_param_shards": 1},
    ),
]


def _numel(shape: ShapeSpec) -> int:
    return int(shape.kwargs["d"]) * int(shape.kwargs["k"])


def _n_samples(shape: ShapeSpec) -> int:
    return sum(int(v) for v in shape.kwargs["shard_sizes"])


def accum_depth(shape: ShapeSpec) -> int:
    """Longest chain of dependent float adds.

    Per step: the sample reduction inside a shard (worst shard), then the
    all-reduce over ranks, then the parameter-shard reduction and its
    all-reduce. Steps chain through ``w``, so they multiply.
    """
    kw = shape.kwargs
    sizes = [int(v) for v in kw["shard_sizes"]]
    per_step = max(sizes) + len(sizes) + _numel(shape) + int(kw.get("n_param_shards", 2))
    return int(kw.get("steps", 8)) * per_step


def bytes_moved(shape: ShapeSpec) -> int:
    """fp32 traffic per step: read x and w, write the gradient and w."""
    steps = int(shape.kwargs.get("steps", 8))
    n = _n_samples(shape)
    d = int(shape.kwargs["d"])
    p = _numel(shape)
    per_step = (n * d + n + 3 * p) * 4
    return steps * per_step


def flops(shape: ShapeSpec) -> int:
    """Forward (x @ w) and backward (x^T @ dlogits) are each 2*n*d*k."""
    steps = int(shape.kwargs.get("steps", 8))
    n = _n_samples(shape)
    d = int(shape.kwargs["d"])
    k = int(shape.kwargs["k"])
    return steps * (4 * n * d * k + 6 * n * k)


SEED: SeedSpec = register(
    SeedSpec(
        id="collectives.ddp_train_step",
        domain="distributed",
        tiers=("T2", "T3", "T6"),
        description=(
            "DDP/FSDP training step with an explicit gradient all-reduce, a global "
            "grad norm reduced before the sqrt, a global loss reduction and an "
            "unconditional collective."
        ),
        entry="train_step",
        source=SOURCE,
        make_inputs=make_inputs,
        reference=reference,
        shape_sweep=list(SHAPES),
        accum_depth=accum_depth,
        bytes_moved=bytes_moved,
        flops=flops,
        denylist=(
            # The ground truth's own tools. A candidate that reaches for them is
            # not implementing the sharded step, it is calling the answer.
            "torch.nn.utils.clip_grad_norm_",
            "torch.nn.utils.clip_grad_value_",
            "torch.autograd.grad",
            "torch.autograd.backward",
            "torch.nn.functional.cross_entropy",
            "torch.nn.functional.nll_loss",
            "torch.nn.functional.log_softmax",
            "torch.nn.CrossEntropyLoss",
            "torch.optim",
            "crucible.seeds.collectives",
        ),
        compare=compare,
        supports_cpu=True,
        module=__name__,
        extras={
            "gloo_source": GLOO_SOURCE,
            "gloo_entry": "run_rank",
            "gloo_step_entry": "train_step_dist",
            "gloo_world_sizes": (2, 4),
            "silent_failures": (
                "grad_norm_scope: per-shard sqrt, loss curve still descends",
                "collective_ordering: all-reduce moved under a conditional",
                "dp_loss_reduction: mean of per-rank means on uneven shards",
            ),
        },
    )
)

__all__ = [
    "SEED",
    "SOURCE",
    "GLOO_SOURCE",
    "make_inputs",
    "reference",
    "compare",
    "SHAPES",
    "accum_depth",
    "bytes_moved",
    "flops",
]
