"""Seed: save, reshard, resume - and prove the trajectory survived.

This is the T6 seed. Its failure mode is the one that costs the most and shows
up the latest: optimizer moments written at one precision and read back at
another. An ``exp_avg_sq`` round-tripped through bf16 loses about eight mantissa
bits, which perturbs Adam's denominator by ~0.4%. Nothing raises, the resumed
loss curve descends exactly as smoothly as the uninterrupted one, and the two
runs are still visibly the same run for hundreds of steps. By the time the gap
is obvious in a training dashboard the checkpoint that caused it is a thousand
steps in the past.

So the ground truth is not "the loss looks reasonable". It is: **run K more
steps and land on the same trajectory an uninterrupted run would have reached**,
compared against ``torch.optim.Adam`` with a tolerance tight enough that a
single bf16 round trip is unambiguous. Everything computes in fp64 for exactly
that reason - the checkpoint round trip is then the only place in the pipeline
where a bit can be lost, so a discrepancy has one possible cause instead of ten.

Three explicit sites carry the mutations:

* ``SAVE_DTYPE`` / ``LOAD_DTYPE`` - the precision at each end, named at module
  level rather than buried in a ``.to()``;
* ``_restore_moment`` - the one place the optimizer moments are widened back to
  compute precision, spelled as a literal ``torch.float64`` so that the read
  precision for optimizer state is a single visible decision. Narrowing exactly
  that call is the ``resume_fidelity`` defect, and narrowing it there touches
  the moments and *only* the moments - the weights come back through their own
  path, so a witness proves something about optimizer state rather than about
  the checkpoint in general;
* ``_reshard`` - a real TP/PP topology change. The bytes move: shards are
  gathered into the full flat tensor and split again for the new topology, so a
  wrong gather order or a wrong pipeline-stage assignment corrupts state rather
  than relabelling it.

``extras["gloo_source"]`` carries the same claim written for N ranks, so O5 has
a real multi-process program to run: each rank owns a shard of the flat
optimizer state, the shards are resharded onto a different rank layout across
the resume, and the resumed trajectory must not be a function of the world size.
"""

from __future__ import annotations

from typing import Any

from ..schema import ShapeSpec
from .registry import CompareResult, SeedSpec, register

# --------------------------------------------------------------------------- #
# the known-good baseline
# --------------------------------------------------------------------------- #

SOURCE = '''"""Train, checkpoint, reshard across a topology change, resume."""
import torch

# The precision at each end of the checkpoint, named at module level so it is
# impossible to change one end without seeing the other. Weights and optimizer
# moments are separable on purpose: shipping bf16 weights alongside fp32 moments
# is a real configuration, and so is the mistake of doing it the other way
# round. exp_avg_sq feeds a square root and then a division, so a relative error
# there lands straight on the update direction.
COMPUTE_DTYPE = "float64"
PARAM_SAVE_DTYPE = "float64"
STATE_SAVE_DTYPE = "float64"
STATE_LOAD_DTYPE = "float64"

FIELDS = ("param", "exp_avg", "exp_avg_sq")


def _save_dtype(field):
    """Write precision for one checkpoint field."""
    if field == "param":
        return getattr(torch, PARAM_SAVE_DTYPE)
    return getattr(torch, STATE_SAVE_DTYPE)


def _load_dtype(field):
    """Read precision for one checkpoint field, before widening to compute."""
    if field == "param":
        return getattr(torch, COMPUTE_DTYPE)
    return getattr(torch, STATE_LOAD_DTYPE)


def _restore_moment(flat, shape):
    """Widen one checkpointed optimizer moment back to compute precision.

    Written as a literal dtype instead of the module constant on purpose: the
    read precision for optimizer state is the single decision this whole seed is
    about, so it lives in one call that can be read - and got wrong - on its own.
    Narrow it and exp_avg_sq comes back with about eight mantissa bits missing;
    Adam's denominator is then off by a fraction of a percent from the first
    post-resume step onward, every later moment inherits the error through the
    exponential average, and the loss curve never stops looking smooth.

    Only the moments come through here. The weights are widened on their own
    path so that a defect introduced at this call is a statement about optimizer
    state and not about the checkpoint in general.
    """
    return flat.to(torch.float64).reshape(shape)


def _series(vals):
    """Stack a list of scalar tensors into an fp64 series.

    Module level rather than nested in the entry point so that an empty series -
    which happens legitimately when a shape resumes at step 0 - cannot be
    mistaken for a place where a dtype decision matters. Nothing is rounded here
    that was not already rounded upstream.
    """
    if not vals:
        return torch.zeros(0, dtype=getattr(torch, COMPUTE_DTYPE))
    return torch.stack(vals).to(getattr(torch, COMPUTE_DTYPE))


def _shard_flat(flat, n_shards):
    """Contiguous tensor-parallel shards of a flat tensor. The tail shard may be
    short, and with more ranks than elements a shard is legitimately empty."""
    numel = int(flat.shape[0])
    per = (numel + n_shards - 1) // n_shards
    pieces = []
    for r in range(n_shards):
        lo = min(r * per, numel)
        hi = min(lo + per, numel)
        pieces.append(flat[lo:hi])
    return pieces


def _stage_of(index, n_layers, n_stages):
    """Contiguous pipeline split. A stage may end up owning no layer at all."""
    return (index * n_stages) // n_layers


def _forward(x, w1, w2, y):
    """Two-layer tanh MLP with a cross-entropy head, plus its exact gradients."""
    z = x @ w1
    h = torch.tanh(z)
    logits = h @ w2
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp = torch.exp(shifted)
    probs = exp / exp.sum(dim=1, keepdim=True)
    n = x.shape[0]
    picked = probs.gather(1, y.view(-1, 1)).clamp_min(1e-300)
    loss = -torch.log(picked).mean()
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, y.view(-1, 1), 1.0)
    dlogits = (probs - onehot) / n
    gw2 = h.transpose(0, 1) @ dlogits
    dh = dlogits @ w2.transpose(0, 1)
    dz = dh * (1.0 - h * h)
    gw1 = x.transpose(0, 1) @ dz
    return loss, {"w1": gw1, "w2": gw2}


def _adam_step(params, grads, moments, step, lr, beta1, beta2, eps):
    """One Adam step, written out so exp_avg and exp_avg_sq are real state that
    a checkpoint has to carry rather than something an optimizer object hides."""
    bias1 = 1.0 - beta1 ** step
    bias2 = 1.0 - beta2 ** step
    step_size = lr / bias1
    bias2_sqrt = bias2 ** 0.5
    for name in params:
        g = grads[name]
        m, v = moments[name]
        m = m + (1.0 - beta1) * (g - m)
        v = v * beta2 + (1.0 - beta2) * g * g
        denom = torch.sqrt(v) / bias2_sqrt + eps
        params[name] = params[name] - step_size * m / denom
        moments[name] = (m, v)
    return params, moments


def _save(params, moments, step, tp, pp):
    """Write the sharded checkpoint, each field at its own declared precision."""
    names = list(params.keys())
    shards = {}
    for i in range(len(names)):
        name = names[i]
        stage = _stage_of(i, len(names), pp)
        m, v = moments[name]
        flat = {
            "param": params[name].reshape(-1),
            "exp_avg": m.reshape(-1),
            "exp_avg_sq": v.reshape(-1),
        }
        numel = int(flat["param"].shape[0])
        shape = tuple(int(d) for d in params[name].shape)
        split = {}
        for field in FIELDS:
            split[field] = _shard_flat(flat[field], tp)
        for r in range(tp):
            bucket = shards.setdefault((stage, r), {})
            entry = {"numel": numel, "shape": shape}
            for field in FIELDS:
                entry[field] = split[field][r].to(_save_dtype(field)).clone()
            bucket[name] = entry
    return {"shards": shards, "tp": tp, "pp": pp, "step": step, "names": names}


def _gather(ckpt, index, name, field):
    """Reassemble one flat tensor from its tensor-parallel shards, in rank
    order, and trim the padding the split introduced."""
    stage = _stage_of(index, len(ckpt["names"]), ckpt["pp"])
    pieces = []
    for r in range(ckpt["tp"]):
        pieces.append(ckpt["shards"][(stage, r)][name][field])
    numel = ckpt["shards"][(stage, 0)][name]["numel"]
    if not pieces:
        return torch.zeros(0, dtype=getattr(torch, COMPUTE_DTYPE))
    return torch.cat(pieces)[:numel]


def _reshard(ckpt, tp_after, pp_after):
    """Rebuild the checkpoint under a new TP/PP topology.

    A topology change is not a relabelling: every shard is gathered into the
    full flat tensor and split again against the new rank count, and the
    pipeline stage each parameter lives on is recomputed. Nothing here may
    depend on the old topology dividing the new one.
    """
    names = ckpt["names"]
    new_shards = {}
    for i in range(len(names)):
        name = names[i]
        old_stage = _stage_of(i, len(names), ckpt["pp"])
        new_stage = _stage_of(i, len(names), pp_after)
        meta = ckpt["shards"][(old_stage, 0)][name]
        gathered = {}
        for field in FIELDS:
            gathered[field] = _gather(ckpt, i, name, field)
        split = {}
        for field in FIELDS:
            split[field] = _shard_flat(gathered[field], tp_after)
        for r in range(tp_after):
            bucket = new_shards.setdefault((new_stage, r), {})
            entry = {"numel": meta["numel"], "shape": meta["shape"]}
            for field in FIELDS:
                entry[field] = split[field][r].clone()
            bucket[name] = entry
    return {"shards": new_shards, "tp": tp_after, "pp": pp_after,
            "step": ckpt["step"], "names": names}


def _restore(ckpt):
    """Read each field back at its declared precision, then widen to compute."""
    compute_dt = getattr(torch, COMPUTE_DTYPE)
    names = ckpt["names"]
    params = {}
    moments = {}
    for i in range(len(names)):
        name = names[i]
        shape = ckpt["shards"][(_stage_of(i, len(names), ckpt["pp"]), 0)][name]["shape"]
        flat = {}
        for field in FIELDS:
            flat[field] = _gather(ckpt, i, name, field).to(_load_dtype(field))
        # Weights and moments widen on separate paths: they are separately
        # configurable in every real checkpoint format, and separately wrong.
        params[name] = flat["param"].to(compute_dt).reshape(shape)
        moments[name] = (_restore_moment(flat["exp_avg"], shape),
                         _restore_moment(flat["exp_avg_sq"], shape))
    return params, moments, int(ckpt["step"])


def save_and_resume(x, y, w1_0, w2_0, resume_at=6, post_steps=12, lr=0.05,
                    beta1=0.9, beta2=0.999, eps=1e-8, tp_before=2, pp_before=1,
                    tp_after=3, pp_after=2):
    """Train, checkpoint, change topology, resume, keep training.

    Returns the loss curve on both sides of the resume plus the final parameters
    and moments; the post-resume curve is what has to match an uninterrupted run.
    """
    compute_dt = getattr(torch, COMPUTE_DTYPE)
    xf = x.to(compute_dt)
    params = {"w1": w1_0.to(compute_dt).clone(), "w2": w2_0.to(compute_dt).clone()}
    moments = {}
    for name in params:
        moments[name] = (torch.zeros_like(params[name]),
                         torch.zeros_like(params[name]))
    step = 0

    pre_losses = []
    for _ in range(int(resume_at)):
        loss, grads = _forward(xf, params["w1"], params["w2"], y)
        step += 1
        params, moments = _adam_step(params, grads, moments, step, lr,
                                     beta1, beta2, eps)
        pre_losses.append(loss)

    ckpt = _save(params, moments, step, int(tp_before), int(pp_before))
    ckpt = _reshard(ckpt, int(tp_after), int(pp_after))
    params, moments, step = _restore(ckpt)

    post_losses = []
    for _ in range(int(post_steps)):
        loss, grads = _forward(xf, params["w1"], params["w2"], y)
        step += 1
        params, moments = _adam_step(params, grads, moments, step, lr,
                                     beta1, beta2, eps)
        post_losses.append(loss)

    return {
        "pre_losses": _series(pre_losses),
        "post_losses": _series(post_losses),
        "w1": params["w1"].to(torch.float64),
        "w2": params["w2"].to(torch.float64),
        "exp_avg": torch.cat([moments["w1"][0].reshape(-1),
                              moments["w2"][0].reshape(-1)]).to(torch.float64),
        "exp_avg_sq": torch.cat([moments["w1"][1].reshape(-1),
                                 moments["w2"][1].reshape(-1)]).to(torch.float64),
    }
'''


# --------------------------------------------------------------------------- #
# the same claim under a real gloo process group (O5 launches this)
# --------------------------------------------------------------------------- #

GLOO_SOURCE = '''"""Multi-rank save, reshard and resume. gloo on CPU only.

NCCL does not exist on the machine this bank is built on, and a distributed
check that cannot run on the cheap path is a check that never runs, so every
collective here is a gloo collective and every tensor stays on the CPU.

The claim is the single-process seed's claim restated for N ranks: **a
checkpoint may not be a function of the world size.** Each rank owns a
contiguous shard of the flat optimizer state - ``exp_avg`` and ``exp_avg_sq``
alongside the weights, because an Adam resume that carries only the weights
restarts the denominator at zero and takes one enormous step the loss curve
absorbs within a handful of steps - the shards are resharded onto a *different*
rank layout, and every rank reads the whole state back before training
continues.

Reassembly is a zero-padded SUM all-reduce rather than a gather-and-concatenate.
Adding zeros is exact in every float format and in every reduction order, so the
restored state is bit-identical at 1, 2 and 4 ranks. That is deliberate: it
keeps the resume itself out of the error budget and leaves the batch reduction
as the only place where the world size is allowed to change a bit, which is
precisely the quantity O5 calibrates its noise floor on.

``run_rank`` is not defined here: the harness owns process-group setup and
teardown (:mod:`crucible.oracles.o5_worker`), and this module only has to expose
``train_step_dist`` at module scope with the group already built.
"""

import torch
import torch.distributed as dist

# The precision at each end of the checkpoint, named at module level for the
# same reason as in the single-process seed: neither end can be changed without
# the other being visible on the next line.
STATE_SAVE_DTYPE = "float32"
STATE_LOAD_DTYPE = "float32"

# The O5 payload's ``lr`` is calibrated for a raw SGD step, where the update is
# ``lr * grad`` and clipping therefore bounds it. Adam's update is about ``lr``
# per element whatever the gradient is, so reusing 0.5 unchanged would take
# steps larger than the parameters themselves and turn an equivalence test into
# a divergence test. The rescale is a property of the optimizer, not a tolerance.
ADAM_LR_SCALE = 0.02

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


def _data_bounds(shard_sizes, rank):
    """This rank's contiguous slice of the global batch."""
    off = 0
    for r in range(int(rank)):
        off += int(shard_sizes[r])
    return off, off + int(shard_sizes[int(rank)])


def _flat_bounds(numel, n_shards, rank):
    """This rank's contiguous slice of a flat state vector. The tail shard may
    be short, and with more ranks than elements a shard is legitimately empty."""
    per = (int(numel) + int(n_shards) - 1) // int(n_shards)
    lo = min(int(rank) * per, int(numel))
    return lo, min(lo + per, int(numel))


def _gather_full(shard, numel, lo, hi):
    """Rebuild the whole flat tensor from every rank's shard, exactly."""
    buf = torch.zeros(int(numel), dtype=shard.dtype)
    if hi > lo:
        buf[lo:hi] = shard
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return buf


def _checkpoint_roundtrip(flat, rank, world_size):
    """Write this rank's shard, reshard onto the mirrored layout, read it back.

    Rank ``r`` writes the slice it owns and comes back holding the slice rank
    ``world_size - 1 - r`` wrote, so the topology genuinely changes across the
    resume instead of every shard landing back where it started. Every rank
    performs the same number of collectives whatever it owns, including when its
    shard is empty.
    """
    numel = int(flat.shape[0])
    lo, hi = _flat_bounds(numel, world_size, rank)
    saved = flat[lo:hi].to(getattr(torch, STATE_SAVE_DTYPE)).clone()
    full = _gather_full(saved, numel, lo, hi)

    mlo, mhi = _flat_bounds(numel, world_size, world_size - 1 - rank)
    resharded = full[mlo:mhi].to(getattr(torch, STATE_LOAD_DTYPE)).clone()
    return _gather_full(resharded, numel, mlo, mhi)


def _local_forward(xs, ys, w1, w2):
    """Loss SUM and exact gradients over one data-parallel shard."""
    h = torch.tanh(xs @ w1)
    logits = h @ w2
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp = torch.exp(shifted)
    probs = exp / exp.sum(dim=1, keepdim=True)
    picked = probs.gather(1, ys.view(-1, 1)).clamp_min(1e-30)
    loss_sum = -torch.log(picked).sum()
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, ys.view(-1, 1), 1.0)
    dlogits = probs - onehot
    gw2 = h.transpose(0, 1) @ dlogits
    dz = (dlogits @ w2.transpose(0, 1)) * (1.0 - h * h)
    gw1 = xs.transpose(0, 1) @ dz
    return loss_sum, gw1, gw2


def _adam(param, m, v, grad, step, lr):
    """One Adam step, written out so the moments are state a checkpoint carries."""
    bias1 = 1.0 - ADAM_BETA1 ** step
    bias2 = 1.0 - ADAM_BETA2 ** step
    m = m + (1.0 - ADAM_BETA1) * (grad - m)
    v = v * ADAM_BETA2 + (1.0 - ADAM_BETA2) * grad * grad
    denom = torch.sqrt(v) / (bias2 ** 0.5) + ADAM_EPS
    return param - (lr / bias1) * m / denom, m, v


def train_step_dist(x, y, w1_0, w2_0, shard_sizes, steps=50, lr=0.5,
                    max_grad_norm=0.05, accum_dtype="float32"):
    """One rank's view of the run. The process group is already initialised."""
    acc = getattr(torch, accum_dtype)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total_n = 0
    for n in shard_sizes:
        total_n += int(n)
    if total_n <= 0:
        raise ValueError("global batch is empty; every shard has zero samples")

    lo, hi = _data_bounds(shard_sizes, rank)
    xs = x.to(acc)[lo:hi]
    ys = y[lo:hi]
    w1 = w1_0.to(acc).clone()
    w2 = w2_0.to(acc).clone()
    m1 = torch.zeros_like(w1)
    v1 = torch.zeros_like(w1)
    m2 = torch.zeros_like(w2)
    v2 = torch.zeros_like(w2)
    adam_lr = float(lr) * ADAM_LR_SCALE
    resume_at = max(int(steps) // 2, 1)

    losses = []
    norms = []
    for step in range(1, int(steps) + 1):
        if int(xs.shape[0]) == 0:                       # empty-shard guard
            loss_sum = torch.zeros((), dtype=acc)
            g1 = torch.zeros_like(w1)
            g2 = torch.zeros_like(w2)
        else:
            loss_sum, g1, g2 = _local_forward(xs, ys, w1, w2)

        # Outside the guard on purpose: an empty rank still participates in
        # every collective its peers enter.
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(g1, op=dist.ReduceOp.SUM)
        dist.all_reduce(g2, op=dist.ReduceOp.SUM)
        # Sums of per-rank SUMS over the GLOBAL sample count: a mean of per-rank
        # means silently reweights the shards the moment they are uneven.
        loss = loss_sum / total_n
        g1 = g1 / total_n
        g2 = g2 / total_n

        # GLOBAL grad norm: each rank squares only its own parameter shard, the
        # all-reduce happens BEFORE the sqrt, and the sqrt is taken exactly once.
        flat_grad = torch.cat([g1.reshape(-1), g2.reshape(-1)])
        plo, phi = _flat_bounds(int(flat_grad.shape[0]), world_size, rank)
        my_shard = flat_grad[plo:phi]
        local_sq = (my_shard * my_shard).sum()
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        global_norm = torch.sqrt(local_sq)

        # A clamp rather than an if: a coefficient that stepped discontinuously
        # at the threshold would turn a one-ulp disagreement between world sizes
        # into a macroscopic one, and the calibrated floor would then be
        # measuring the branch instead of float non-associativity.
        coef = (float(max_grad_norm) / (global_norm + 1e-6)).clamp(max=1.0)
        g1 = g1 * coef
        g2 = g2 * coef

        w1, m1, v1 = _adam(w1, m1, v1, g1, step, adam_lr)
        w2, m2, v2 = _adam(w2, m2, v2, g2, step, adam_lr)
        losses.append(loss)
        norms.append(global_norm)

        if step == resume_at:
            state = [w1, m1, v1, w2, m2, v2]
            restored = []
            for t in state:
                restored.append(
                    _checkpoint_roundtrip(t.reshape(-1), rank, world_size)
                )
            w1 = restored[0].reshape(w1.shape).to(acc)
            m1 = restored[1].reshape(m1.shape).to(acc)
            v1 = restored[2].reshape(v1.shape).to(acc)
            w2 = restored[3].reshape(w2.shape).to(acc)
            m2 = restored[4].reshape(m2.shape).to(acc)
            v2 = restored[5].reshape(v2.shape).to(acc)

    return {
        "losses": torch.stack(losses).to(torch.float32),
        "grad_norms": torch.stack(norms).to(torch.float32),
        "w": torch.cat([w1.reshape(-1), w2.reshape(-1)]).to(torch.float32),
    }
'''


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def make_inputs(shape: ShapeSpec, device: Any = "cpu", generator: Any = None) -> dict[str, Any]:
    import torch

    kw = shape.kwargs
    n = int(kw["n"])
    d = int(kw["d"])
    h = int(kw["h"])
    k = int(kw["k"])
    x = torch.randn(n, d, generator=generator, device=device, dtype=torch.float32)
    y = torch.randint(0, k, (n,), generator=generator, device=device, dtype=torch.int64)
    w1_0 = 0.5 * torch.randn(d, h, generator=generator, device=device, dtype=torch.float32)
    w2_0 = 0.5 * torch.randn(h, k, generator=generator, device=device, dtype=torch.float32)
    return {
        "x": x,
        "y": y,
        "w1_0": w1_0,
        "w2_0": w2_0,
        "resume_at": int(kw.get("resume_at", 6)),
        "post_steps": int(kw.get("post_steps", 12)),
        "lr": float(kw.get("lr", 0.05)),
        "beta1": float(kw.get("beta1", 0.9)),
        "beta2": float(kw.get("beta2", 0.999)),
        "eps": float(kw.get("eps", 1e-8)),
        "tp_before": int(kw.get("tp_before", 2)),
        "pp_before": int(kw.get("pp_before", 1)),
        "tp_after": int(kw.get("tp_after", 3)),
        "pp_after": int(kw.get("pp_after", 2)),
    }


# --------------------------------------------------------------------------- #
# independent ground truth
# --------------------------------------------------------------------------- #


def reference(
    x: Any,
    y: Any,
    w1_0: Any,
    w2_0: Any,
    resume_at: int = 6,
    post_steps: int = 12,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    tp_before: int = 2,
    pp_before: int = 1,
    tp_after: int = 3,
    pp_after: int = 2,
) -> dict[str, Any]:
    """Ground truth: the same run, never interrupted.

    autograd for the gradients and ``torch.optim.Adam`` for the update - a
    different code path from the hand-rolled step, and one that never
    serialises anything. The topology arguments are accepted and ignored: a
    checkpoint that changes the trajectory has failed no matter what topology it
    was written under, and that invariant is the whole test.
    """
    import torch
    import torch.nn.functional as F

    xf = torch.as_tensor(x).to(torch.float64)
    yl = torch.as_tensor(y).to(torch.int64)
    w1 = torch.nn.Parameter(torch.as_tensor(w1_0).to(torch.float64).clone())
    w2 = torch.nn.Parameter(torch.as_tensor(w2_0).to(torch.float64).clone())
    opt = torch.optim.Adam([w1, w2], lr=float(lr), betas=(float(beta1), float(beta2)), eps=float(eps))

    losses: list[float] = []
    for _ in range(int(resume_at) + int(post_steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(torch.tanh(xf @ w1) @ w2, yl)
        loss.backward()
        losses.append(float(loss.detach()))
        opt.step()

    def moment(param: Any, field: str) -> Any:
        state = opt.state.get(param, {})
        if field in state:
            return state[field].detach().reshape(-1)
        return torch.zeros(param.numel(), dtype=torch.float64)

    return {
        "pre_losses": torch.tensor(losses[: int(resume_at)], dtype=torch.float64),
        "post_losses": torch.tensor(losses[int(resume_at) :], dtype=torch.float64),
        "w1": w1.detach().to(torch.float64),
        "w2": w2.detach().to(torch.float64),
        "exp_avg": torch.cat([moment(w1, "exp_avg"), moment(w2, "exp_avg")]).to(torch.float64),
        "exp_avg_sq": torch.cat([moment(w1, "exp_avg_sq"), moment(w2, "exp_avg_sq")]).to(
            torch.float64
        ),
    }


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #

#: post_losses first: it is the trajectory claim, so it is the failure we want
#: named when several keys disagree.
#:
#: The two moment vectors are graded, not just the final weights, and that is
#: what makes a rounded resume detectable at all. A bf16 round trip of
#: ``exp_avg_sq`` lands on the weights only through Adam's denominator, where it
#: is diluted by the square root and then largely cancelled by the same rounding
#: appearing in ``exp_avg`` - the weights move by far less than the state did,
#: and after a handful of steps they are still visibly the same run. The moments
#: carry the error at full size on the first post-resume step and keep carrying
#: it, because an exponential moving average never forgets: every later moment
#: is ``beta`` times the rounded one plus a correct increment. Grading the state
#: itself, and grading a post-resume trajectory long enough for the state error
#: to reach the loss, is the difference between an observable and a decoy.
_KEYS = ("post_losses", "exp_avg_sq", "exp_avg", "w1", "w2", "pre_losses")

#: fp64 hand-rolled Adam agrees with torch's to ~1e-14 relative. A single bf16
#: round trip of exp_avg_sq perturbs it by ~4e-3. Six orders of margin either
#: way, which is the point of computing in fp64: the checkpoint is then the only
#: lossy step in the pipeline.
_RTOL = 1.0e-8
_ATOL = 1.0e-12


def compare(got: Any, want: Any) -> CompareResult:
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
        rel_err = abs_err / w.abs().clamp_min(1e-300)
        budget = _ATOL + _RTOL * w.abs()
        bad = abs_err > budget
        worst_abs = max(worst_abs, float(abs_err.max()))
        worst_rel = max(worst_rel, float(rel_err.max()))
        if bool(bad.any()) and ok:
            ok = False
            i = int(torch.nonzero(bad)[0])
            detail = (
                f"resumed run left the trajectory: {key}[{i}] = {float(g[i]):.17g} "
                f"vs uninterrupted {float(w[i]):.17g} (abs {float(abs_err[i]):.3g} "
                f"> budget {float(budget[i]):.3g})"
            )
    if ok:
        detail = (
            f"resumed trajectory matches the uninterrupted run: "
            f"max_abs={worst_abs:.3g} max_rel={worst_rel:.3g} (rtol={_RTOL:g})"
        )
    return CompareResult(
        ok=ok, max_abs_err=worst_abs, max_rel_err=worst_rel, detail=detail, kind="loss_curve"
    )


# --------------------------------------------------------------------------- #
# adversarial sweep
# --------------------------------------------------------------------------- #

_BASE = {"n": 24, "d": 6, "h": 5, "k": 3}

#: Every shape that can detect anything runs a long post-resume tail. A rounded
#: moment reaches the loss curve through Adam's denominator, so it needs steps to
#: turn a state error into a trajectory error; twenty is comfortably past the
#: point where the two curves separate, and the model is small enough that the
#: extra steps cost microseconds.
_POST = 20

SHAPES: list[ShapeSpec] = [
    # TP 2 -> 3 and PP 1 -> 2: neither rank count divides the other and the
    # parameter counts (30, 15) divide neither.
    ShapeSpec(
        name="tp2to3_pp1to2",
        kwargs={**_BASE, "resume_at": 6, "post_steps": _POST, "tp_before": 2, "pp_before": 1,
                "tp_after": 3, "pp_after": 2},
    ),
    # Shrinking the topology instead of growing it.
    ShapeSpec(
        name="tp4to1_pp2to1",
        kwargs={**_BASE, "resume_at": 6, "post_steps": _POST, "tp_before": 4, "pp_before": 2,
                "tp_after": 1, "pp_after": 1},
    ),
    # More tensor-parallel ranks than parameters in a layer: empty shards on
    # both sides of the reshard.
    ShapeSpec(
        name="tp_more_ranks_than_params",
        kwargs={"n": 24, "d": 2, "h": 2, "k": 2, "resume_at": 5, "post_steps": _POST,
                "tp_before": 9, "pp_before": 1, "tp_after": 7, "pp_after": 2},
    ),
    # More pipeline stages than layers: a stage owns nothing.
    ShapeSpec(
        name="pp3_empty_stage",
        kwargs={**_BASE, "resume_at": 6, "post_steps": _POST, "tp_before": 2, "pp_before": 1,
                "tp_after": 2, "pp_after": 3},
    ),
    # Resume exactly at the first Adam step: bias correction is at its most
    # extreme and the moments are one step old.
    ShapeSpec(
        name="resume_at_step1",
        kwargs={**_BASE, "resume_at": 1, "post_steps": _POST, "tp_before": 2, "pp_before": 1,
                "tp_after": 3, "pp_after": 2},
    ),
    # Resume before any step at all: the moments are exactly zero, so any
    # save/load dtype round-trips them losslessly. A decoy, not a detector.
    ShapeSpec(
        name="resume_at_step0_decoy",
        kwargs={**_BASE, "resume_at": 0, "post_steps": _POST, "tp_before": 2, "pp_before": 1,
                "tp_after": 3, "pp_after": 2},
    ),
    # Topology unchanged across the resume: resharding is the identity here, so
    # a broken reshard can still pass. Also a decoy.
    ShapeSpec(
        name="topology_unchanged_decoy",
        kwargs={**_BASE, "resume_at": 6, "post_steps": _POST, "tp_before": 1, "pp_before": 1,
                "tp_after": 1, "pp_after": 1},
    ),
    # Twice the tail again: drift that is invisible one step after the resume is
    # not invisible forty steps after it.
    ShapeSpec(
        name="long_tail_after_resume",
        kwargs={**_BASE, "resume_at": 4, "post_steps": 2 * _POST, "tp_before": 3, "pp_before": 2,
                "tp_after": 2, "pp_after": 1},
    ),
]


def _n_params(shape: ShapeSpec) -> int:
    kw = shape.kwargs
    return int(kw["d"]) * int(kw["h"]) + int(kw["h"]) * int(kw["k"])


def _total_steps(shape: ShapeSpec) -> int:
    return int(shape.kwargs.get("resume_at", 6)) + int(shape.kwargs.get("post_steps", 12))


def accum_depth(shape: ShapeSpec) -> int:
    """Adam's moments are an exponential moving average, so every step chains on
    the previous one: the accumulation is as deep as the run is long. The batch
    reduction inside a step adds ``n``."""
    return _total_steps(shape) * (int(shape.kwargs["n"]) + 1)


def bytes_moved(shape: ShapeSpec) -> int:
    """fp64 traffic. Per step: read x, read params and both moments, write
    params and both moments. Plus one checkpoint write and read of
    3 * n_params, and a reshard that gathers and re-splits all three fields."""
    kw = shape.kwargs
    p = _n_params(shape)
    per_step = (int(kw["n"]) * int(kw["d"]) + 7 * p) * 8
    checkpoint = 3 * p * 8 * 4  # save, gather, re-split, restore
    return _total_steps(shape) * per_step + checkpoint


def flops(shape: ShapeSpec) -> int:
    """Forward and backward through both layers, plus Adam's elementwise work."""
    kw = shape.kwargs
    n = int(kw["n"])
    d = int(kw["d"])
    h = int(kw["h"])
    k = int(kw["k"])
    matmuls = 6 * n * d * h + 6 * n * h * k
    adam = 10 * _n_params(shape)
    return _total_steps(shape) * (matmuls + adam)


SEED: SeedSpec = register(
    SeedSpec(
        id="checkpointing.save_and_resume",
        domain="checkpointing",
        tiers=("T4", "T5", "T6"),
        description=(
            "Adam state saved and restored at an explicit dtype and resharded "
            "across a TP/PP topology change; graded on whether the resumed run "
            "stays on the uninterrupted trajectory."
        ),
        entry="save_and_resume",
        source=SOURCE,
        make_inputs=make_inputs,
        reference=reference,
        shape_sweep=list(SHAPES),
        accum_depth=accum_depth,
        bytes_moved=bytes_moved,
        flops=flops,
        denylist=(
            # The optimizer under test must be the one in the candidate's source.
            "torch.optim",
            "torch.autograd.grad",
            "torch.autograd.backward",
            "torch.nn.functional.cross_entropy",
            "torch.nn.CrossEntropyLoss",
            # Pickle round trips hide the dtype decision this seed is about, and
            # deserialising a checkpoint is not something a grader should do.
            "torch.save",
            "torch.load",
            "pickle",
            "crucible.seeds.checkpointing",
        ),
        compare=compare,
        supports_cpu=True,
        module=__name__,
        extras={
            "compute_dtype": "float64",
            # O5 needs a real multi-rank program or it cannot make an
            # equivalence claim at all, and a SKIP is never a PASS.
            "gloo_source": GLOO_SOURCE,
            "gloo_step_entry": "train_step_dist",
            "gloo_world_sizes": (2, 4),
            "silent_failures": (
                "resume_fidelity: exp_avg_sq saved fp32 restored bf16, curve still smooth",
                "reshard order: gather in the wrong rank order across a TP change",
                "stage assignment: a pipeline stage claims a layer it does not own",
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
