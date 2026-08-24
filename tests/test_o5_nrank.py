"""Tests for O5, the N-rank equivalence oracle.

Which launcher these tests exercise, stated plainly because it matters:

* Every **behavioural** assertion (noise floor, correct step passes, each of the
  four defects is caught, verdict wiring) runs through the ``thread`` launcher.
  It runs the same candidate source through the same argument marshalling
  (:mod:`crucible.oracles.o5_worker`) as production, on threads against a
  deterministic ``torch.distributed`` stand-in. That keeps the suite CPU-only,
  free of process spawning under a test runner, and fast enough that the
  calibration can be repeated in every test.
* The **spawn** launcher - the production path, real gloo, real processes - is
  exercised by two tests of its own: one that runs two real ranks end to end,
  and one that hangs a rank and asserts the process tree was reaped. Both skip
  with the observed reason if this machine cannot spawn, and neither is allowed
  to make a behavioural claim on its own.

Nothing here needs a GPU, a network, or triton.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import pytest

from crucible.capabilities import Capabilities
from crucible.config import Config
from crucible.oracles import o5_nrank, o5_worker
from crucible.oracles.base import ORACLES, OracleContext, gate_capabilities
from crucible.schema import Task

# --------------------------------------------------------------------------- #
# candidate sources: one correct, four defective, one that hangs
# --------------------------------------------------------------------------- #

CORRECT_SRC = '''"""A correct two-layer data-parallel training step.

Every reduction is global: the gradient and the loss are summed across ranks and
divided by the GLOBAL sample count, and the grad norm's sum of squares is
all-reduced BEFORE the square root is taken.
"""
import torch
import torch.distributed as dist


def _shard_bounds(shard_sizes, rank):
    off = 0
    for r in range(int(rank)):
        off += int(shard_sizes[r])
    return off, off + int(shard_sizes[int(rank)])


def _flat_bounds(numel, n_shards, rank):
    per = (numel + n_shards - 1) // n_shards
    lo = min(rank * per, numel)
    return lo, min(lo + per, numel)


def _local(xs, ys, w1, w2):
    h = torch.tanh(xs @ w1)
    logits = h @ w2
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp = torch.exp(shifted)
    probs = exp / exp.sum(dim=1, keepdim=True)
    picked = probs.gather(1, ys.view(-1, 1)).clamp_min(1e-30)
    loss_sum = -torch.log(picked).sum().reshape(1)
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, ys.view(-1, 1), 1.0)
    dlogits = probs - onehot
    g2 = h.transpose(0, 1) @ dlogits
    dh = dlogits @ w2.transpose(0, 1)
    dz = dh * (1.0 - h * h)
    g1 = xs.transpose(0, 1) @ dz
    return loss_sum, g1, g2


def train_step_dist(x, y, w1_0, w2_0, shard_sizes, steps=8, lr=0.5,
                    max_grad_norm=0.05, accum_dtype="float32"):
    acc = getattr(torch, accum_dtype)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total_n = 0
    for n in shard_sizes:
        total_n += int(n)
    lo, hi = _shard_bounds(shard_sizes, rank)
    xs = x.to(acc)[lo:hi]
    ys = y[lo:hi]
    w1 = w1_0.to(acc).clone()
    w2 = w2_0.to(acc).clone()

    losses = []
    norms = []
    for _ in range(int(steps)):
        if int(xs.shape[0]) == 0:
            l = torch.zeros(1, dtype=acc)
            g1 = torch.zeros_like(w1)
            g2 = torch.zeros_like(w2)
        else:
            l, g1, g2 = _local(xs, ys, w1, w2)
        dist.all_reduce(l, op=dist.ReduceOp.SUM)
        dist.all_reduce(g1, op=dist.ReduceOp.SUM)
        dist.all_reduce(g2, op=dist.ReduceOp.SUM)
        loss = l[0] / total_n
        g1 = g1 / total_n
        g2 = g2 / total_n

        flat = torch.cat([g1.reshape(-1), g2.reshape(-1)])
        plo, phi = _flat_bounds(int(flat.shape[0]), world_size, rank)
        mine = flat[plo:phi]
        local_sq = (mine * mine).sum().reshape(1)
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        global_norm = torch.sqrt(local_sq[0])

        if global_norm > max_grad_norm:
            scale = max_grad_norm / (global_norm + 1e-6)
            g1 = g1 * scale
            g2 = g2 * scale
        w1 = w1 - lr * g1
        w2 = w2 - lr * g2
        losses.append(loss)
        norms.append(global_norm)

    return {
        "losses": torch.stack(losses).to(torch.float32),
        "grad_norms": torch.stack(norms).to(torch.float32),
        "w": torch.cat([w1.reshape(-1), w2.reshape(-1)]).to(torch.float32),
    }
'''


def patched(source: str, old: str, new: str) -> str:
    """Rewrite exactly one occurrence, or fail loudly.

    A silently-missed patch would hand the buggy test a correct program and turn
    a red-team assertion into a green one that proves nothing.
    """
    if source.count(old) != 1:
        raise AssertionError(
            f"expected exactly one occurrence of the patch site, found {source.count(old)}:\n{old}"
        )
    return source.replace(old, new, 1)


#: Defect 1 - the sum of squares is never all-reduced, so every rank clips by
#: the norm of its own parameter shard. The loss curve still descends.
PER_SHARD_NORM_SRC = patched(
    CORRECT_SRC,
    "        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)\n"
    "        global_norm = torch.sqrt(local_sq[0])\n",
    "        global_norm = torch.sqrt(local_sq[0])\n",
)

#: Defect 2 - the gradient all-reduce is dropped entirely; each rank trains on
#: its own shard's gradient scaled by the global sample count.
NO_ALLREDUCE_SRC = patched(
    CORRECT_SRC,
    "        dist.all_reduce(g1, op=dist.ReduceOp.SUM)\n"
    "        dist.all_reduce(g2, op=dist.ReduceOp.SUM)\n",
    "",
)

#: Defect 3 - the data-parallel reduction is a mean of per-rank means instead of
#: a global mean. Exactly correct on even shards; wrong the moment they are not.
MEAN_OF_MEANS_SRC = patched(
    patched(
        CORRECT_SRC,
        "        dist.all_reduce(l, op=dist.ReduceOp.SUM)\n",
        "        n_local = max(int(shard_sizes[rank]), 1)\n"
        "        l = l / n_local\n"
        "        g1 = g1 / n_local\n"
        "        g2 = g2 / n_local\n"
        "        dist.all_reduce(l, op=dist.ReduceOp.SUM)\n",
    ),
    "        loss = l[0] / total_n\n"
    "        g1 = g1 / total_n\n"
    "        g2 = g2 / total_n\n",
    "        loss = l[0] / world_size\n"
    "        g1 = g1 / world_size\n"
    "        g2 = g2 / world_size\n",
)

#: Defect 4 - optimizer/parameter state that is mis-sharded when the run
#: resumes: the rank reloads its slice of the flat parameter vector at the wrong
#: offset, so ranks start the resumed steps from different weights.
MISSHARDED_RESUME_SRC = patched(
    CORRECT_SRC,
    "    w1 = w1_0.to(acc).clone()\n    w2 = w2_0.to(acc).clone()\n",
    "    # 'resume': rebuild the parameters from the flat checkpoint. The offset\n"
    "    # is computed from the rank instead of from the parameter layout, so a\n"
    "    # multi-rank resume reads its neighbour's slice.\n"
    "    _flat0 = torch.cat([w1_0.reshape(-1), w2_0.reshape(-1)]).to(acc)\n"
    "    _n1 = int(w1_0.numel())\n"
    "    _off = (int(rank) * 3) % max(int(_flat0.shape[0]), 1)\n"
    "    _rolled = torch.cat([_flat0[_off:], _flat0[:_off]])\n"
    "    w1 = _rolled[:_n1].reshape(w1_0.shape).clone()\n"
    "    w2 = _rolled[_n1:].reshape(w2_0.shape).clone()\n",
)

#: Not a numeric defect: rank 1 never arrives at the collectives its peers
#: entered. In a real process group that is a deadlock, and it is a T3 finding.
HANG_SRC = patched(
    patched(CORRECT_SRC, "import torch\nimport torch.distributed as dist\n",
            "import time\n\nimport torch\nimport torch.distributed as dist\n"),
    "    for _ in range(int(steps)):\n",
    "    for _ in range(int(steps)):\n"
    "        if int(rank) == 1:\n"
    "            time.sleep(120.0)\n",
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

#: Small enough that three calibration runs plus two world sizes finish in
#: about a second, large enough that reordering the batch actually moves a bit.
TEST_SPEC = {
    "n_samples": 18,
    "d": 8,
    "h": 8,
    "k": 3,
    "steps": 8,
    "lr": 0.5,
    "max_grad_norm": 0.05,
    "seed": 1234,
}


@pytest.fixture
def nrank_cfg(cfg: Config) -> Config:
    """Three calibration runs, so the floor is a max over three pairs."""
    return cfg.model_copy(update={"nrank_noise_floor_runs": 3, "nrank_k": 3.0})


def make_ctx(
    task: Task,
    caps: Capabilities,
    cfg: Config,
    workdir: Path,
    source: str,
    world_sizes: list[int],
    launcher: str = "thread",
    **extra: Any,
) -> OracleContext:
    extras: dict[str, Any] = {
        "nrank_source": source,
        "nrank_entry": "train_step_dist",
        "nrank_launcher": launcher,
        "nrank_world_sizes": list(world_sizes),
        "nrank_spec": dict(TEST_SPEC),
        "nrank_launch_timeout_s": 60.0,
    }
    extras.update(extra)
    return OracleContext(
        task=task,
        candidate_src=source,
        seed=None,
        caps=caps,
        workdir=workdir,
        cfg=cfg,
        rng_seed=1234,
        device="cpu",
        extras=extras,
    )


# --------------------------------------------------------------------------- #
# registration and gating
# --------------------------------------------------------------------------- #


def test_oracle_registers_itself_as_O5() -> None:
    from crucible.oracles import load_oracles

    load_oracles()
    assert "O5" in ORACLES
    assert ORACLES["O5"].id == "O5"
    assert ORACLES["O5"].required_caps == ("gloo",)


def test_applies_to_distributed_tasks_and_to_tasks_that_ask_for_O5(
    make_task: Callable[..., Task]
) -> None:
    oracle = o5_nrank.O5
    assert oracle.applies_to(make_task(domain="distributed", oracles=["O1"]))
    assert oracle.applies_to(make_task(domain="pytorch", oracles=["O1", "O5"]))
    assert not oracle.applies_to(make_task(domain="pytorch", oracles=["O1", "O3"]))


def test_capability_gate_skips_when_gloo_is_absent(caps: Capabilities) -> None:
    no_gloo = Capabilities(
        **{**caps.as_dict(), "gloo": False, "details": {"gloo": "is_gloo_available() returned False"}}
    )
    reason = gate_capabilities(o5_nrank.O5, no_gloo)
    assert reason is not None
    assert "gloo" in reason


def test_run_skips_without_gloo_rather_than_passing(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    no_gloo = Capabilities(**{**caps.as_dict(), "gloo": False, "details": {"gloo": "no gloo here"}})
    ctx = make_ctx(sample_task, no_gloo, nrank_cfg, tmp_workdir, CORRECT_SRC, [2])
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "SKIP"
    assert "gloo" in result.reason
    assert "no gloo here" in result.reason


def test_run_skips_when_there_is_no_distributed_source(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    ctx = OracleContext(
        task=sample_task,
        candidate_src="",
        seed=None,
        caps=caps,
        workdir=tmp_workdir,
        cfg=nrank_cfg,
        extras={},
    )
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "SKIP"
    assert "gloo_source" in result.reason


# --------------------------------------------------------------------------- #
# the workload
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 2, 7, 18, 50])
@pytest.mark.parametrize("salt", [1, 2, 13])
def test_permutation_is_a_bijection(n: int, salt: int) -> None:
    perm = o5_nrank.permutation(n, salt)
    assert sorted(perm) == list(range(n))


def test_permutation_with_salt_zero_is_not_requested_and_identity_is_used() -> None:
    payload_plain = o5_nrank.build_payload(o5_nrank.NRankSpec(**TEST_SPEC), 1, perm_salt=0)
    payload_perm = o5_nrank.build_payload(o5_nrank.NRankSpec(**TEST_SPEC), 1, perm_salt=1)
    assert payload_plain["y"] != payload_perm["y"]
    assert sorted(payload_plain["y"]) == sorted(payload_perm["y"])


def test_shard_sizes_are_contiguous_uneven_and_cover_the_batch() -> None:
    spec = o5_nrank.NRankSpec(**TEST_SPEC)
    for ws in (1, 2, 3, 4, 5):
        sizes = spec.shard_sizes(ws)
        assert len(sizes) == ws
        assert sum(sizes) == spec.n_samples
        assert max(sizes) - min(sizes) <= 1
    assert spec.shard_sizes(4) == [5, 5, 4, 4]


def test_spec_refuses_a_model_above_the_parameter_budget() -> None:
    spec = o5_nrank.NRankSpec(d=1024, h=1024, k=1024)
    assert spec.n_params() > o5_nrank.MAX_PARAMS
    with pytest.raises(ValueError, match="parameter"):
        spec.validate()


def test_unknown_spec_key_is_an_error_not_a_silent_default(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(
        sample_task, caps, nrank_cfg, tmp_workdir, CORRECT_SRC, [2],
        nrank_spec={"n_smaples": 4},
    )
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "SKIP"
    assert "n_smaples" in result.reason


# --------------------------------------------------------------------------- #
# stage 1: the calibration itself
# --------------------------------------------------------------------------- #


def test_noise_floor_is_measured_small_and_nonzero(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    """The whole oracle rests on this number, so it is asserted directly."""
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, CORRECT_SRC, [2])
    result = o5_nrank.O5.run(ctx)

    floor = result.evidence["noise_floor"]
    assert floor["runs"] == 3
    assert floor["permutation_salts"] == [0, 1, 2]

    entries = floor["entries"]
    assert set(entries) == {"losses", "grad_norms", "final_params"}

    # Measured, not guessed: at least one quantity actually moved when the
    # microbatch accumulation order changed, and none of them moved much.
    measured = {name: e["measured"] for name, e in entries.items()}
    assert any(v > 0.0 for v in measured.values()), measured
    assert all(v < 1e-4 for v in measured.values()), measured

    # Whatever wins, it is derived from an observation or from unit roundoff at
    # the quantity's own magnitude - never from a constant.
    for name, e in entries.items():
        assert e["source"] in ("measured", "ulp_fallback"), name
        assert e["used"] == pytest.approx(max(e["measured"], e["ulp_bound"]))
        assert e["used"] > 0.0
        assert e["ulp_bound"] == pytest.approx(1.1920929e-07 * e["scale"], rel=1e-6)

    assert entries["losses"]["measured"] > 0.0
    assert len(floor["baseline_series"]["losses"]) == TEST_SPEC["steps"]


def test_noise_floor_needs_at_least_two_runs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        o5_nrank.calibrate_noise_floor([], [], "thread")


def test_calibration_happens_before_any_n_rank_comparison(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    """A failing candidate still gets its floor measured first: the threshold
    must not be a function of the thing it is judging."""
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, PER_SHARD_NORM_SRC, [2])
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "FAIL"
    assert "noise_floor" in result.evidence
    assert len(result.evidence["noise_floor_launches"]) == 3
    assert all(rec["world_size"] == 1 for rec in result.evidence["noise_floor_launches"])


# --------------------------------------------------------------------------- #
# stage 2: the equivalence claim
# --------------------------------------------------------------------------- #


def test_correct_ddp_step_passes_at_two_ranks(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, CORRECT_SRC, [2])
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "PASS", result.reason

    rec = result.evidence["world_sizes"]["2"]
    assert rec["status"] == "pass"
    assert rec["shard_sizes"] == [9, 9]
    assert set(rec["checks"]) == set(o5_nrank.CHECK_NAMES)
    for name, check in rec["checks"].items():
        assert check["verdict"] == "PASS", (name, check["detail"])
        assert check["margin"] >= 0.0
        assert check["k"] == 3.0
    # The full per-step series is reported, not just its maximum.
    assert len(rec["checks"]["losses"]["per_step_deviation"]) == TEST_SPEC["steps"]
    assert len(rec["checks"]["grad_norms"]["per_step_deviation"]) == TEST_SPEC["steps"]
    assert rec["final_params_l2"] >= 0.0
    assert "gloo" in result.capabilities_used


def test_correct_ddp_step_passes_at_four_uneven_ranks(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, CORRECT_SRC, [2, 4])
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "PASS", result.reason
    assert set(result.evidence["world_sizes"]) == {"2", "4"}
    assert result.evidence["world_sizes"]["4"]["shard_sizes"] == [5, 5, 4, 4]


def test_per_shard_grad_norm_fails_and_shows_up_first_in_the_norm_series(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, PER_SHARD_NORM_SRC, [2])
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "FAIL", result.reason

    checks = result.evidence["world_sizes"]["2"]["checks"]
    assert checks["grad_norms"]["verdict"] == "FAIL"
    assert "grad_norms" in result.evidence["world_sizes"]["2"]["failed_checks"]

    norm_step = checks["grad_norms"]["first_bad_step"]
    assert norm_step is not None
    # This is why O5 grades a norm series at all: the defect is in the norm from
    # the very first step, and only reaches the loss once a wrong clipping
    # decision has had time to move the weights.
    loss_step = checks["losses"]["first_bad_step"]
    if loss_step is not None:
        assert norm_step <= loss_step
    assert norm_step == 0

    assert checks["grad_norms"]["observed_max_deviation"] > checks["grad_norms"]["threshold"]
    assert checks["grad_norms"]["margin"] < 0.0
    assert "noise floor" in checks["grad_norms"]["detail"]

    # Each rank clipped by the norm of its own shard, so the ranks did not even
    # come out of the step reporting the same norm.
    assert checks["rank_agreement"]["verdict"] == "FAIL"


def test_dropped_gradient_all_reduce_fails(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, NO_ALLREDUCE_SRC, [2])
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "FAIL", result.reason
    failed = result.evidence["world_sizes"]["2"]["failed_checks"]
    assert "losses" in failed
    # Not marginal: each rank trained on a gradient that never left it, so the
    # trajectory is a different run, not a reordered one.
    losses = result.evidence["world_sizes"]["2"]["checks"]["losses"]
    assert losses["observed_max_deviation"] > 1000.0 * losses["threshold"]
    assert result.evidence["world_sizes"]["2"]["checks"]["final_params"]["verdict"] == "FAIL"


def test_wrong_dp_loss_reduction_fails_on_uneven_shards(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    """Mean-of-per-rank-means. Four ranks over 18 samples is [5, 5, 4, 4]."""
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, MEAN_OF_MEANS_SRC, [4])
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "FAIL", result.reason
    assert result.evidence["world_sizes"]["4"]["shard_sizes"] == [5, 5, 4, 4]
    assert "losses" in result.evidence["world_sizes"]["4"]["failed_checks"]


def test_missharded_state_on_resume_fails(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, MISSHARDED_RESUME_SRC, [2])
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "FAIL", result.reason
    rec = result.evidence["world_sizes"]["2"]
    assert rec["failed_checks"]
    assert rec["checks"]["final_params"]["observed_max_deviation"] > 0.0


def test_failure_reason_quotes_the_floor_and_k(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, PER_SHARD_NORM_SRC, [2])
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "FAIL"
    assert "Noise floor" in result.reason
    assert "k=3" in result.reason
    assert "measured" in result.reason or "ulp_fallback" in result.reason


def test_pass_reason_quotes_the_floor_too(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, CORRECT_SRC, [2])
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "PASS"
    assert "invariant to the world size" in result.reason
    assert "k=3" in result.reason


def test_candidate_exception_is_a_fail_not_an_error(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    broken = patched(
        CORRECT_SRC,
        "    losses = []\n",
        "    raise RuntimeError('candidate blew up')\n    losses = []\n",
    )
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, broken, [2])
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "FAIL"
    assert "candidate blew up" in result.reason


def test_missing_entry_point_is_reported_by_name(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(
        sample_task, caps, nrank_cfg, tmp_workdir, CORRECT_SRC, [2],
        nrank_entry="no_such_entry",
    )
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "FAIL"
    assert "no_such_entry" in result.reason


def test_world_size_larger_than_the_batch_is_excluded_with_a_reason(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, CORRECT_SRC, [2, 64])
    result = o5_nrank.O5.run(ctx)
    excluded = result.evidence["world_sizes_excluded"]
    assert [e["world_size"] for e in excluded] == [64]
    assert "no data" in excluded[0]["reason"]
    assert result.verdict == "PASS", result.reason


def test_unknown_launcher_is_an_error(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    ctx = make_ctx(sample_task, caps, nrank_cfg, tmp_workdir, CORRECT_SRC, [2], launcher="mpi")
    result = o5_nrank.O5.run(ctx)
    assert result.verdict == "ERROR"
    assert "mpi" in result.reason


# --------------------------------------------------------------------------- #
# the thread simulation itself
# --------------------------------------------------------------------------- #


def test_thread_sim_all_reduce_sums_across_ranks(tmp_path: Path) -> None:
    src = '''
import torch
import torch.distributed as dist


def train_step_dist(shard_sizes, steps=1):
    rank = dist.get_rank()
    t = torch.tensor([float(rank) + 1.0])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {"losses": t, "grad_norms": t, "w": t}
'''
    res = o5_nrank.run_world(
        src, "train_step_dist", {"shard_sizes": [1, 1, 1], "steps": 1},
        3, tmp_path / "sum", 30.0, launcher="thread",
    )
    assert res.ok, res.error
    for rank in range(3):
        assert res.summaries[rank]["losses"] == [6.0]


def test_thread_sim_rejects_a_collective_it_does_not_implement(tmp_path: Path) -> None:
    src = '''
import torch
import torch.distributed as dist


def train_step_dist(shard_sizes, steps=1):
    out = [torch.zeros(1) for _ in range(dist.get_world_size())]
    dist.all_gather(out, torch.ones(1))
    return {"losses": out[0], "grad_norms": out[0], "w": out[0]}
'''
    res = o5_nrank.run_world(
        src, "train_step_dist", {"shard_sizes": [1, 1], "steps": 1},
        2, tmp_path / "gather", 30.0, launcher="thread",
    )
    assert not res.ok
    assert res.error_kind == "candidate"
    assert "all_gather" in res.error


def test_thread_sim_refuses_a_source_it_cannot_intercept(tmp_path: Path) -> None:
    src = '''
def train_step_dist(shard_sizes, steps=1):
    return {"losses": [0.0], "grad_norms": [0.0], "w": [0.0]}
'''
    res = o5_nrank.run_world(
        src, "train_step_dist", {"shard_sizes": [1, 1], "steps": 1},
        2, tmp_path / "nodist", 30.0, launcher="thread",
    )
    assert not res.ok
    assert "torch.distributed" in res.error


def test_a_hanging_rank_is_a_finding_not_a_pass(
    sample_task: Task, caps: Capabilities, nrank_cfg: Config, tmp_workdir: Path
) -> None:
    """The thread launcher cannot kill a thread and says so; the verdict is
    still FAIL, because a rank that never reaches a collective is the defect."""
    ctx = make_ctx(
        sample_task, caps, nrank_cfg, tmp_workdir, HANG_SRC, [2],
        nrank_launch_timeout_s=2.0,
    )
    started = time.perf_counter()
    result = o5_nrank.O5.run(ctx)
    elapsed = time.perf_counter() - started

    assert result.verdict == "FAIL", result.reason
    assert "hung" in result.reason or "desync" in result.reason
    assert elapsed < 60.0
    launch = result.evidence["world_sizes"]["2"]["launch"]
    assert launch["timed_out"] is True
    assert launch["error_kind"] == "hang"
    # Honest about what the simulation cannot do, rather than claiming a clean
    # teardown it did not perform.
    assert launch["reaped"] is False


# --------------------------------------------------------------------------- #
# the production launcher: real processes, real gloo
# --------------------------------------------------------------------------- #


def _skip_if_unspawnable(res: o5_nrank.LaunchResult) -> None:
    if res.error_kind == "infrastructure":
        pytest.skip(f"this machine could not spawn a gloo process group: {res.error}")


def test_spawn_launcher_runs_two_real_gloo_ranks(tmp_path: Path) -> None:
    spec = o5_nrank.NRankSpec(**{**TEST_SPEC, "steps": 2})
    payload = o5_nrank.build_payload(spec, 2)
    res = o5_nrank.run_world(
        CORRECT_SRC, "train_step_dist", payload, 2,
        tmp_path / "spawn2", 180.0, launcher="spawn",
    )
    _skip_if_unspawnable(res)
    assert res.ok, res.error
    assert res.launcher == "spawn"
    assert sorted(res.summaries) == [0, 1]
    assert len(res.pids) == 2
    assert res.alive_after_reap == []
    assert res.reaped is True
    # Both ranks came out of the collectives with the same answer.
    assert res.summaries[0]["losses"] == pytest.approx(res.summaries[1]["losses"], rel=1e-6)
    assert len(res.summaries[0]["grad_norms"]) == 2


def test_spawn_hang_is_reaped_and_leaves_no_orphans(tmp_path: Path) -> None:
    spec = o5_nrank.NRankSpec(**{**TEST_SPEC, "steps": 2})
    payload = o5_nrank.build_payload(spec, 2)
    res = o5_nrank.run_world(
        HANG_SRC, "train_step_dist", payload, 2,
        tmp_path / "spawnhang", 25.0, launcher="spawn",
    )
    if res.error_kind == "infrastructure":
        pytest.skip(f"this machine could not spawn a gloo process group: {res.error}")
    assert not res.ok
    assert res.timed_out is True
    assert res.error_kind == "hang"
    # The invariant the finally block exists for.
    assert res.alive_after_reap == [], f"orphaned rank processes: {res.alive_after_reap}"
    assert res.reaped is True
    assert res.pids and all(pid > 0 for pid in res.pids)
