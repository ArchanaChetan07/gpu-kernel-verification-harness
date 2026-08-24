"""Tests for the mutation engine.

Everything here runs on CPU with no network and no GPU. The two execution tests
really do spawn sandbox subprocesses and really do compare tensors, because the
claim under test -- "a task ships only if a witness was executed" -- cannot be
verified by a mock.

The fixtures are inline on purpose: this module must not depend on the real seed
modules existing, and the twelve-site source is easier to reason about when the
pattern each class is supposed to find is visible next to the assertion.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from crucible.capabilities import Capabilities
from crucible.config import Config
from crucible.mutate import astutil
from crucible.mutate.astutil import (
    Site,
    SiteError,
    apply_unified_diff,
    canonical,
    ensure_trailing_newline,
    parse,
    unified_diff,
    unparse,
)
from crucible.mutate.classes import ALL_CLASSES, MUTATION_CLASSES, class_ids, get, resolve_classes
from crucible.mutate.engine import (
    PromptLeak,
    assert_no_leak,
    build_prompt,
    default_rubric,
    generate,
    oracles_for,
    refine_tier,
)
from crucible.mutate.witness import derive_tolerance, search, short_dtype
from crucible.schema import ShapeSpec, Task
from crucible.seeds.registry import SeedSpec

# --------------------------------------------------------------------------- #
# a source carrying one site for each of the twelve classes
# --------------------------------------------------------------------------- #

TWELVE_SITE_SOURCE = '''import torch
import torch.distributed as dist

CONFIG_CACHE = {}


def blocked_reduce(x, block=32):
    """Blocked reduction with an explicit accumulator and an explicit tail guard."""
    n = x.shape[-1]
    n_blocks = (n + block - 1) // block
    partials = torch.zeros(n_blocks, dtype=torch.float32, device=x.device)
    acc = torch.zeros(x.shape[:-1], dtype=torch.float32, device=x.device)
    for b in range(n_blocks):
        start = b * block
        stop = min(start + block, n)
        if stop - start == 0:
            continue
        offs = torch.arange(start, stop, device=x.device)
        mask = offs < n
        idx = offs[mask]
        chunk = x.index_select(-1, idx).to(torch.float32)
        partial = chunk.sum(dim=-1)
        partials[b] = partial.sum()
        rescale = float(block) / float(max(stop - start, 1))
        acc = acc * rescale + partial
    return acc, partials


def store_paths(x, flag, out):
    y = x * 2.0
    if flag:
        out.index_copy_(0, torch.arange(y.shape[0]), y)
    else:
        out[:] = y + 1.0
    return out


def ping_pong(buf, steps):
    for _ in range(steps):
        prev = buf.clone()
        buf[1:] = prev[:-1] + prev[1:]
    return buf


def scale_view(x, factor):
    view = x.transpose(0, 1)
    view.mul_(factor)
    return x


def clip_grads(grads, max_norm):
    sq_norm = torch.zeros((), dtype=torch.float32)
    for g in grads:
        sq_norm = sq_norm + g.pow(2).sum()
    dist.all_reduce(sq_norm)
    total_norm = sq_norm.sqrt()
    clip = max_norm / (total_norm + 1e-06)
    for g in grads:
        g.mul_(torch.clamp(clip, max=1.0))
    return total_norm


def sync_step(tensor, do_extra):
    dist.all_reduce(tensor)
    if do_extra:
        tensor = tensor * 0.5
    else:
        tensor = tensor + 1.0
    return tensor


def load_optimizer_state(state):
    restored = {}
    for name, tensor in state.items():
        restored[name] = tensor.to(torch.float32)
    return restored


def pick_config(m, n, dtype, table):
    key = (m, n, str(dtype))
    if key not in CONFIG_CACHE:
        CONFIG_CACHE[key] = table[min(m, len(table) - 1)]
    return CONFIG_CACHE[key]
'''


# --------------------------------------------------------------------------- #
# an executable seed whose tail guard is the mutation target
# --------------------------------------------------------------------------- #

# Correct on any length that is a whole number of blocks; the tail block is the
# only place the mask matters. Relaxing `offs < n` to `offs <= n` therefore
# double-counts element 0 (via the modulo) exactly when n % block != 0.
WITNESS_SOURCE = '''import torch


def rowsum(x, block=8):
    n = x.shape[-1]
    acc = torch.zeros(x.shape[:-1], dtype=torch.float32, device=x.device)
    start = 0
    while start < n:
        offs = torch.arange(start, start + block, device=x.device)
        mask = offs < n
        idx = offs[mask] % n
        acc = acc + x.index_select(-1, idx).to(torch.float32).sum(dim=-1)
        start = start + block
    return acc
'''

FULL_SWEEP = [
    ShapeSpec(name="m2n8", kwargs={"rows": 2, "cols": 8, "dtype": "float32"}),
    ShapeSpec(name="m2n16", kwargs={"rows": 2, "cols": 16, "dtype": "float32"}),
    ShapeSpec(name="m2n20", kwargs={"rows": 2, "cols": 20, "dtype": "float32"}),
    ShapeSpec(name="m2n24", kwargs={"rows": 2, "cols": 24, "dtype": "float32"}),
]


def _make_inputs(shape: ShapeSpec, device: Any = "cpu", generator: Any = None) -> dict[str, Any]:
    import torch

    rows = int(shape.kwargs["rows"])
    cols = int(shape.kwargs["cols"])
    base = torch.arange(rows * cols, dtype=torch.float32, device=device)
    return {"x": (base % 7.0 + 1.0).reshape(rows, cols)}


def _reference(x: Any) -> Any:
    import torch

    return torch.sum(x.to(torch.float32), dim=-1)


def witness_seed(shapes: list[ShapeSpec] | None = None) -> SeedSpec:
    return SeedSpec(
        id="synthetic.rowsum",
        domain="pytorch",
        tiers=("T2", "T5"),
        description="Blocked row-sum whose tail mask is the mutation target.",
        entry="rowsum",
        source=WITNESS_SOURCE,
        make_inputs=_make_inputs,
        reference=_reference,
        shape_sweep=list(FULL_SWEEP if shapes is None else shapes),
        accum_depth=lambda s: int(s.kwargs["cols"]),
        bytes_moved=lambda s: int(s.kwargs["rows"]) * int(s.kwargs["cols"]) * 4,
        flops=lambda s: int(s.kwargs["rows"]) * int(s.kwargs["cols"]),
        denylist=("torch.sum",),
        supports_cpu=True,
        module="tests.test_mutate",
    )


@pytest.fixture
def tree_and_src() -> tuple[ast.Module, str]:
    src = canonical(TWELVE_SITE_SOURCE)
    return parse(src), src


# --------------------------------------------------------------------------- #
# astutil
# --------------------------------------------------------------------------- #


def test_roundtrip_reparses_and_is_idempotent() -> None:
    once = astutil.roundtrip(TWELVE_SITE_SOURCE)
    parse(once)  # must not raise
    assert astutil.roundtrip(once) == once
    assert astutil.check_roundtrip(TWELVE_SITE_SOURCE)
    assert canonical(TWELVE_SITE_SOURCE).endswith("\n")


def test_site_ordinal_disambiguates_nodes_at_the_same_position() -> None:
    # `x[0][0]`: the outer Subscript and its value start at the same column.
    tree = parse("y = x[0][0]\n")
    subs = [n for n in ast.walk(tree) if isinstance(n, ast.Subscript)]
    assert len(subs) == 2
    sites = [astutil.site_for(tree, n) for n in subs]
    assert sites[0].lineno == sites[1].lineno
    assert sites[0].col_offset == sites[1].col_offset
    assert {s.ordinal for s in sites} == {0, 1}
    for site, node in zip(sites, subs):
        resolved = astutil.resolve(tree, site)
        assert resolved is not None
        assert unparse(resolved) == unparse(node)


def test_resolve_returns_none_for_a_foreign_site() -> None:
    tree = parse("y = 1\n")
    assert astutil.resolve(tree, Site(lineno=99, col_offset=0, node_type="Call", ordinal=0)) is None
    with pytest.raises(SiteError):
        astutil.require(tree, Site(lineno=99, col_offset=0, node_type="Call", ordinal=0))


def test_apply_to_copy_leaves_the_original_tree_untouched(tree_and_src) -> None:
    tree, src = tree_and_src
    cls = get("boundary_mask")
    site = cls.sites(tree, src)[0]
    mutated = cls.apply(tree, site)
    assert unparse(mutated) != unparse(tree)
    assert ensure_trailing_newline(unparse(tree)) == src


def test_unified_diff_applier_handles_change_insert_and_delete() -> None:
    a = "one\ntwo\nthree\nfour\nfive\n"
    for b in (
        "one\ntwo\nCHANGED\nfour\nfive\n",
        "one\ntwo\nthree\nextra\nfour\nfive\n",
        "one\ntwo\nfour\nfive\n",
        "one\ntwo\nthree\nfour\nfive\nsix\n",
        "prefix\none\ntwo\nthree\nfour\nfive\n",
    ):
        diff = unified_diff(a, b)
        assert apply_unified_diff(a, diff) == b


def test_unified_diff_applier_rejects_a_patch_that_does_not_apply() -> None:
    diff = unified_diff("one\ntwo\n", "one\nTWO\n")
    with pytest.raises(SiteError):
        apply_unified_diff("completely\ndifferent\n", diff)


# --------------------------------------------------------------------------- #
# the twelve classes
# --------------------------------------------------------------------------- #


def test_registry_holds_exactly_the_twelve_contract_classes() -> None:
    assert set(class_ids()) == {
        "boundary_mask",
        "accum_dtype",
        "shmem_sizing",
        "missing_barrier",
        "cross_block_reduction",
        "empty_input_guard",
        "uninit_output",
        "layout_conflict",
        "grad_norm_scope",
        "collective_ordering",
        "resume_fidelity",
        "autotune_staleness",
    }
    assert len(ALL_CLASSES) == 12
    assert all(c.description and c.tier for c in ALL_CLASSES)
    assert [c.id for c in resolve_classes(None)] == class_ids()
    assert [c.id for c in resolve_classes(["accum_dtype"])] == ["accum_dtype"]


@pytest.mark.parametrize("cls_id", sorted(MUTATION_CLASSES))
def test_every_class_finds_a_site_on_the_fixture(cls_id: str, tree_and_src) -> None:
    tree, src = tree_and_src
    found = get(cls_id).sites(tree, src)
    assert found, f"{cls_id} found no site in the twelve-site fixture"
    assert all(isinstance(s, Site) for s in found)
    assert found == sorted(found, key=lambda s: (s.lineno, s.col_offset, s.ordinal))


@pytest.mark.parametrize("cls_id", sorted(MUTATION_CLASSES))
def test_apply_produces_a_different_program_that_still_parses(cls_id: str, tree_and_src) -> None:
    tree, src = tree_and_src
    cls = get(cls_id)
    site = cls.sites(tree, src)[0]
    mutant = ensure_trailing_newline(unparse(cls.apply(tree, site)))
    parse(mutant)  # must not raise
    assert mutant != src, f"{cls_id} produced a textually identical program"
    assert astutil.check_roundtrip(mutant)


@pytest.mark.parametrize("cls_id", sorted(MUTATION_CLASSES))
def test_ground_truth_diff_reproduces_the_baseline_exactly(cls_id: str, tree_and_src) -> None:
    tree, src = tree_and_src
    cls = get(cls_id)
    site = cls.sites(tree, src)[0]
    mutant = ensure_trailing_newline(unparse(cls.apply(tree, site)))
    diff = unified_diff(mutant, src, fromfile="mutant", tofile="baseline")
    assert diff.strip(), "an admitted mutation must produce a non-empty diff"
    assert apply_unified_diff(mutant, diff) == src


def test_class_specific_edits_are_what_the_contract_says(tree_and_src) -> None:
    tree, src = tree_and_src

    def mutate(cls_id: str) -> str:
        cls = get(cls_id)
        return unparse(cls.apply(tree, cls.sites(tree, src)[0]))

    assert "mask = offs <= n" in mutate("boundary_mask")
    assert "dtype=torch.float16" in mutate("accum_dtype")
    assert "torch.zeros(32, dtype=torch.float32" in mutate("shmem_sizing")
    assert "prev = buf\n" in mutate("missing_barrier") + "\n"
    assert "acc = partial" in mutate("cross_block_reduction")
    assert "if stop - start == 0" not in mutate("empty_input_guard")
    assert "out[:] = y + 1.0" not in mutate("uninit_output")
    assert ".contiguous().expand_as(" in mutate("layout_conflict")
    assert "dist.all_reduce(sq_norm)" not in mutate("grad_norm_scope")
    assert "torch.bfloat16" in mutate("resume_fidelity")
    assert "key = (str(dtype),)" in mutate("autotune_staleness")

    ordered = mutate("collective_ordering")
    # The collective now lives on one branch only.
    assert "if do_extra:\n        dist.all_reduce(tensor)" in ordered


def test_uninit_output_removes_the_store_on_exactly_one_path(tree_and_src) -> None:
    tree, src = tree_and_src
    cls = get("uninit_output")
    mutant = unparse(cls.apply(tree, cls.sites(tree, src)[0]))
    assert "out.index_copy_" in mutant  # the surviving path still stores
    assert "out[:] = y + 1.0" not in mutant  # the other path does not


def test_sites_return_empty_when_the_pattern_is_absent() -> None:
    for source in ("", '"""just a docstring."""\n', "x = 1\n", "def f():\n    return 1\n"):
        tree = parse(source)
        for cls in ALL_CLASSES:
            assert cls.sites(tree, source) == [], f"{cls.id} hallucinated a site in {source!r}"


def test_sites_never_raises_even_on_exotic_or_broken_detectors() -> None:
    exotic = (
        "async def f(a):\n"
        "    return [await x async for x in a if (y := x) > 0]\n"
        "\n"
        "g = lambda *a, **k: {**k, 'z': [*a]}\n"
        "\n"
        "def h(v):\n"
        "    match v:\n"
        "        case {'a': [1, *rest]}:\n"
        "            return rest\n"
        "        case _:\n"
        "            return None\n"
    )
    tree = parse(exotic)
    for cls in ALL_CLASSES:
        cls.sites(tree, exotic)  # must not raise

    class Exploding(type(get("boundary_mask"))):  # type: ignore[misc]
        def _sites(self, tree: ast.AST, src: str) -> list[Site]:
            raise RuntimeError("detector defect")

    assert Exploding().sites(parse("x = 1\n"), "x = 1\n") == []


def test_apply_raises_site_error_for_a_site_of_the_wrong_shape() -> None:
    tree = parse("x = 1\n")
    bogus = Site(lineno=1, col_offset=4, node_type="Constant", ordinal=0)
    with pytest.raises(SiteError):
        get("boundary_mask").apply(tree, bogus)


# --------------------------------------------------------------------------- #
# tolerance
# --------------------------------------------------------------------------- #


def test_tolerance_is_derived_named_and_scales_with_depth(cfg: Config) -> None:
    shallow = derive_tolerance("float32", 1, cfg)
    deep = derive_tolerance("float32", 10_000, cfg)
    assert deep.rel > shallow.rel
    assert shallow.formula, "a numeric budget with no derivation is not evidence"
    assert shallow.source
    assert derive_tolerance("bfloat16", 16, cfg).rel > derive_tolerance("float32", 16, cfg).rel
    assert short_dtype("torch.bfloat16") == "bf16"
    # An unknown dtype must widen, never tighten: a tight guess invents witnesses.
    assert derive_tolerance("mystery9", 16, cfg).rel >= derive_tolerance("float32", 16, cfg).rel


# --------------------------------------------------------------------------- #
# witness search (real sandbox execution)
# --------------------------------------------------------------------------- #


def _mutant_of(source: str, cls_id: str) -> str:
    src = canonical(source)
    tree = parse(src)
    cls = get(cls_id)
    sites = cls.sites(tree, src)
    assert sites, f"{cls_id} found no site in the witness seed"
    return ensure_trailing_newline(unparse(cls.apply(tree, sites[0])))


def test_witness_search_finds_the_first_breaking_shape(
    caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    seed = witness_seed()
    mutant = _mutant_of(WITNESS_SOURCE, "boundary_mask")
    result = search(seed, mutant, seed.shape_sweep, caps, cfg, workdir=tmp_workdir)

    assert result.discarded_reason is None
    assert result.witness is not None
    assert result.admitted
    # 8, 16 and 24 are whole numbers of blocks, so the tail mask never matters
    # there; only 20 has a partial tail. The witness is still the FIRST
    # breaking shape, but the sweep now runs to completion so that 24 is
    # classified too -- a decoy on the far side of the witness is exactly the
    # difficulty dial, and stopping at the first break never finds it.
    assert result.witness.shape.name == "m2n20"
    assert result.witness.kind == "numeric"
    assert [s.name for s in result.decoys] == ["m2n8", "m2n16", "m2n24"]
    assert [s.name for s in result.detects] == ["m2n20"]
    assert result.n_evaluated == 4, "the whole sweep must be classified"
    assert result.truncated is None
    assert result.witness.max_rel_err > result.witness.tolerance > 0.0
    assert result.witness.baseline_checksum and result.witness.mutant_checksum
    assert result.witness.baseline_checksum != result.witness.mutant_checksum
    # The reported budget is the one the witness was actually judged against.
    assert result.tolerance is not None
    assert result.tolerance.accum_depth == 20
    assert result.tolerance.formula in result.witness.detail
    assert result.witness.tolerance == pytest.approx(result.tolerance.rel)
    # The tolerance reported on the result is the witness's own budget, not
    # the last shape swept, even though the sweep continued past it.
    assert result.n_evaluated == 4


def test_semantically_neutral_mutation_is_discarded(
    caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    # Narrowing the accumulator is defeated by torch type promotion here: after
    # the first `acc + <fp32>` the accumulator is fp32 again, so nothing changes.
    seed = witness_seed(shapes=FULL_SWEEP[:2])
    mutant = _mutant_of(WITNESS_SOURCE, "accum_dtype")
    assert "dtype=torch.float16" in mutant

    result = search(seed, mutant, seed.shape_sweep, caps, cfg, workdir=tmp_workdir)

    assert result.witness is None
    assert not result.admitted
    assert result.discarded_reason is not None
    assert "neutral" in result.discarded_reason
    assert result.n_evaluated == 2
    assert [s.name for s in result.decoys] == ["m2n8", "m2n16"]
    assert result.detects == []


def test_a_raising_mutant_is_an_exception_witness(
    caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    seed = witness_seed(shapes=FULL_SWEEP[:1])
    mutant = "import torch\n\n\ndef rowsum(x, block=8):\n    raise RuntimeError('boom')\n"
    result = search(seed, mutant, seed.shape_sweep, caps, cfg, workdir=tmp_workdir)
    assert result.witness is not None
    assert result.witness.kind == "exception"
    assert "boom" in result.witness.detail
    assert result.decoys == []


def test_a_wrong_shaped_mutant_is_a_shape_witness(
    caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    seed = witness_seed(shapes=FULL_SWEEP[:1])
    mutant = (
        "import torch\n\n\ndef rowsum(x, block=8):\n"
        "    return torch.zeros(x.shape[0], 2, dtype=torch.float32)\n"
    )
    result = search(seed, mutant, seed.shape_sweep, caps, cfg, workdir=tmp_workdir)
    assert result.witness is not None
    assert result.witness.kind == "shape"


def test_a_seed_that_cannot_run_here_is_discarded_not_passed(
    caps: Capabilities, cfg: Config, tmp_workdir: Path
) -> None:
    seed = witness_seed(shapes=FULL_SWEEP[:1])
    seed.supports_cpu = False
    result = search(seed, WITNESS_SOURCE, seed.shape_sweep, caps, cfg, workdir=tmp_workdir)
    assert result.witness is None
    assert result.discarded_reason is not None
    assert "untested-by-construction" in result.discarded_reason
    assert result.n_evaluated == 0


def test_a_seed_with_no_sweep_is_discarded(caps: Capabilities, cfg: Config) -> None:
    seed = witness_seed(shapes=[])
    result = search(seed, WITNESS_SOURCE, [], caps, cfg)
    assert result.witness is None
    assert "untested-by-construction" in (result.discarded_reason or "")


# --------------------------------------------------------------------------- #
# prompt hygiene and task assembly helpers
# --------------------------------------------------------------------------- #


def test_oracle_selection_follows_tier_and_domain() -> None:
    assert oracles_for("T5", "pytorch") == ["O1", "O3", "O4"]
    assert "O2" in oracles_for("T4", "cuda")
    assert "O5" in oracles_for("T6", "pytorch")
    assert "O5" in oracles_for("T3", "distributed")
    assert oracles_for("T5", "data_pipeline") == ["O1", "O3"]


def test_tier_records_the_observed_failure_not_the_intended_one() -> None:
    assert refine_tier("T5", "numeric") == "T5"
    assert refine_tier("T5", "exception") == "T2"
    assert refine_tier("T5", "hang") == "T2"
    assert refine_tier("T5", "shape") == "T3"
    assert refine_tier("T4", "loss_curve") == "T6"


def test_default_rubric_machine_probes_what_an_oracle_measures() -> None:
    rubric = default_rubric(None, "T5", "pytorch")
    probed = [c for c in rubric.criteria if c.machine_probed]
    assert probed, "objective criteria must not be left to human variance"
    assert all(c.auto_probe for c in probed)
    assert rubric.machine_probed_share() >= 0.5
    assert "O5" not in str(rubric.model_dump())
    distributed = default_rubric(None, "T6", "distributed")
    assert any(c.id == "multi_rank_agreement" for c in distributed.criteria)


def test_prompt_carries_the_decoys_and_hides_everything_else() -> None:
    seed = witness_seed()
    mutant = _mutant_of(WITNESS_SOURCE, "boundary_mask")
    decoys = FULL_SWEEP[:2]
    prompt = build_prompt(seed, mutant, decoys, "pytorch")
    assert "m2n8" in prompt and "m2n16" in prompt
    assert "m2n20" not in prompt
    assert "boundary_mask" not in prompt
    assert "offs < n" not in prompt  # the baseline's version of the mutated line
    assert "@@" not in prompt
    assert seed.entry in prompt


def test_leak_check_rejects_a_prompt_that_names_the_class_or_a_detect_shape() -> None:
    diff = "--- a\n+++ b\n@@ -1 +1 @@\n-bad\n+good enough line\n"
    with pytest.raises(PromptLeak):
        assert_no_leak("hint: boundary_mask", "boundary_mask", [], "", "", "")
    with pytest.raises(PromptLeak):
        assert_no_leak("graded on m2n20", "x", [ShapeSpec(name="m2n20")], "", "", "")
    with pytest.raises(PromptLeak):
        assert_no_leak("here it is\ngood enough line\n", "x", [], diff, "", "bad\n")
    # The same line present in the mutant is not a leak: the prompt embeds it.
    assert_no_leak("here it is\ngood enough line\n", "x", [], diff, "", "good enough line\n")


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


def test_generate_admits_a_proven_task_and_discards_a_neutral_one(
    caps: Capabilities, cfg: Config, tmp_path: Path
) -> None:
    seed = witness_seed()
    report = generate(
        [seed],
        ["boundary_mask", "accum_dtype"],
        caps,
        cfg,
        out_dir=tmp_path / "bank",
        workdir=tmp_path / "work",
    )

    admitted = report.admitted()
    discarded = report.discarded()
    assert len(admitted) == 1 and len(report.tasks) == 1
    assert admitted[0].cls == "boundary_mask"
    assert len(discarded) == 1
    assert discarded[0].cls == "accum_dtype"
    assert "neutral" in (discarded[0].discarded_reason or "")

    task = report.tasks[0]
    assert task.task_id.startswith("mut-rowsum-boundary_mask-")
    assert len(task.task_id.rsplit("-", 1)[1]) == 4

    # The recorded fix really is the fix.
    assert apply_unified_diff(task.mutant_code, task.ground_truth_diff) == task.baseline_code
    assert task.mutant_code != task.baseline_code
    assert "offs <= n" in task.mutant_code and "offs < n" in task.baseline_code

    assert task.witness is not None and task.witness.shape.name == "m2n20"
    assert task.failure_tier == "T5" and task.domain == "pytorch"
    assert [s.name for s in task.detect_shapes] == ["m2n20"]
    # Includes m2n24, which lies past the witness: a full sweep finds decoys on
    # both sides of the breaking shape, so the prompt cannot be solved by
    # noticing that every supplied shape is smaller than the failing one.
    assert [s.name for s in task.decoy_shapes] == ["m2n8", "m2n16", "m2n24"]
    assert task.oracles == ["O1", "O3", "O4"]
    assert task.mutation.cls == "boundary_mask"
    assert task.mutation.site.startswith("synthetic.rowsum:Compare@")
    assert task.provenance["shapes_evaluated"] == 4
    assert task.provenance["tolerance"]["formula"]

    # The prompt withholds the answer.
    assert "boundary_mask" not in task.prompt
    assert "m2n20" not in task.prompt
    assert "offs < n" not in task.prompt
    assert task.ground_truth_diff not in task.prompt

    # And the whole thing survives a save/load round trip.
    saved = Path(admitted[0].path or "")
    assert saved.exists()
    reloaded = Task.load(saved)
    assert reloaded.task_id == task.task_id
    assert reloaded.ground_truth_diff == task.ground_truth_diff
    assert apply_unified_diff(reloaded.mutant_code, reloaded.ground_truth_diff) == reloaded.baseline_code


def test_generate_skips_classes_with_no_site_without_executing_anything(
    caps: Capabilities, cfg: Config, tmp_path: Path
) -> None:
    seed = witness_seed(shapes=FULL_SWEEP[:1])
    report = generate(
        [seed],
        ["grad_norm_scope", "collective_ordering", "resume_fidelity"],
        caps,
        cfg,
        workdir=tmp_path / "work",
    )
    assert report.outcomes == []
    assert report.tasks == []
    assert report.as_dict()["n_admitted"] == 0
