"""Tests for the four kernel-domain seeds.

The load-bearing test in this file is ``test_source_matches_reference``: every
seed, on every shape in its adversarial sweep, must agree with its *independent*
reference within a tolerance derived from the dtype and the true reduction
depth. A seed that cannot pass that is a broken baseline, and every task minted
from it would be minted from a lie -- so the fix is always to the seed, never to
the tolerance.

Everything else here defends that test: that the reference really is a different
code path, that the sweep really is adversarial, that the source text really
contains the structure the mutation engine will address, and that the padded
region of every input really is unread.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Callable, Iterator

import numpy as np
import pytest
import torch

from crucible.schema import ShapeSpec
from crucible.seeds import registry
from crucible.seeds.registry import SeedSpec

KERNEL_SEED_IDS = (
    "attention.blocked_fwd",
    "reduction.blocked_layernorm",
    "matmul.tiled",
    "quant.int4_dequant_matmul",
)

EXPECTED = {
    "attention.blocked_fwd": ("triton", "blocked_attention_fwd", "crucible.seeds.attention"),
    "reduction.blocked_layernorm": ("triton", "blocked_layernorm_fwd", "crucible.seeds.reduction"),
    "matmul.tiled": ("cuda", "tiled_matmul", "crucible.seeds.matmul"),
    "quant.int4_dequant_matmul": ("cuda", "int4_dequant_matmul", "crucible.seeds.quant"),
}


def _seed(seed_id: str) -> SeedSpec:
    return registry.get(seed_id)


def _all_kernel_seeds() -> list[SeedSpec]:
    return [_seed(sid) for sid in KERNEL_SEED_IDS]


def _entry_fn(seed: SeedSpec) -> Callable[..., Any]:
    """The live baseline function. Its text *is* ``seed.source`` by construction.

    The seed modules build ``source`` with ``inspect.getsource`` on this very
    function, so calling it in-process tests exactly the text a candidate is
    handed. Untrusted candidate code still goes through the sandbox; this is the
    seed's own known-good baseline.
    """
    import importlib

    module = importlib.import_module(EXPECTED[seed.id][2])
    fn = getattr(module, seed.entry)
    assert callable(fn)
    return fn


# --------------------------------------------------------------------------- #
# tolerance
# --------------------------------------------------------------------------- #

#: Unit roundoff per dtype. Mirrors CONTRACT.md section 7 (`oracles/tolerance.py`)
#: so this file still has a defensible number before that module lands.
_EPS = {"float32": 2.0**-24, "bfloat16": 2.0**-8, "float16": 2.0**-11}
_SHORT = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}

_TOLERANCE_SAFETY = 4.0
_TOLERANCE_MODE = "stochastic"


def _derive_tolerance(dtype_name: str, depth: int) -> tuple[float, str]:
    """Relative tolerance for a reduction ``depth`` deep in ``dtype_name``.

    Uses ``crucible.oracles.tolerance`` when it exists so this file cannot drift
    from the real error-budget model. The fallback is the documented conservative
    default: ``rel = safety * sqrt(depth) * eps(dtype)`` -- stochastic rounding
    growth, which is the model the config defaults to.
    """
    depth = max(1, int(depth))
    try:
        from crucible.oracles import tolerance as tol_mod  # type: ignore[attr-defined]
    except ImportError:
        tol_mod = None
    derive = getattr(tol_mod, "derive", None) if tol_mod is not None else None
    if derive is not None:
        for key in (_SHORT[dtype_name], dtype_name):
            try:
                t = derive(key, depth, mode=_TOLERANCE_MODE, safety=_TOLERANCE_SAFETY)
            except (KeyError, ValueError, TypeError):
                continue
            return float(t.rel), f"tolerance.derive({key!r}, {depth}): {getattr(t, 'formula', '')}"
    eps = _EPS[dtype_name]
    growth = math.sqrt(depth) if _TOLERANCE_MODE == "stochastic" else float(depth)
    rel = _TOLERANCE_SAFETY * growth * eps
    formula = (
        f"fallback rel = safety({_TOLERANCE_SAFETY}) * sqrt(depth={depth}) "
        f"* eps({_SHORT[dtype_name]})={eps:.3e} = {rel:.3e}"
    )
    return rel, formula


def _errors(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float]:
    """(max absolute error, magnitude scale of the reference)."""
    g = got.detach().to(torch.float32)
    w = want.detach().to(torch.float32)
    assert g.shape == w.shape, f"shape {tuple(g.shape)} != reference {tuple(w.shape)}"
    if g.numel() == 0:
        return 0.0, 1.0
    abs_err = float((g - w).abs().max())
    scale = max(float(w.abs().max()), 1e-6)
    return abs_err, scale


def _cases() -> Iterator[Any]:
    for seed in _all_kernel_seeds():
        for shape in seed.shape_sweep:
            yield pytest.param(seed, shape, id=f"{seed.short_id()}-{shape.name}")


# --------------------------------------------------------------------------- #
# registration and identity
# --------------------------------------------------------------------------- #


def test_four_kernel_seeds_are_discoverable() -> None:
    ids = registry.seed_ids()
    for sid in KERNEL_SEED_IDS:
        assert sid in ids, f"{sid} missing; registry has {ids}; import errors: {registry.import_errors()}"
    for sid, (domain, entry, module) in EXPECTED.items():
        seed = _seed(sid)
        assert seed.domain == domain
        assert seed.entry == entry
        assert seed.module == module
        assert seed.tiers, f"{sid} advertises no tier"
        assert seed.supports_cpu is True


def test_no_seed_module_failed_to_import() -> None:
    errors = {k: v for k, v in registry.import_errors().items() if k.split(".")[-1] in
              ("attention", "reduction", "matmul", "quant")}
    assert errors == {}, f"kernel seed modules failed to import: {errors}"


@pytest.mark.parametrize("seed_id", KERNEL_SEED_IDS)
def test_source_is_standalone_and_defines_entry(seed_id: str) -> None:
    seed = _seed(seed_id)
    # compile() parses and byte-compiles without executing anything.
    code = compile(seed.source, f"<{seed_id}>", "exec")
    assert code is not None
    tree = ast.parse(seed.source)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert seed.entry in defined, f"{seed_id}: source does not define {seed.entry}"
    imported = {
        n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)
    }
    # The baseline must be pure torch: no numpy, no triton, no seed-package import.
    assert imported <= {"torch"}, f"{seed_id}: source imports {imported}, expected only torch"
    assert seed.content_sha256() == seed.content_sha256()


@pytest.mark.parametrize("seed_id", KERNEL_SEED_IDS)
def test_reference_is_an_independent_code_path(seed_id: str) -> None:
    """The reference must not be the baseline, and must reach a fused op."""
    import inspect

    seed = _seed(seed_id)
    ref = seed.reference
    assert ref is not _entry_fn(seed)
    assert getattr(ref, "__name__", "") != seed.entry
    ref_src = inspect.getsource(ref)
    fused = {
        "attention.blocked_fwd": "scaled_dot_product_attention",
        "reduction.blocked_layernorm": "layer_norm",
        "matmul.tiled": "torch.matmul",
        "quant.int4_dequant_matmul": "torch.matmul",
    }[seed_id]
    assert fused in ref_src, f"{seed_id}: reference does not call {fused}"
    # ... and the baseline must not, or the comparison would be circular.
    assert fused not in seed.source, f"{seed_id}: baseline calls the reference op {fused}"


# --------------------------------------------------------------------------- #
# structure the mutation engine addresses
# --------------------------------------------------------------------------- #


def _entry_ast(seed: SeedSpec) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(seed.source)):
        if isinstance(node, ast.FunctionDef) and node.name == seed.entry:
            return node
    raise AssertionError(f"{seed.id}: no def {seed.entry}")


def _calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            func = n.func
            if isinstance(func, ast.Attribute) and func.attr == name:
                out.append(n)
            elif isinstance(func, ast.Name) and func.id == name:
                out.append(n)
    return out


@pytest.mark.parametrize("seed_id", KERNEL_SEED_IDS)
def test_source_carries_the_kernel_structure(seed_id: str) -> None:
    seed = _seed(seed_id)
    fn = _entry_ast(seed)
    src = seed.source

    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)]
    assert loops, f"{seed_id}: no block loop"

    # an offsets vector and a boundary mask `offs < N`
    assert _calls_named(fn, "arange"), f"{seed_id}: no offsets vector"
    lt_compares = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.Lt)
    ]
    assert lt_compares, f"{seed_id}: no strict-less-than boundary mask"

    # an explicit fp32 accumulator
    fp32_zeros = [
        call
        for call in _calls_named(fn, "zeros")
        for kw in call.keywords
        if kw.arg == "dtype" and "float32" in ast.unparse(kw.value)
    ]
    assert fp32_zeros, f"{seed_id}: no torch.zeros(..., dtype=torch.float32) accumulator"

    # an explicit zero-length-block guard
    guards = [
        n
        for loop in loops
        for n in ast.walk(loop)
        if isinstance(n, ast.If) and any(isinstance(b, ast.Continue) for b in n.body)
    ]
    assert guards, f"{seed_id}: no `if ...: continue` empty-block guard"

    # an explicit clone/sync boundary inside a loop
    clones = [c for loop in loops for c in _calls_named(loop, "clone")]
    assert clones, f"{seed_id}: no clone boundary inside the block loop"

    # a layout / replication choice
    assert _calls_named(fn, "contiguous") or _calls_named(fn, "expand"), (
        f"{seed_id}: no explicit contiguous()/expand() layout choice"
    )

    # an explicit store on every path: a zero_() for the degenerate path plus a
    # write into the output buffer on the main path.
    assert _calls_named(fn, "zero_"), f"{seed_id}: no explicit store on the degenerate path"
    stores = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript) for t in n.targets)
    ]
    assert stores or _calls_named(fn, "copy_"), f"{seed_id}: no explicit store on the main path"

    # a lane-indexed staging buffer
    assert "lane" in src, f"{seed_id}: no lane index"


@pytest.mark.parametrize("seed_id", KERNEL_SEED_IDS)
def test_denylist_is_dotted_and_does_not_ban_the_baseline(seed_id: str) -> None:
    seed = _seed(seed_id)
    assert seed.denylist, f"{seed_id}: empty denylist"
    for symbol in seed.denylist:
        assert "." in symbol, f"{seed_id}: denylist entry {symbol!r} is not a dotted path"
        assert " " not in symbol.strip(), f"{seed_id}: denylist entry {symbol!r} is not a symbol"
    # Every attribute chain the baseline itself uses must be legal.
    used: set[str] = set()
    for node in ast.walk(ast.parse(seed.source)):
        if isinstance(node, ast.Attribute):
            try:
                used.add(ast.unparse(node))
            except (ValueError, AttributeError):  # pragma: no cover - unparse is total here
                continue
    banned = sorted(set(seed.denylist) & used)
    assert banned == [], f"{seed_id}: baseline itself uses denylisted symbols {banned}"
    # The seed's own reference must be unreachable by import.
    assert any(s.startswith("crucible.seeds.") for s in seed.denylist), (
        f"{seed_id}: denylist does not close the reference-import route"
    )


# --------------------------------------------------------------------------- #
# the sweep is adversarial
# --------------------------------------------------------------------------- #

_LENGTH_KEY = {
    "attention.blocked_fwd": "n_ctx",
    "reduction.blocked_layernorm": "n_valid",
    "matmul.tiled": "k_valid",
    "quant.int4_dequant_matmul": "k_valid",
}
_PAD_KEY = {
    "attention.blocked_fwd": "n_pad",
    "reduction.blocked_layernorm": "n_pad",
    "matmul.tiled": "k_pad",
    "quant.int4_dequant_matmul": "k_pad",
}


@pytest.mark.parametrize("seed_id", KERNEL_SEED_IDS)
def test_shape_sweep_is_adversarial(seed_id: str) -> None:
    seed = _seed(seed_id)
    sweep = seed.shape_sweep
    assert len(sweep) >= 12, f"{seed_id}: only {len(sweep)} shapes"
    names = [s.name for s in sweep]
    assert len(set(names)) == len(names), f"{seed_id}: duplicate shape names {names}"

    lengths = {int(s.kwargs[_LENGTH_KEY[seed_id]]) for s in sweep}
    required = {1, 127, 128, 129, 1023, 4096}
    assert required <= lengths, f"{seed_id}: sweep misses lengths {sorted(required - lengths)}"

    # a zero-length case, and a case where a whole block lies in the padding
    assert 0 in lengths, f"{seed_id}: no zero-length case"
    block_key = str(seed.extras.get("block_arg", "block_n"))
    has_empty_block = any(
        int(s.kwargs[_PAD_KEY[seed_id]]) - int(s.kwargs[_LENGTH_KEY[seed_id]])
        >= int(s.kwargs.get(block_key, 64))
        and int(s.kwargs[_LENGTH_KEY[seed_id]]) > 0
        for s in sweep
    )
    assert has_empty_block, f"{seed_id}: no shape leaves a whole block inside the padding"

    dtypes = {str(s.kwargs["dtype"]) for s in sweep}
    assert {"float32", "bfloat16", "float16"} <= dtypes, f"{seed_id}: dtypes {dtypes}"

    assert any(bool(s.kwargs.get("noncontig")) for s in sweep), f"{seed_id}: no non-contiguous case"

    # a single-element batch somewhere in the sweep
    lead_key = {"attention.blocked_fwd": "batch", "reduction.blocked_layernorm": "rows",
                "matmul.tiled": "m", "quant.int4_dequant_matmul": "m"}[seed_id]
    assert any(int(s.kwargs[lead_key]) == 1 for s in sweep), f"{seed_id}: no single-element batch"


def test_attention_sweep_covers_head_dim_at_and_above_block() -> None:
    seed = _seed("attention.blocked_fwd")
    pairs = {(int(s.kwargs["head_dim"]), int(s.kwargs["block_n"])) for s in seed.shape_sweep}
    assert any(d == b for d, b in pairs), "no head_dim exactly at the block size"
    assert any(d > b for d, b in pairs), "no head_dim above the block size"


# --------------------------------------------------------------------------- #
# analytic counters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed_id", KERNEL_SEED_IDS)
def test_counters_are_positive_and_grow_with_the_problem(seed_id: str) -> None:
    seed = _seed(seed_id)
    for shape in seed.shape_sweep:
        depth = seed.accum_depth(shape)
        assert isinstance(depth, int) and depth >= 1, f"{seed_id}/{shape.name}: depth {depth}"
        assert seed.bytes_moved(shape) > 0, f"{seed_id}/{shape.name}: bytes_moved <= 0"
        assert seed.flops(shape) >= 0, f"{seed_id}/{shape.name}: negative flops"

    length_key = _LENGTH_KEY[seed_id]
    graded = sorted(
        (s for s in seed.shape_sweep if str(s.kwargs["dtype"]) == "float32"),
        key=lambda s: int(s.kwargs[length_key]),
    )
    depths = [seed.accum_depth(s) for s in graded]
    assert depths == sorted(depths), f"{seed_id}: accum_depth is not monotone in the reduction length"
    deepest = max(graded, key=lambda s: int(s.kwargs[length_key]))
    assert seed.accum_depth(deepest) == int(deepest.kwargs[length_key])


def test_matmul_counters_match_the_documented_formula() -> None:
    seed = _seed("matmul.tiled")
    shape = seed.shape("k1023_fp32")
    m, n, k = 4, 4, 1023
    assert seed.flops(shape) == 2 * m * n * k
    assert seed.bytes_moved(shape) == 4 * (m * k + k * n + m * n)


def test_quant_weight_stream_is_half_a_byte_per_element() -> None:
    seed = _seed("quant.int4_dequant_matmul")
    shape = seed.shape("k1023_fp32")
    m, n, k = 4, 8, 1023
    n_groups = 1024 // 32
    assert seed.bytes_moved(shape) == 4 * (m * k + m * n) + (k * n) // 2 + 4 * (n_groups * n + n_groups)
    assert seed.flops(shape) == 2 * m * n * k + 2 * n * k


# --------------------------------------------------------------------------- #
# THE test: baseline == independent reference, everywhere in the sweep
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed,shape", list(_cases()))
def test_source_matches_reference(seed: SeedSpec, shape: ShapeSpec) -> None:
    fn = _entry_fn(seed)
    inputs = seed.make_inputs(shape, "cpu", None)
    got = fn(**inputs)
    want = seed.reference(**inputs)

    assert isinstance(got, torch.Tensor), f"{seed.id}/{shape.name}: baseline returned {type(got)}"
    assert got.dtype == want.dtype, f"{seed.id}/{shape.name}: {got.dtype} vs {want.dtype}"
    assert torch.isfinite(got.to(torch.float32)).all(), f"{seed.id}/{shape.name}: non-finite output"

    dtype_name = str(shape.kwargs["dtype"])
    rel, formula = _derive_tolerance(dtype_name, seed.accum_depth(shape))
    abs_err, scale = _errors(got, want)
    budget = rel * scale
    assert abs_err <= budget, (
        f"{seed.id}/{shape.name}: max_abs_err={abs_err:.3e} exceeds budget={budget:.3e} "
        f"(rel={rel:.3e} x scale={scale:.3e}); {formula}. "
        "The seed is wrong -- fix the baseline, do not widen the tolerance."
    )


@pytest.mark.parametrize("seed_id", KERNEL_SEED_IDS)
def test_make_inputs_is_reproducible(seed_id: str) -> None:
    seed = _seed(seed_id)
    shape = seed.shape_sweep[1]
    a = seed.make_inputs(shape, "cpu", None)
    b = seed.make_inputs(shape, "cpu", None)
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], torch.Tensor):
            assert torch.equal(a[key], b[key]), f"{seed_id}: input {key!r} is not reproducible"
        else:
            assert a[key] == b[key]


# --------------------------------------------------------------------------- #
# the boundary mask actually masks
# --------------------------------------------------------------------------- #

def _attention_padding(inputs: dict[str, Any], shape: ShapeSpec) -> list[torch.Tensor]:
    lo = int(shape.kwargs.get("k_start", 0))
    hi = lo + int(shape.kwargs["n_ctx"])
    views = [inputs["k"][..., hi:, :], inputs["v"][..., hi:, :]]
    if lo > 0:
        views += [inputs["k"][..., :lo, :], inputs["v"][..., :lo, :]]
    return views


_PADDED_SLICES: dict[str, Callable[[dict[str, Any], ShapeSpec], list[torch.Tensor]]] = {
    "attention.blocked_fwd": _attention_padding,
    "reduction.blocked_layernorm": lambda i, s: [
        i["x"][:, int(s.kwargs["n_valid"]) :],
        i["weight"][int(s.kwargs["n_valid"]) :],
        i["bias"][int(s.kwargs["n_valid"]) :],
    ],
    "matmul.tiled": lambda i, s: [
        i["a"][:, int(s.kwargs["k_valid"]) :],
        i["b"][int(s.kwargs["k_valid"]) :, :],
    ],
    "quant.int4_dequant_matmul": lambda i, s: [
        i["a"][:, int(s.kwargs["k_valid"]) :],
        i["b_packed"][int(s.kwargs["k_valid"]) :, :],
    ],
}

#: For attention this is the shape with slack on *both* sides of the window.
_PADDING_SHAPE = {
    "attention.blocked_fwd": "unaligned_window_fp32",
    "reduction.blocked_layernorm": "empty_block_fp32",
    "matmul.tiled": "empty_block_fp32",
    "quant.int4_dequant_matmul": "empty_block_fp32",
}


@pytest.mark.parametrize("seed_id", KERNEL_SEED_IDS)
def test_padding_never_reaches_the_result(seed_id: str) -> None:
    """Rewrite everything outside the logical window; the answer must not move.

    This is the property the boundary mask and the empty-block guard exist for,
    and it is the property an off-by-one in either of them destroys.
    """
    seed = _seed(seed_id)
    shape = seed.shape(_PADDING_SHAPE[seed_id])
    fn = _entry_fn(seed)

    inputs = seed.make_inputs(shape, "cpu", None)
    before = fn(**inputs).clone()

    generator = torch.Generator().manual_seed(20260813)
    for view in _PADDED_SLICES[seed_id](inputs, shape):
        assert view.numel() > 0, f"{seed_id}/{shape.name}: no padding to perturb"
        if view.dtype == torch.uint8:
            view.copy_(
                torch.randint(0, 256, view.shape, generator=generator, dtype=torch.uint8)
            )
        else:
            noise = torch.randn(view.shape, generator=generator, dtype=torch.float32) * 1e3
            view.copy_(noise.to(view.dtype))
    after = fn(**inputs)

    assert torch.equal(before, after), (
        f"{seed_id}/{shape.name}: perturbing the padded region changed the result by "
        f"{float((before.to(torch.float32) - after.to(torch.float32)).abs().max()):.3e}"
    )


# --------------------------------------------------------------------------- #
# the source text runs as a candidate module, out of process
# --------------------------------------------------------------------------- #

_SANDBOX_SHAPE = {
    "attention.blocked_fwd": "ctx127_fp32",
    "reduction.blocked_layernorm": "n127_fp32",
    "matmul.tiled": "k127_fp32",
    "quant.int4_dequant_matmul": "k127_fp32",
}


@pytest.mark.parametrize("seed_id", KERNEL_SEED_IDS)
def test_source_text_runs_in_the_sandbox(seed_id: str, tmp_workdir: Any) -> None:
    """The shipped text -- not the imported function -- must execute and be right."""
    from crucible.runner import sandbox

    seed = _seed(seed_id)
    shape = seed.shape(_SANDBOX_SHAPE[seed_id])
    inputs = seed.make_inputs(shape, "cpu", None)
    result = sandbox.call_entry(
        seed.source,
        seed.entry,
        inputs=inputs,
        workdir=tmp_workdir / seed.short_id(),
        device="cpu",
        timeout_s=180.0,
        call_style="kwargs",
    )
    assert result.ok, f"{seed_id}: sandbox run failed: {result.message}\n{result.traceback_text}"
    outputs = result.outputs()
    assert len(outputs) == 1, f"{seed_id}: expected one output tensor, got {len(outputs)}"
    record = outputs[0]
    assert record["kind"] == "array"
    got = torch.from_numpy(np.load(record["path"], allow_pickle=False))

    want = seed.reference(**inputs)
    rel, formula = _derive_tolerance(str(shape.kwargs["dtype"]), seed.accum_depth(shape))
    abs_err, scale = _errors(got, want)
    assert abs_err <= rel * scale, (
        f"{seed_id}/{shape.name} via sandbox: max_abs_err={abs_err:.3e} > {rel * scale:.3e}; {formula}"
    )
