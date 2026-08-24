"""O4 - compilation and portability: four sub-checks, each gated on its own capability.

A "compiles" claim is only worth what the compiler actually did. This oracle is
built so that the difference between *compiled and found nothing wrong* and
*never compiled at all* can never be lost:

1. ``graph_break`` - ``torch._dynamo.explain()`` enumerates the graphs, the
   graph breaks and the reason for each break, and ``torch.compile(...,
   fullgraph=True)`` is then run as an independent confirmation. Three outcomes
   are reported as three distinct strings - ``compiled_cleanly``,
   ``graph_breaks`` and ``unavailable`` - and only the first two are ever
   accompanied by a break count. If dynamo could not trace at all, the count is
   ``None`` and the sub-check SKIPs with the real exception text. "No graph
   breaks" is never printed for a compilation that did not happen.
   Note on backends: dynamo *traces* without a compiler backend. Where no
   inductor backend exists (no cl.exe, no triton) this check still runs with
   ``backend="eager"``, and the evidence says ``lowered: false`` so nobody reads
   a tracing result as a lowering result.
2. ``triton_lowering`` - register spills (``n_spills``) read off the compiled
   triton kernels the candidate actually launched. Spilling is invisible to a
   numerics oracle and is the usual cause of a "correct but 4x slow" kernel.
   When triton cannot be imported this SKIPs carrying ``caps.triton_error``
   verbatim; the DLL text is the finding, and paraphrasing it would destroy it.
3. ``pallas_portability`` - a pallas kernel that lowers on GPU and not on TPU is
   a portability defect that only a lowering attempt on each backend can find.
   The check enumerates the jax platforms that exist and lowers on each. Where
   jax is not installed it SKIPs with the import error.
4. ``ir_diff`` - the emitted inductor output code for the baseline and for the
   candidate, plus a unified diff. For a compiler-class task the IR *is* the
   answer, so this is a deliverable even on a clean run; it FAILs only when the
   baseline lowered and the candidate did not.

Everything that compiles runs in a ``crucible.runner.sandbox`` subprocess with a
hard timeout, so a compiler hang or a DLL initialisation crash kills a
disposable child instead of the harness.

Aggregation follows the project invariant: any sub-check firing is a FAIL; if
nothing fired but nothing could run either, the oracle is SKIP, never PASS.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..runner.sandbox import SandboxResult, input_refs_from, run_source
from ..schema import OracleResult, ShapeSpec, Task
from .base import OracleContext, register_oracle

logger = logging.getLogger(__name__)

ORACLE_ID = "O4"

#: Order is the order they are reported in.
CHECK_NAMES: tuple[str, ...] = (
    "graph_break",
    "triton_lowering",
    "pallas_portability",
    "ir_diff",
)

#: Informational: the capability each sub-check needs. ``ir_diff`` needs *either*
#: inductor backend, which ``Capabilities.missing`` cannot express, so the gate
#: is written out explicitly in the check itself.
CHECK_CAPS: dict[str, tuple[str, ...]] = {
    "graph_break": (),
    "triton_lowering": ("triton",),
    "pallas_portability": (),
    "ir_diff": ("inductor_cpu",),
}

_PAYLOAD_TOKEN = "__CRUCIBLE_O4_PAYLOAD__"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_CODE_CHARS = 20000
_MAX_DIFF_LINES = 400


# --------------------------------------------------------------------------- #
# result type
# --------------------------------------------------------------------------- #


@dataclass
class CheckResult:
    """One sub-check's own verdict. ``FAIL`` means the check fired."""

    name: str
    verdict: str  # PASS | FAIL | SKIP | ERROR
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    capabilities_used: tuple[str, ...] = ()

    @property
    def fired(self) -> bool:
        return self.verdict == "FAIL"

    @property
    def ran(self) -> bool:
        """True when the check reached a real conclusion about the candidate."""
        return self.verdict in ("PASS", "FAIL")

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "verdict": self.verdict,
            "detail": self.detail,
            "required_caps": list(CHECK_CAPS.get(self.name, ())),
            "capabilities_used": list(self.capabilities_used),
            "duration_s": round(self.duration_s, 6),
            **self.evidence,
        }


def _skip(name: str, detail: str, evidence: dict[str, Any], started: float) -> CheckResult:
    """A SKIP with an empty reason is the failure mode this project exists to prevent."""
    if not detail.strip():
        raise ValueError(f"sub-check {name!r} tried to SKIP without a reason")
    return CheckResult(name, "SKIP", detail, evidence, time.perf_counter() - started)


# --------------------------------------------------------------------------- #
# shared probe plumbing
# --------------------------------------------------------------------------- #

_PRELUDE = '''"""CRUCIBLE O4 probe. Runs inside a disposable sandbox child."""
import json
import sys
import traceback
import types
from pathlib import Path

PAYLOAD = json.loads(''' + _PAYLOAD_TOKEN + ''')
ENTRY = PAYLOAD["entry"]
DEVICE = PAYLOAD["device"]
REFS = PAYLOAD["input_refs"]
RESULT = {"ok": False, "error": "probe did not reach its end"}


def _load_inputs(refs, default_device):
    import numpy as np
    try:
        import torch
    except BaseException:
        torch = None
    out = {}
    for name, ref in refs.items():
        kind = ref.get("kind", "json")
        if kind == "json":
            out[name] = ref.get("value")
            continue
        if kind != "npy":
            raise ValueError("unknown input kind %r for %r" % (kind, name))
        arr = np.load(ref["path"], allow_pickle=False)
        if torch is None:
            out[name] = arr
            continue
        t = torch.from_numpy(arr)
        dt = ref.get("torch_dtype")
        if dt:
            t = t.to(getattr(torch, dt))
        t = t.to(ref.get("device", default_device))
        if ref.get("requires_grad"):
            t = t.detach().requires_grad_(True)
        out[name] = t
    return out


def _load_candidate(source, name):
    """Import the source as a real module.

    Dynamo resolves the traced function back to its module and to its source
    lines, so the file has to exist on disk and the module has to be in
    sys.modules; an in-memory-only module makes explain() fail with a
    ModuleNotFoundError that has nothing to do with the candidate.
    """
    path = Path(name + ".py").resolve()
    path.write_text(source, encoding="utf-8")
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


def _err(exc):
    return "%s: %s" % (type(exc).__name__, str(exc)[:4000])
'''


_EXPLAIN_BODY = '''

BACKEND = PAYLOAD["backend"]


def _report(source, tag):
    rep = {
        "tag": tag,
        "outcome": "unavailable",
        "backend": BACKEND,
        "graph_count": None,
        "graph_break_count": None,
        "op_count": None,
        "break_reasons": [],
        "fullgraph_ok": None,
        "fullgraph_error": None,
        "error": None,
    }
    try:
        import torch
        import torch._dynamo as dynamo
    except BaseException as exc:
        rep["error"] = "torch._dynamo could not be imported: " + _err(exc)
        rep["traceback"] = traceback.format_exc()[-4000:]
        return rep
    try:
        mod = _load_candidate(source, "crucible_" + tag)
    except BaseException as exc:
        rep["error"] = "source did not import: " + _err(exc)
        rep["traceback"] = traceback.format_exc()[-4000:]
        return rep
    fn = getattr(mod, ENTRY, None)
    if fn is None or not callable(fn):
        rep["error"] = "source defines no callable %r" % ENTRY
        return rep

    try:
        dynamo.reset()
        explanation = dynamo.explain(fn)(**_load_inputs(REFS, DEVICE))
    except BaseException as exc:
        rep["error"] = "torch._dynamo.explain raised: " + _err(exc)
        rep["traceback"] = traceback.format_exc()[-4000:]
        return rep

    n_breaks = int(getattr(explanation, "graph_break_count", 0) or 0)
    rep["graph_count"] = int(getattr(explanation, "graph_count", 0) or 0)
    rep["graph_break_count"] = n_breaks
    rep["op_count"] = int(getattr(explanation, "op_count", 0) or 0)
    reasons = []
    for item in (getattr(explanation, "break_reasons", []) or []):
        stack = []
        for frame in (getattr(item, "user_stack", []) or []):
            try:
                stack.append(str(frame).strip()[:400])
            except BaseException:
                stack.append("<unprintable frame>")
        reasons.append({"reason": str(getattr(item, "reason", item))[:1000],
                        "user_stack": stack[-4:]})
    rep["break_reasons"] = reasons
    rep["outcome"] = "compiled_cleanly" if n_breaks == 0 else "graph_breaks"

    try:
        dynamo.reset()
        compiled = torch.compile(fn, fullgraph=True, backend=BACKEND)
        compiled(**_load_inputs(REFS, DEVICE))
        rep["fullgraph_ok"] = True
    except BaseException as exc:
        rep["fullgraph_ok"] = False
        rep["fullgraph_error"] = _err(exc)
    return rep


try:
    RESULT = {"ok": True, "reports": [_report(src, tag)
                                      for tag, src in PAYLOAD["sources"]]}
except BaseException as exc:
    RESULT = {"ok": False, "error": _err(exc), "traceback": traceback.format_exc()[-4000:]}
'''


_TRITON_BODY = '''

def _kernel_metadata(kernel):
    rec = {}
    for field_name in ("n_spills", "n_regs", "shared", "num_warps", "num_stages"):
        value = getattr(kernel, field_name, None)
        if value is None:
            meta = getattr(kernel, "metadata", None)
            value = getattr(meta, field_name, None)
            if value is None and isinstance(meta, dict):
                value = meta.get(field_name)
        if value is not None:
            try:
                rec[field_name] = int(value)
            except (TypeError, ValueError):
                rec[field_name] = str(value)
    name = getattr(kernel, "name", None)
    if name is None:
        meta = getattr(kernel, "metadata", None)
        name = getattr(meta, "name", None)
    rec["kernel"] = str(name) if name is not None else "<unnamed>"
    return rec


def _collect(mod, triton):
    jit_type = getattr(getattr(triton, "runtime", None), "jit", None)
    jit_type = getattr(jit_type, "JITFunction", None)
    out = []
    seen = set()
    for attr_name, obj in list(vars(mod).items()):
        if jit_type is not None and not isinstance(obj, jit_type):
            continue
        if jit_type is None and not hasattr(obj, "cache"):
            continue
        cache = getattr(obj, "cache", None)
        if not isinstance(cache, dict):
            continue
        for per_device in cache.values():
            entries = per_device.values() if isinstance(per_device, dict) else [per_device]
            for kernel in entries:
                if id(kernel) in seen:
                    continue
                seen.add(id(kernel))
                rec = _kernel_metadata(kernel)
                rec["symbol"] = attr_name
                out.append(rec)
    return out


try:
    import triton
except BaseException as exc:
    RESULT = {"ok": False, "stage": "import_triton", "error": _err(exc),
              "traceback": traceback.format_exc()[-4000:]}
else:
    try:
        mod = _load_candidate(PAYLOAD["source"], "crucible_candidate")
        fn = getattr(mod, ENTRY, None)
        if fn is None or not callable(fn):
            RESULT = {"ok": False, "stage": "entry",
                      "error": "candidate defines no callable %r" % ENTRY}
        else:
            fn(**_load_inputs(REFS, DEVICE))
            try:
                import torch
                if str(DEVICE).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
            except BaseException:
                pass
            RESULT = {"ok": True, "stage": "collected",
                      "triton_version": str(getattr(triton, "__version__", "?")),
                      "kernels": _collect(mod, triton)}
    except BaseException as exc:
        RESULT = {"ok": False, "stage": "run", "error": _err(exc),
                  "traceback": traceback.format_exc()[-4000:]}
'''


_PALLAS_BODY = '''

try:
    import jax
except BaseException as exc:
    RESULT = {"ok": False, "stage": "import_jax", "error": _err(exc)}
else:
    try:
        from jax.experimental import pallas as pl   # noqa: F401
        pallas_error = None
    except BaseException as exc:
        pallas_error = _err(exc)
    if pallas_error is not None:
        RESULT = {"ok": False, "stage": "import_pallas", "error": pallas_error,
                  "jax_version": str(getattr(jax, "__version__", "?"))}
    else:
        backends = []
        try:
            mod = _load_candidate(PAYLOAD["source"], "crucible_candidate")
            fn = getattr(mod, ENTRY, None)
        except BaseException as exc:
            fn = None
            backends.append({"platform": "<import>", "available": False,
                             "lowered": False, "error": _err(exc)})
        if fn is None or not callable(fn):
            RESULT = {"ok": False, "stage": "entry",
                      "error": "candidate defines no callable %r" % ENTRY,
                      "backends": backends}
        else:
            inputs = _load_inputs(REFS, DEVICE)
            import numpy as _np
            args = {k: _np.asarray(v) if hasattr(v, "__array__") else v
                    for k, v in inputs.items()}
            for platform in PAYLOAD.get("platforms", ["tpu", "gpu", "cpu"]):
                rec = {"platform": platform, "available": False, "lowered": False,
                       "error": None, "device_count": 0}
                try:
                    devices = jax.devices(platform)
                except BaseException as exc:
                    rec["error"] = "no %s backend: %s" % (platform, _err(exc))
                    backends.append(rec)
                    continue
                rec["available"] = bool(devices)
                rec["device_count"] = len(devices)
                if not devices:
                    rec["error"] = "jax.devices(%r) returned nothing" % platform
                    backends.append(rec)
                    continue
                try:
                    with jax.default_device(devices[0]):
                        lowered = jax.jit(fn).lower(**args)
                        compiled = lowered.compile()
                    rec["lowered"] = True
                    try:
                        rec["cost_analysis"] = {k: float(v) for k, v in
                                                (compiled.cost_analysis() or {}).items()}
                    except BaseException:
                        pass
                except BaseException as exc:
                    rec["error"] = _err(exc)
                backends.append(rec)
            RESULT = {"ok": True, "stage": "lowered",
                      "jax_version": str(getattr(jax, "__version__", "?")),
                      "backends": backends}
'''


_IR_BODY = '''

BACKEND = PAYLOAD["backend"]


def _emit(source, tag):
    rec = {"tag": tag, "ok": False, "code": [], "error": None}
    try:
        import torch
        import torch._dynamo as dynamo
        from torch._inductor.utils import run_and_get_code
    except BaseException as exc:
        rec["error"] = "inductor code capture unavailable: " + _err(exc)
        return rec
    try:
        mod = _load_candidate(source, "crucible_" + tag)
    except BaseException as exc:
        rec["error"] = "source did not import: " + _err(exc)
        return rec
    fn = getattr(mod, ENTRY, None)
    if fn is None or not callable(fn):
        rec["error"] = "source defines no callable %r" % ENTRY
        return rec
    try:
        dynamo.reset()
        compiled = torch.compile(fn, backend=BACKEND)
        _out, codes = run_and_get_code(compiled, **_load_inputs(REFS, DEVICE))
        rec["ok"] = True
        rec["code"] = [str(c) for c in (codes or [])]
    except BaseException as exc:
        rec["error"] = _err(exc)
        rec["traceback"] = traceback.format_exc()[-4000:]
    return rec


try:
    RESULT = {"ok": True, "emissions": [_emit(src, tag)
                                        for tag, src in PAYLOAD["sources"]]}
except BaseException as exc:
    RESULT = {"ok": False, "error": _err(exc), "traceback": traceback.format_exc()[-4000:]}
'''


def _build_probe(body: str, payload: Mapping[str, Any]) -> str:
    """Inline the payload as a JSON string literal; nothing is pickled."""
    literal = repr(json.dumps(payload, default=str))
    return (_PRELUDE + body).replace(_PAYLOAD_TOKEN, literal)


def _safe_dir_name(name: str, fallback: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name).strip("._")
    return cleaned or fallback


def _probe_timeout(ctx: OracleContext) -> float:
    override = ctx.extras.get("o4_timeout_s")
    if isinstance(override, (int, float)) and float(override) > 0:
        return float(override)
    return float(ctx.cfg.sandbox_timeout_s)


def _payload_of(res: SandboxResult) -> tuple[dict[str, Any] | None, str]:
    """(RESULT dict, problem). ``problem`` non-empty means the child produced nothing usable."""
    if not res.ok:
        if res.timed_out:
            return None, (
                f"the compilation subprocess was killed after {res.duration_s:.1f}s "
                "(a compiler hang cannot take the harness down, but it also cannot be graded)"
            )
        err = res.error or {}
        return None, (
            f"the compilation subprocess failed: {err.get('type', 'error')}: "
            f"{str(err.get('message', res.message))[:2000]}"
        )
    value = res.value if isinstance(res.value, dict) else {}
    payload = value.get("result")
    if not isinstance(payload, dict):
        return None, (
            f"the compilation subprocess returned {type(payload).__name__}, not a probe result; "
            f"stderr tail: {res.stderr.strip()[-600:]}"
        )
    return payload, ""


def _run_probe(
    ctx: OracleContext,
    check: str,
    body: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    wd = ctx.sub_workdir(f"{ORACLE_ID}/{_safe_dir_name(check, 'check')}")
    source = _build_probe(body, payload)
    res = run_source(
        source,
        workdir=wd,
        timeout_s=_probe_timeout(ctx),
        seed=int(ctx.rng_seed),
    )
    return _payload_of(res)


# --------------------------------------------------------------------------- #
# context helpers
# --------------------------------------------------------------------------- #


def select_shape(ctx: OracleContext) -> ShapeSpec | None:
    """The shape the compile probes are driven with, in a fixed order."""
    override = ctx.extras.get("o4_shape")
    if isinstance(override, ShapeSpec):
        return override
    groups: Sequence[Sequence[ShapeSpec]] = (
        list(getattr(ctx.seed, "shape_sweep", []) or []) if ctx.seed is not None else [],
        list(ctx.task.detect_shapes),
        list(ctx.task.decoy_shapes),
    )
    for group in groups:
        for shape in group:
            return shape
    return None


def _make_inputs(ctx: OracleContext, shape: ShapeSpec, device: str) -> dict[str, Any]:
    """Build the probe inputs, tolerating the simpler ``make_inputs`` signatures."""
    seed = ctx.seed
    generator: Any = None
    try:
        import torch

        generator = torch.Generator(device="cpu" if device == "cpu" else device)
        generator.manual_seed(int(ctx.rng_seed))
    except (ImportError, RuntimeError, TypeError) as exc:
        logger.debug("no seeded generator for O4 shape %s: %s", shape.name, exc)
    for kwargs in ({"device": device, "generator": generator}, {"device": device}, {}):
        try:
            out = seed.make_inputs(shape, **kwargs)
        except TypeError as exc:
            logger.debug("make_inputs rejected %s for %s: %s", sorted(kwargs), shape.name, exc)
            continue
        if not isinstance(out, dict):
            raise TypeError(
                f"seed {getattr(seed, 'id', '?')}: make_inputs returned "
                f"{type(out).__name__}, expected a dict of keyword arguments"
            )
        return out
    raise TypeError(
        f"seed {getattr(seed, 'id', '?')}: make_inputs accepted none of the "
        "supported signatures (shape[, device][, generator])"
    )


def _prepare_inputs(ctx: OracleContext, check: str) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    """(input_refs, problem, note). ``problem`` non-empty is the sub-check's SKIP reason."""
    note: dict[str, Any] = {}
    seed = ctx.seed
    if seed is None:
        return None, "no seed was supplied in the oracle context; there is nothing to compile", note
    entry = str(getattr(seed, "entry", "") or "")
    if not entry:
        return None, f"seed {getattr(seed, 'id', '?')!r} declares no entry point to compile", note
    if not ctx.candidate_src.strip():
        return None, "candidate source is empty; nothing was compiled", note
    shape = select_shape(ctx)
    if shape is None:
        return None, (
            f"seed {getattr(seed, 'id', '?')!r} and task {ctx.task.task_id!r} expose no shape "
            "to drive a compilation with"
        ), note
    note["shape"] = shape.name
    note["shape_key"] = shape.key()
    device = str(ctx.device or "cpu")
    try:
        inputs = _make_inputs(ctx, shape, device)
    except Exception as exc:  # noqa: BLE001 - reported as the SKIP reason, never swallowed
        return None, (
            f"inputs for shape {shape.name!r} could not be built "
            f"({type(exc).__name__}: {exc}); no compilation was attempted"
        ), note
    wd = ctx.sub_workdir(f"{ORACLE_ID}/{_safe_dir_name(check, 'check')}/in")
    try:
        refs = input_refs_from(inputs, wd, device=device)
    except Exception as exc:  # noqa: BLE001
        return None, (
            f"inputs for shape {shape.name!r} could not be materialised for the subprocess "
            f"({type(exc).__name__}: {exc})"
        ), note
    return refs, "", note


def _inductor_backend(ctx: OracleContext) -> tuple[str | None, str, str]:
    """(cap name, device, reason-if-none) for an inductor backend that can emit code."""
    device = str(ctx.device or "cpu")
    if device.startswith("cuda"):
        if ctx.caps.inductor_cuda:
            return "inductor_cuda", device, ""
        cuda_reason = ctx.caps.detail("inductor_cuda") or "inductor_cuda probed False"
        if ctx.caps.inductor_cpu:
            return "inductor_cpu", "cpu", ""
        cpu_reason = ctx.caps.detail("inductor_cpu") or "inductor_cpu probed False"
        return None, device, f"inductor_cuda: {cuda_reason}; inductor_cpu: {cpu_reason}"
    if ctx.caps.inductor_cpu:
        return "inductor_cpu", "cpu", ""
    return None, device, f"inductor_cpu: {ctx.caps.detail('inductor_cpu') or 'inductor_cpu probed False'}"


def _truncate(text: str, limit: int = _MAX_CODE_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


# --------------------------------------------------------------------------- #
# 1. graph breaks
# --------------------------------------------------------------------------- #


def check_graph_break(ctx: OracleContext) -> CheckResult:
    """Enumerate dynamo graph breaks for the candidate and for the baseline.

    Dynamo traces without a compiler backend, so this check runs on machines
    with no inductor at all; the evidence records which backend was used and
    whether the graph was actually lowered.
    """
    started = time.perf_counter()
    evidence: dict[str, Any] = {}

    refs, problem, note = _prepare_inputs(ctx, "graph_break")
    evidence.update(note)
    if refs is None:
        return _skip("graph_break", problem, evidence, started)

    cap, device, _reason = _inductor_backend(ctx)
    backend = "inductor" if cap is not None else "eager"
    caps_used: tuple[str, ...] = (cap,) if cap is not None else ()
    evidence["backend"] = backend
    evidence["lowered"] = backend != "eager"
    evidence["device"] = device
    if backend == "eager":
        evidence["backend_note"] = (
            "no inductor backend on this machine, so dynamo traced with backend='eager': "
            "graph breaks are a tracing property and are genuinely measured, but no kernel "
            "was lowered and no lowering defect could have been observed"
        )

    entry = str(getattr(ctx.seed, "entry", ""))
    payload = {
        "entry": entry,
        "device": device,
        "input_refs": refs,
        "backend": backend,
        "sources": [["candidate", ctx.candidate_src], ["baseline", ctx.task.baseline_code]],
    }
    result, probe_problem = _run_probe(ctx, "graph_break", _EXPLAIN_BODY, payload)
    if result is None:
        return _skip("graph_break", probe_problem, evidence, started)
    if not result.get("ok"):
        return _skip(
            "graph_break",
            f"the dynamo probe did not complete: {result.get('error', 'no error text')}",
            {**evidence, "probe": result},
            started,
        )

    reports = {str(r.get("tag")): r for r in (result.get("reports") or [])}
    cand = reports.get("candidate")
    base = reports.get("baseline")
    evidence["candidate"] = cand
    evidence["baseline"] = base
    if cand is None:
        return _skip(
            "graph_break",
            "the dynamo probe returned no report for the candidate",
            evidence,
            started,
        )

    outcome = str(cand.get("outcome"))
    evidence["outcome"] = outcome
    if outcome == "unavailable":
        # The count stays None. Reporting "no graph breaks" here would be a lie
        # about a compilation that never happened.
        return _skip(
            "graph_break",
            (
                "dynamo could not trace the candidate, so its graph-break count is unknown "
                f"(not zero): {cand.get('error', 'no error text')}"
            ),
            evidence,
            started,
        )

    n_cand = int(cand.get("graph_break_count") or 0)
    reasons = [str(r.get("reason", "")) for r in (cand.get("break_reasons") or [])]
    base_ok = bool(base and base.get("outcome") != "unavailable")
    n_base = int(base.get("graph_break_count") or 0) if base_ok else None
    evidence["baseline_graph_break_count"] = n_base
    evidence["graph_break_count"] = n_cand
    evidence["break_reasons"] = reasons

    where = f"backend={backend}, device={device}, shape={note.get('shape', '?')}"
    if outcome == "compiled_cleanly":
        detail = (
            f"torch._dynamo.explain traced {cand.get('graph_count')} graph(s) with 0 graph breaks "
            f"and torch.compile(fullgraph=True) "
            f"{'succeeded' if cand.get('fullgraph_ok') else 'did not succeed'} ({where})"
        )
        if cand.get("fullgraph_ok") is False:
            return CheckResult(
                "graph_break",
                "FAIL",
                (
                    "explain reported 0 graph breaks but torch.compile(fullgraph=True) still "
                    f"failed, so the graph is not whole: {cand.get('fullgraph_error')} ({where})"
                ),
                evidence,
                time.perf_counter() - started,
                caps_used,
            )
        return CheckResult("graph_break", "PASS", detail, evidence, time.perf_counter() - started, caps_used)

    listed = "; ".join(f"[{i + 1}] {r}" for i, r in enumerate(reasons)) or "no reason text recorded"
    if n_base is not None and n_cand > n_base:
        return CheckResult(
            "graph_break",
            "FAIL",
            (
                f"the candidate has {n_cand} graph break(s) where the baseline has {n_base} "
                f"({where}) -- {listed}"
            ),
            evidence,
            time.perf_counter() - started,
            caps_used,
        )
    if n_base is None:
        detail = (
            f"the candidate has {n_cand} graph break(s) ({where}); the baseline could not be "
            "traced, so this is reported, not judged as a regression -- " + listed
        )
    else:
        detail = (
            f"the candidate has {n_cand} graph break(s), the same as or fewer than the "
            f"baseline's {n_base} ({where}) -- {listed}"
        )
    return CheckResult("graph_break", "PASS", detail, evidence, time.perf_counter() - started, caps_used)


# --------------------------------------------------------------------------- #
# 2. triton lowering / register spills
# --------------------------------------------------------------------------- #


def check_triton_lowering(ctx: OracleContext) -> CheckResult:
    """Read ``n_spills`` off the triton kernels the candidate actually launched."""
    started = time.perf_counter()
    evidence: dict[str, Any] = {"required_cap": "triton"}

    if not ctx.caps.triton:
        # Verbatim. The DLL initialisation text is the finding; a paraphrase of
        # it would be useless to whoever has to fix this machine.
        reason = ctx.caps.triton_error or ctx.caps.detail("triton") or "triton probed unavailable"
        evidence["triton_error"] = ctx.caps.triton_error
        return _skip(
            "triton_lowering",
            f"triton is unavailable on this machine, so no kernel was compiled and no "
            f"register-spill count exists: {reason}",
            evidence,
            started,
        )
    if not ctx.caps.cuda:
        return _skip(
            "triton_lowering",
            "triton is importable but there is no CUDA device to compile a kernel for: "
            + (ctx.caps.detail("cuda") or "torch.cuda.is_available() returned False"),
            evidence,
            started,
        )

    refs, problem, note = _prepare_inputs(ctx, "triton_lowering")
    evidence.update(note)
    if refs is None:
        return _skip("triton_lowering", problem, evidence, started)

    device = str(ctx.device or "cpu")
    if not device.startswith("cuda"):
        device = "cuda"
    payload = {
        "entry": str(getattr(ctx.seed, "entry", "")),
        "device": device,
        "input_refs": refs,
        "source": ctx.candidate_src,
    }
    result, probe_problem = _run_probe(ctx, "triton_lowering", _TRITON_BODY, payload)
    if result is None:
        return _skip("triton_lowering", probe_problem, evidence, started)
    evidence["probe"] = result
    if not result.get("ok"):
        return _skip(
            "triton_lowering",
            f"the triton probe stopped at stage {result.get('stage', '?')!r}: "
            f"{result.get('error', 'no error text')}",
            evidence,
            started,
        )

    kernels = list(result.get("kernels") or [])
    evidence["kernels"] = kernels
    evidence["triton_version"] = result.get("triton_version")
    if not kernels:
        return _skip(
            "triton_lowering",
            "the candidate launched no triton kernels whose compiled metadata could be read, "
            "so there is no register-spill count to report",
            evidence,
            started,
        )
    with_counts = [k for k in kernels if isinstance(k.get("n_spills"), int)]
    if not with_counts:
        return _skip(
            "triton_lowering",
            f"{len(kernels)} triton kernel(s) compiled but none exposed an n_spills field in "
            "its metadata on this triton version; the spill count is absent, not zero",
            evidence,
            started,
        )

    limit = int(ctx.extras.get("o4_max_spills", 0))
    evidence["max_spills_allowed"] = limit
    spilling = [k for k in with_counts if int(k["n_spills"]) > limit]
    summary = ", ".join(f"{k['kernel']}: n_spills={k['n_spills']}" for k in with_counts)
    if spilling:
        return CheckResult(
            "triton_lowering",
            "FAIL",
            f"{len(spilling)}/{len(with_counts)} compiled triton kernel(s) spill registers "
            f"above the allowed {limit}: {summary}",
            evidence,
            time.perf_counter() - started,
            ("triton", "cuda"),
        )
    return CheckResult(
        "triton_lowering",
        "PASS",
        f"{len(with_counts)} compiled triton kernel(s) report no register spills "
        f"above the allowed {limit}: {summary}",
        evidence,
        time.perf_counter() - started,
        ("triton", "cuda"),
    )


# --------------------------------------------------------------------------- #
# 3. pallas portability
# --------------------------------------------------------------------------- #


def check_pallas_portability(ctx: OracleContext) -> CheckResult:
    """Lower the candidate on every jax platform that exists and compare."""
    started = time.perf_counter()
    platforms = list(ctx.extras.get("o4_jax_platforms") or ("tpu", "gpu", "cpu"))
    evidence: dict[str, Any] = {"platforms_requested": platforms}

    refs, problem, note = _prepare_inputs(ctx, "pallas_portability")
    evidence.update(note)
    if refs is None:
        return _skip("pallas_portability", problem, evidence, started)

    payload = {
        "entry": str(getattr(ctx.seed, "entry", "")),
        "device": "cpu",
        "input_refs": refs,
        "source": ctx.candidate_src,
        "platforms": platforms,
    }
    result, probe_problem = _run_probe(ctx, "pallas_portability", _PALLAS_BODY, payload)
    if result is None:
        return _skip("pallas_portability", probe_problem, evidence, started)
    evidence["probe"] = result

    if not result.get("ok"):
        stage = str(result.get("stage", "?"))
        error = str(result.get("error", "no error text"))
        if stage == "import_jax":
            return _skip(
                "pallas_portability",
                f"jax is not installed, so no pallas kernel could be lowered for any backend: {error}",
                evidence,
                started,
            )
        if stage == "import_pallas":
            return _skip(
                "pallas_portability",
                f"jax {result.get('jax_version', '?')} is installed but jax.experimental.pallas "
                f"could not be imported: {error}",
                evidence,
                started,
            )
        return _skip(
            "pallas_portability",
            f"the pallas probe stopped at stage {stage!r}: {error}",
            evidence,
            started,
        )

    backends = list(result.get("backends") or [])
    evidence["backends"] = backends
    present = [b for b in backends if b.get("available")]
    if not present:
        detail = "; ".join(f"{b.get('platform')}: {b.get('error')}" for b in backends)
        return _skip(
            "pallas_portability",
            f"jax {result.get('jax_version', '?')} is installed but exposes no usable backend "
            f"among {platforms}, so portability could not be compared: {detail}",
            evidence,
            started,
        )
    lowered = [b for b in present if b.get("lowered")]
    failed = [b for b in present if not b.get("lowered")]
    names_ok = ", ".join(str(b.get("platform")) for b in lowered) or "none"
    if failed and lowered:
        detail = "; ".join(f"{b.get('platform')}: {b.get('error')}" for b in failed)
        return CheckResult(
            "pallas_portability",
            "FAIL",
            f"the kernel lowers on {names_ok} but not on "
            f"{', '.join(str(b.get('platform')) for b in failed)}, which is a portability "
            f"defect -- {detail}",
            evidence,
            time.perf_counter() - started,
        )
    if not lowered:
        detail = "; ".join(f"{b.get('platform')}: {b.get('error')}" for b in failed)
        return CheckResult(
            "pallas_portability",
            "FAIL",
            f"the kernel lowers on none of the available backends "
            f"({', '.join(str(b.get('platform')) for b in present)}) -- {detail}",
            evidence,
            time.perf_counter() - started,
        )
    return CheckResult(
        "pallas_portability",
        "PASS",
        f"the kernel lowered on every available jax backend ({names_ok})",
        evidence,
        time.perf_counter() - started,
    )


# --------------------------------------------------------------------------- #
# 4. emitted IR diff
# --------------------------------------------------------------------------- #


def unified_ir_diff(baseline: str, candidate: str, max_lines: int = _MAX_DIFF_LINES) -> tuple[list[str], bool]:
    """(diff lines, truncated). Empty list means the two emissions are identical."""
    lines = list(
        difflib.unified_diff(
            baseline.splitlines(),
            candidate.splitlines(),
            fromfile="baseline.inductor.py",
            tofile="candidate.inductor.py",
            lineterm="",
            n=3,
        )
    )
    if len(lines) <= max_lines:
        return lines, False
    return lines[:max_lines], True


def check_ir_diff(ctx: OracleContext) -> CheckResult:
    """Capture the emitted inductor output code for both sides and diff it."""
    started = time.perf_counter()
    evidence: dict[str, Any] = {}

    cap, device, reason = _inductor_backend(ctx)
    if cap is None:
        return _skip(
            "ir_diff",
            "no inductor backend on this machine can emit output code, so there is no IR to "
            f"capture or diff -- {reason}",
            evidence,
            started,
        )
    evidence["backend_capability"] = cap
    evidence["device"] = device

    refs, problem, note = _prepare_inputs(ctx, "ir_diff")
    evidence.update(note)
    if refs is None:
        return _skip("ir_diff", problem, evidence, started)

    payload = {
        "entry": str(getattr(ctx.seed, "entry", "")),
        "device": device,
        "input_refs": refs,
        "backend": "inductor",
        "sources": [["baseline", ctx.task.baseline_code], ["candidate", ctx.candidate_src]],
    }
    result, probe_problem = _run_probe(ctx, "ir_diff", _IR_BODY, payload)
    if result is None:
        return _skip("ir_diff", probe_problem, evidence, started)
    if not result.get("ok"):
        return _skip(
            "ir_diff",
            f"the inductor capture probe did not complete: {result.get('error', 'no error text')}",
            {**evidence, "probe": result},
            started,
        )

    emissions = {str(e.get("tag")): e for e in (result.get("emissions") or [])}
    base = emissions.get("baseline") or {}
    cand = emissions.get("candidate") or {}
    evidence["baseline_emitted"] = bool(base.get("ok"))
    evidence["candidate_emitted"] = bool(cand.get("ok"))
    evidence["baseline_error"] = base.get("error")
    evidence["candidate_error"] = cand.get("error")

    if not base.get("ok") and not cand.get("ok"):
        return _skip(
            "ir_diff",
            "inductor emitted no output code for either side, so no IR could be captured "
            f"(baseline: {base.get('error', 'no error text')}; "
            f"candidate: {cand.get('error', 'no error text')})",
            evidence,
            started,
        )

    base_code = "\n\n".join(str(c) for c in (base.get("code") or []))
    cand_code = "\n\n".join(str(c) for c in (cand.get("code") or []))
    base_text, base_trunc = _truncate(base_code)
    cand_text, cand_trunc = _truncate(cand_code)
    evidence["baseline_output_code"] = base_text
    evidence["candidate_output_code"] = cand_text
    evidence["baseline_output_code_truncated"] = base_trunc
    evidence["candidate_output_code_truncated"] = cand_trunc
    evidence["baseline_kernel_count"] = len(base.get("code") or [])
    evidence["candidate_kernel_count"] = len(cand.get("code") or [])

    if base.get("ok") and not cand.get("ok"):
        return CheckResult(
            "ir_diff",
            "FAIL",
            "inductor lowered the baseline but could not lower the candidate: "
            f"{cand.get('error', 'no error text')}",
            evidence,
            time.perf_counter() - started,
            (cap,),
        )
    if cand.get("ok") and not base.get("ok"):
        return _skip(
            "ir_diff",
            "inductor lowered the candidate but not the baseline, so there is no reference IR "
            f"to diff against: {base.get('error', 'no error text')}",
            evidence,
            started,
        )

    diff_lines, diff_trunc = unified_ir_diff(base_code, cand_code)
    evidence["ir_diff"] = diff_lines
    evidence["ir_diff_truncated"] = diff_trunc
    evidence["ir_identical"] = not diff_lines
    n_added = sum(1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++"))
    n_removed = sum(1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---"))
    evidence["ir_lines_added"] = n_added
    evidence["ir_lines_removed"] = n_removed
    if not diff_lines:
        detail = (
            f"inductor emitted identical output code for the baseline and the candidate on "
            f"{device} ({evidence['candidate_kernel_count']} emission(s) captured)"
        )
    else:
        detail = (
            f"inductor output code captured for both sides on {device}; the unified diff has "
            f"+{n_added}/-{n_removed} line(s)"
            + (" (truncated)" if diff_trunc else "")
        )
    return CheckResult("ir_diff", "PASS", detail, evidence, time.perf_counter() - started, (cap,))


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

CHECKS: dict[str, Callable[[OracleContext], CheckResult]] = {
    "graph_break": check_graph_break,
    "triton_lowering": check_triton_lowering,
    "pallas_portability": check_pallas_portability,
    "ir_diff": check_ir_diff,
}


def run_check(name: str, ctx: OracleContext) -> CheckResult:
    """Run one named sub-check, converting an unexpected exception into ERROR."""
    fn = CHECKS.get(name)
    if fn is None:
        raise ValueError(f"unknown compile sub-check {name!r}; known checks: {sorted(CHECKS)}")
    started = time.perf_counter()
    try:
        return fn(ctx)
    except Exception as exc:  # noqa: BLE001 - reported as ERROR, never swallowed
        return CheckResult(
            name=name,
            verdict="ERROR",
            detail=f"{type(exc).__name__}: {exc}",
            evidence={"traceback": traceback.format_exc()[-4000:]},
            duration_s=time.perf_counter() - started,
        )


def run_checks(ctx: OracleContext, names: Iterable[str] | None = None) -> list[CheckResult]:
    requested = list(names) if names is not None else list(CHECK_NAMES)
    return [run_check(name, ctx) for name in requested]


class CompileOracle:
    """O4. Four sub-checks with independent capability gates and skip reasons."""

    id = ORACLE_ID
    name = "compilation and portability"
    #: Empty on purpose: the gates are per sub-check, so a machine without
    #: triton still gets its graph-break and IR evidence instead of one
    #: undifferentiated SKIP.
    required_caps: tuple[str, ...] = ()

    def applies_to(self, task: Task) -> bool:
        return True

    def run(self, ctx: OracleContext) -> OracleResult:
        started = time.perf_counter()
        results = run_checks(ctx, ctx.extras.get("o4_checks"))
        checks = {r.name: r.as_dict() for r in results}
        fired = [r.name for r in results if r.verdict == "FAIL"]
        errored = [r.name for r in results if r.verdict == "ERROR"]
        skipped = {r.name: r.detail for r in results if r.verdict == "SKIP"}
        passed = [r.name for r in results if r.verdict == "PASS"]
        caps_used: list[str] = []
        for r in results:
            for cap in r.capabilities_used:
                if cap not in caps_used:
                    caps_used.append(cap)
        evidence: dict[str, Any] = {
            "checks": checks,
            "fired": fired,
            "errored": errored,
            "skipped": skipped,
            "passed": passed,
            "ran": [r.name for r in results if r.ran],
        }
        duration = time.perf_counter() - started

        if fired:
            details = "; ".join(f"{r.name}: {r.detail}" for r in results if r.verdict == "FAIL")
            return OracleResult(
                oracle=ORACLE_ID,
                verdict="FAIL",
                reason=f"compile sub-check(s) {fired} fired -- {details}",
                evidence=evidence,
                duration_s=duration,
                capabilities_used=caps_used,
            )
        if errored:
            details = "; ".join(f"{r.name}: {r.detail}" for r in results if r.verdict == "ERROR")
            return OracleResult(
                oracle=ORACLE_ID,
                verdict="ERROR",
                reason=f"compile sub-check(s) {errored} could not complete -- {details}",
                evidence=evidence,
                duration_s=duration,
                capabilities_used=caps_used,
            )
        if not passed:
            # Nothing was compiled anywhere. This must read as unverified, and
            # the caller must be able to see exactly which backend was missing.
            return OracleResult(
                oracle=ORACLE_ID,
                verdict="SKIP",
                reason=(
                    "no compile sub-check could run on this machine, so nothing about "
                    "compilation was verified: "
                    + "; ".join(f"{k}: {v}" for k, v in skipped.items())
                ),
                evidence=evidence,
                duration_s=duration,
                capabilities_used=caps_used,
            )
        return OracleResult(
            oracle=ORACLE_ID,
            verdict="PASS",
            reason="",
            evidence=evidence,
            duration_s=duration,
            capabilities_used=caps_used,
        )


O4 = CompileOracle()
ORACLE = register_oracle(O4)

__all__ = [
    "O4",
    "ORACLE",
    "ORACLE_ID",
    "CompileOracle",
    "CHECKS",
    "CHECK_NAMES",
    "CHECK_CAPS",
    "CheckResult",
    "check_graph_break",
    "check_triton_lowering",
    "check_pallas_portability",
    "check_ir_diff",
    "run_check",
    "run_checks",
    "select_shape",
    "unified_ir_diff",
]
