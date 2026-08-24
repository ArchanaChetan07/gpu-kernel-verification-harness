"""Run candidate code in a separate interpreter, never in this one.

Two things make this module load-bearing:

* **Isolation.** Candidate source is written to a file and executed by a fresh
  ``sys.executable``. An ``exec()`` in-process would let a candidate mutate the
  grader, monkeypatch the reference, or crash the whole run with a CUDA fault.
* **A hard timeout that actually kills.** ``subprocess.run(timeout=...)`` kills
  the direct child but leaves grandchildren (torch worker processes, nvcc)
  behind on Windows. We use ``taskkill /T /F`` so a hung candidate cannot
  survive as an orphan holding the GPU.

Tensors cross the boundary as ``.npy`` files plus checksums. Nothing is pickled:
unpickling attacker-controlled bytes in the parent would undo the isolation.

Job protocol (JSON in, JSON out over files)::

    {"op": "call"|"time"|"exec",
     "source": "<python source of the candidate module>",
     "entry": "<function name>",                     # call / time
     "inputs": {"x": {"kind": "npy", "path": ...,    # call / time
                      "torch_dtype": "bfloat16", "device": "cpu"},
                "n": {"kind": "json", "value": 128}},
     "call_style": "kwargs"|"args"|"single_dict",
     "reps": 30, "warmup": 10,                       # time
     "device": "cpu"|"cuda", "seed": 0,
     "deterministic": true, "save_outputs": true}

Result::

    {"ok": bool, "value": {...} | null,
     "error": {"type": ..., "message": ..., "traceback": ...} | null}
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..errors import SandboxError

logger = logging.getLogger(__name__)

JOB_VERSION = 1

# --------------------------------------------------------------------------- #
# the child program (kept as text so the parent never imports candidate code)
# --------------------------------------------------------------------------- #

_CHILD_SOURCE = r'''"""CRUCIBLE sandbox child. Executes one job and exits."""
import hashlib
import json
import os
import random
import sys
import time
import traceback
import types
from pathlib import Path

JOB = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
RESULT_PATH = Path(JOB["result_path"])


def write_result(payload):
    RESULT_PATH.write_text(json.dumps(payload, default=str), encoding="utf-8")


def finish_ok(value):
    write_result({"ok": True, "value": value, "error": None})
    sys.stdout.flush()
    sys.exit(0)


def finish_err(kind, message, tb=""):
    write_result({"ok": False, "value": None,
                  "error": {"type": str(kind), "message": str(message)[:8000],
                            "traceback": str(tb)[-12000:]}})
    sys.stdout.flush()
    sys.exit(0)


def try_torch():
    try:
        import torch
        return torch
    except BaseException:
        return None


def checksum(arr):
    import numpy as np
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()


def to_numpy(value):
    """(ndarray, original-dtype-name) or (None, type name) for opaque objects."""
    import numpy as np
    torch = try_torch()
    if torch is not None and isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        orig = str(t.dtype).replace("torch.", "")
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)   # numpy has no bfloat16; widening is exact
        return t.contiguous().numpy(), orig
    if isinstance(value, np.ndarray):
        return value, str(value.dtype)
    if isinstance(value, (bool, int, float, complex)):
        return np.asarray(value), type(value).__name__
    return None, type(value).__name__


def flatten(value):
    if isinstance(value, dict):
        return [(str(k), v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))]
    if isinstance(value, (list, tuple)):
        return [("out%d" % i, v) for i, v in enumerate(value)]
    return [("out", value)]


def describe(name, obj, save_dir):
    import numpy as np
    arr, orig = to_numpy(obj)
    if arr is None:
        return {"name": name, "kind": "opaque", "python_type": orig,
                "repr": repr(obj)[:400]}
    rec = {"name": name, "kind": "array", "dtype": str(arr.dtype), "orig_dtype": orig,
           "shape": list(arr.shape), "size": int(arr.size), "checksum": checksum(arr)}
    if arr.size and np.issubdtype(arr.dtype, np.floating):
        finite = np.isfinite(arr)
        rec["nan_count"] = int(np.isnan(arr).sum())
        rec["inf_count"] = int(np.isinf(arr).sum())
        if finite.any():
            f = arr[finite]
            rec["min"] = float(f.min())
            rec["max"] = float(f.max())
            rec["mean"] = float(f.mean())
            rec["absmax"] = float(np.abs(f).max())
    elif arr.size and np.issubdtype(arr.dtype, np.integer):
        rec["min"] = int(arr.min())
        rec["max"] = int(arr.max())
    if save_dir is not None:
        p = Path(save_dir) / ("%s.npy" % name)
        np.save(str(p), arr, allow_pickle=False)
        rec["path"] = str(p)
    return rec


def load_inputs(refs, default_device):
    import numpy as np
    torch = try_torch()
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
        stride = ref.get("stride")
        if stride:
            t = torch.as_strided(t, tuple(ref["view_shape"]), tuple(stride),
                                 int(ref.get("storage_offset", 0)))
        if ref.get("requires_grad"):
            t = t.detach().requires_grad_(True)
        out[name] = t
    return out


def seed_all(n):
    random.seed(n)
    try:
        import numpy as np
        np.random.seed(n % (2 ** 32))
    except BaseException:
        pass
    torch = try_torch()
    if torch is not None:
        torch.manual_seed(n)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(n)


def sync(device):
    torch = try_torch()
    if torch is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_module(source, path):
    Path(path).write_text(source, encoding="utf-8")
    mod = types.ModuleType("crucible_candidate")
    mod.__file__ = str(path)
    code = compile(source, str(path), "exec")
    exec(code, mod.__dict__)      # candidate code; this process is disposable
    return mod


def main():
    op = JOB.get("op", "call")
    device = JOB.get("device", "cpu")
    seed = int(JOB.get("seed", 0))
    save_dir = None
    if JOB.get("save_outputs"):
        save_dir = Path(JOB["out_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

    if JOB.get("deterministic"):
        torch = try_torch()
        if torch is not None:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except BaseException as exc:
                sys.stderr.write("deterministic algorithms unavailable: %s\n" % exc)

    seed_all(seed)

    src_path = Path(JOB.get("source_path", "candidate.py"))
    try:
        mod = load_module(JOB.get("source", ""), src_path)
    except BaseException as exc:
        finish_err("import_error", "%s: %s" % (type(exc).__name__, exc), traceback.format_exc())

    if op == "exec":
        value = getattr(mod, "RESULT", None)
        try:
            json.dumps(value, default=str)
        except (TypeError, ValueError) as exc:
            finish_err("result_not_serializable", str(exc))
        finish_ok({"result": value})

    entry = JOB.get("entry")
    fn = getattr(mod, entry, None)
    if fn is None:
        finish_err("missing_entry", "candidate does not define %r" % entry)
    if not callable(fn):
        finish_err("missing_entry", "%r is not callable" % entry)

    try:
        inputs = load_inputs(JOB.get("inputs", {}), device)
    except BaseException as exc:
        finish_err("input_error", "%s: %s" % (type(exc).__name__, exc), traceback.format_exc())

    style = JOB.get("call_style", "kwargs")
    order = JOB.get("arg_order") or sorted(inputs.keys())

    def invoke():
        if style == "kwargs":
            return fn(**inputs)
        if style == "args":
            return fn(*[inputs[k] for k in order])
        if style == "single_dict":
            return fn(inputs)
        raise ValueError("unknown call_style %r" % style)

    if op == "call":
        try:
            sync(device)
            t0 = time.perf_counter()
            out = invoke()
            sync(device)
            dt = time.perf_counter() - t0
        except BaseException as exc:
            finish_err("candidate_exception", "%s: %s" % (type(exc).__name__, exc),
                       traceback.format_exc())
        try:
            records = [describe(n, v, save_dir) for n, v in flatten(out)]
        except BaseException as exc:
            finish_err("output_error", "%s: %s" % (type(exc).__name__, exc),
                       traceback.format_exc())
        finish_ok({"outputs": records,
                   "checksums": {r["name"]: r.get("checksum") for r in records},
                   "returned_type": type(out).__name__,
                   "duration_s": dt,
                   "device": device})

    if op == "time":
        reps = int(JOB.get("reps", 30))
        warmup = int(JOB.get("warmup", 10))
        try:
            for _ in range(warmup):
                invoke()
            sync(device)
            times = []
            out = None
            for _ in range(reps):
                t0 = time.perf_counter()
                out = invoke()
                sync(device)
                times.append(time.perf_counter() - t0)
        except BaseException as exc:
            finish_err("candidate_exception", "%s: %s" % (type(exc).__name__, exc),
                       traceback.format_exc())
        times_sorted = sorted(times)
        mid = len(times_sorted) // 2
        median = (times_sorted[mid] if len(times_sorted) % 2
                  else 0.5 * (times_sorted[mid - 1] + times_sorted[mid]))
        # The final output is described so that a caller can prove the timed
        # work was not dead-code-eliminated.
        records = [describe(n, v, None) for n, v in flatten(out)]
        finish_ok({"reps": reps, "warmup": warmup, "times_s": times,
                   "median_s": median, "min_s": min(times), "max_s": max(times),
                   "device": device,
                   "checksums": {r["name"]: r.get("checksum") for r in records}})

    finish_err("bad_job", "unknown op %r" % op)


try:
    main()
except SystemExit:
    raise
except BaseException as exc:
    try:
        finish_err("child_crash", "%s: %s" % (type(exc).__name__, exc), traceback.format_exc())
    except BaseException:
        sys.stderr.write(traceback.format_exc())
        sys.exit(70)
'''


# --------------------------------------------------------------------------- #
# parent side
# --------------------------------------------------------------------------- #


@dataclass
class SandboxResult:
    ok: bool
    value: Any | None
    stdout: str
    stderr: str
    exit_code: int | None
    duration_s: float
    timed_out: bool
    error: dict[str, Any] | None = None
    workdir: Path | None = None
    argv: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        if self.ok:
            return ""
        if self.timed_out:
            return f"timed out after {self.duration_s:.1f}s"
        if self.error:
            return f"{self.error.get('type', 'error')}: {self.error.get('message', '')}"
        return f"child exited {self.exit_code} without writing a result"

    @property
    def traceback_text(self) -> str:
        return str((self.error or {}).get("traceback", ""))

    def checksums(self) -> dict[str, str]:
        if not self.ok or not isinstance(self.value, dict):
            return {}
        return dict(self.value.get("checksums") or {})

    def outputs(self) -> list[dict[str, Any]]:
        if not self.ok or not isinstance(self.value, dict):
            return []
        return list(self.value.get("outputs") or [])

    def raise_for_status(self) -> SandboxResult:
        if self.ok:
            return self
        raise SandboxError(
            self.message,
            exit_code=self.exit_code,
            stdout=self.stdout[-4000:],
            stderr=self.stderr[-4000:],
            timed_out=self.timed_out,
        )


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """Kill the child and everything it spawned. Best effort, never raises."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("taskkill failed for pid %s: %s", proc.pid, exc)
    try:
        proc.kill()
    except OSError as exc:
        logger.debug("kill failed for pid %s: %s", proc.pid, exc)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("sandbox pid %s did not reap within 15s of kill", proc.pid)


def _prepare(workdir: Path | str | None) -> Path:
    if workdir is None:
        base = Path.cwd() / ".crucible" / "work" / f"sandbox-{uuid.uuid4().hex[:12]}"
    else:
        base = Path(workdir)
    base.mkdir(parents=True, exist_ok=True)
    return base


def run_job(
    job: Mapping[str, Any],
    workdir: Path | str | None = None,
    timeout_s: float = 120.0,
    env_extra: Mapping[str, str] | None = None,
) -> SandboxResult:
    """Execute one sandbox job. Never raises for candidate misbehaviour."""
    wd = _prepare(workdir)
    child_path = wd / "_crucible_child.py"
    child_path.write_text(_CHILD_SOURCE, encoding="utf-8")

    payload: dict[str, Any] = dict(job)
    payload.setdefault("job_version", JOB_VERSION)
    payload["source_path"] = str(wd / "candidate.py")
    payload["result_path"] = str(wd / "result.json")
    payload.setdefault("out_dir", str(wd / "out"))
    job_path = wd / "job.json"
    job_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    result_path = Path(payload["result_path"])
    if result_path.exists():
        result_path.unlink()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = str(int(payload.get("seed", 0)))
    if payload.get("deterministic"):
        # Must be set before torch initialises cuBLAS; hence in the child's env.
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})

    argv = [sys.executable, str(child_path), str(job_path)]
    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(wd),
            env=env,
        )
    except OSError as exc:
        return SandboxResult(
            ok=False,
            value=None,
            stdout="",
            stderr=str(exc),
            exit_code=None,
            duration_s=time.perf_counter() - t0,
            timed_out=False,
            error={"type": "spawn_error", "message": str(exc), "traceback": ""},
            workdir=wd,
            argv=argv,
        )

    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
            stdout, stderr = "", f"could not drain pipes after kill: {exc}"
    duration = time.perf_counter() - t0
    exit_code = proc.poll()

    if timed_out:
        return SandboxResult(
            ok=False,
            value=None,
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=exit_code,
            duration_s=duration,
            timed_out=True,
            error={
                "type": "timeout",
                "message": f"exceeded {timeout_s:.1f}s wall clock; process tree killed",
                "traceback": "",
            },
            workdir=wd,
            argv=argv,
        )

    if not result_path.exists():
        return SandboxResult(
            ok=False,
            value=None,
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=exit_code,
            duration_s=duration,
            timed_out=False,
            error={
                "type": "no_result",
                "message": (
                    f"child exited {exit_code} without writing a result file; "
                    f"stderr tail: {(stderr or '').strip()[-600:]}"
                ),
                "traceback": "",
            },
            workdir=wd,
            argv=argv,
        )

    try:
        parsed = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return SandboxResult(
            ok=False,
            value=None,
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=exit_code,
            duration_s=duration,
            timed_out=False,
            error={"type": "bad_result", "message": str(exc), "traceback": ""},
            workdir=wd,
            argv=argv,
        )

    return SandboxResult(
        ok=bool(parsed.get("ok")),
        value=parsed.get("value"),
        stdout=stdout or "",
        stderr=stderr or "",
        exit_code=exit_code,
        duration_s=duration,
        timed_out=False,
        error=parsed.get("error"),
        workdir=wd,
        argv=argv,
    )


def run_source(
    source: str,
    workdir: Path | str | None = None,
    timeout_s: float = 120.0,
    seed: int = 0,
    env_extra: Mapping[str, str] | None = None,
) -> SandboxResult:
    """Execute a module body out of process; its ``RESULT`` global is returned.

    Used by oracles that need to observe something inside a fresh interpreter
    (dynamo explain output, a distributed launch) rather than call an entry.
    """
    return run_job(
        {"op": "exec", "source": source, "seed": seed},
        workdir=workdir,
        timeout_s=timeout_s,
        env_extra=env_extra,
    )


def input_refs_from(
    inputs: Mapping[str, Any],
    out_dir: Path | str,
    device: str = "cpu",
) -> dict[str, dict[str, Any]]:
    """Materialise a ``make_inputs`` dict into JSON-safe references.

    Tensors and arrays become ``.npy`` files (never pickles); scalars, strings
    and plain containers ride along as JSON.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    import numpy as np

    torch: Any = sys.modules.get("torch")
    if torch is None:
        try:
            import torch as _torch

            torch = _torch
        except ImportError:
            torch = None

    refs: dict[str, dict[str, Any]] = {}
    for name, value in inputs.items():
        if torch is not None and isinstance(value, torch.Tensor):
            t = value.detach().cpu()
            orig = str(t.dtype).replace("torch.", "")
            arr = (t.to(torch.float32) if t.dtype == torch.bfloat16 else t).contiguous().numpy()
            path = d / f"in_{name}.npy"
            np.save(str(path), arr, allow_pickle=False)
            refs[name] = {
                "kind": "npy",
                "path": str(path),
                "torch_dtype": orig,
                "device": device,
                "requires_grad": bool(value.requires_grad),
            }
        elif isinstance(value, np.ndarray):
            path = d / f"in_{name}.npy"
            np.save(str(path), value, allow_pickle=False)
            refs[name] = {"kind": "npy", "path": str(path), "device": device}
        else:
            refs[name] = {"kind": "json", "value": value}
    return refs


def call_entry(
    source: str,
    entry: str,
    inputs: Mapping[str, Any] | None = None,
    *,
    input_refs: Mapping[str, Any] | None = None,
    workdir: Path | str | None = None,
    device: str = "cpu",
    timeout_s: float = 120.0,
    seed: int = 0,
    deterministic: bool = True,
    save_outputs: bool = True,
    call_style: str = "kwargs",
    arg_order: Sequence[str] | None = None,
    env_extra: Mapping[str, str] | None = None,
) -> SandboxResult:
    """Call ``entry(**inputs)`` in a subprocess; return checksums and ``.npy`` outputs.

    Pass either live ``inputs`` (materialised here) or pre-built ``input_refs``.
    """
    wd = _prepare(workdir)
    if input_refs is None:
        refs = input_refs_from(inputs or {}, wd / "in", device=device)
    else:
        refs = dict(input_refs)
    job = {
        "op": "call",
        "source": source,
        "entry": entry,
        "inputs": refs,
        "call_style": call_style,
        "arg_order": list(arg_order) if arg_order else None,
        "device": device,
        "seed": seed,
        "deterministic": deterministic,
        "save_outputs": save_outputs,
    }
    return run_job(job, workdir=wd, timeout_s=timeout_s, env_extra=env_extra)


def time_entry(
    source: str,
    entry: str,
    inputs: Mapping[str, Any] | None = None,
    *,
    input_refs: Mapping[str, Any] | None = None,
    workdir: Path | str | None = None,
    device: str = "cpu",
    reps: int = 30,
    warmup: int = 10,
    timeout_s: float = 600.0,
    seed: int = 0,
    deterministic: bool = False,
    call_style: str = "kwargs",
    arg_order: Sequence[str] | None = None,
    env_extra: Mapping[str, str] | None = None,
) -> SandboxResult:
    """Time ``entry`` over ``reps`` timed repetitions after discarded warmup."""
    wd = _prepare(workdir)
    if input_refs is None:
        refs = input_refs_from(inputs or {}, wd / "in", device=device)
    else:
        refs = dict(input_refs)
    job = {
        "op": "time",
        "source": source,
        "entry": entry,
        "inputs": refs,
        "call_style": call_style,
        "arg_order": list(arg_order) if arg_order else None,
        "device": device,
        "reps": reps,
        "warmup": warmup,
        "seed": seed,
        "deterministic": deterministic,
        "save_outputs": False,
    }
    return run_job(job, workdir=wd, timeout_s=timeout_s, env_extra=env_extra)


__all__ = [
    "SandboxResult",
    "run_job",
    "run_source",
    "call_entry",
    "time_entry",
    "input_refs_from",
    "JOB_VERSION",
]
