"""The O5 multiprocess worker: what one rank actually does.

This module exists as a separate file for one reason: ``torch.multiprocessing.
spawn`` on Windows re-imports the target by its fully qualified name, so the
rank entry point has to be reachable at module scope in an importable module.
Nothing here may be a closure, a lambda, a bound method or anything else that
only exists inside the parent's call stack, and nothing here may assume ``fork``
- there is no inherited memory, so every input arrives through the pickled
argument tuple or through a file.

The division of labour with :mod:`crucible.oracles.o5_nrank` is deliberate:

* this module knows how to build one rank's call and how to summarise its
  result, and it is imported by both launchers, so the threaded simulation and
  the real gloo spawn run *identical* argument marshalling. A simulation that
  marshalled its arguments differently would be calibrating a different program
  than the one production runs.
* ``run_rank`` is the only function that touches ``torch.distributed``. The
  process group is torn down in a ``finally`` even when the candidate raises,
  because a rank that exits holding a group can wedge its peers and turn a
  candidate defect into an infrastructure mystery.

A rank that fails writes ``rank<N>.error.json`` **and** re-raises. The file is
what the parent reads (a non-zero exit code alone says nothing about why), and
the re-raise is what makes the exit code non-zero so the parent knows to look.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The function a candidate's distributed source must expose. It is called with
#: an already-initialised process group and works out its own shard from
#: ``dist.get_rank()``.
DEFAULT_STEP_ENTRY = "train_step_dist"

#: Output keys every distributed step must return. ``grad_norms`` is not
#: optional: the per-shard grad-norm defect is visible there steps before it is
#: visible in the loss, which is the whole reason O5 grades a norm series.
REQUIRED_OUTPUTS: tuple[str, ...] = ("losses", "grad_norms", "w")

#: Payload entries that are tensors, and the dtype they are rebuilt at. Anything
#: not named here crosses as a plain JSON scalar/list and is passed through
#: untouched, so a candidate signature can take ints, floats and strings.
TENSOR_DTYPES: dict[str, str] = {
    "x": "float32",
    "y": "int64",
    "w0": "float32",
    "w1_0": "float32",
    "w2_0": "float32",
}


# --------------------------------------------------------------------------- #
# loading the candidate's distributed source
# --------------------------------------------------------------------------- #


def load_module_from_path(path: Path | str, name: str) -> Any:
    """Import ``path`` as a module called ``name``.

    Registered in ``sys.modules`` before execution: a module that is not there
    breaks dataclasses, pickling and ``inspect.getsource`` inside the candidate
    for reasons that look nothing like the real cause.
    """
    import importlib.util

    p = Path(path)
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"no import machinery could load a module from {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_source(source: str, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(source, encoding="utf-8")
    return p


def resolve_entry(module: Any, entry: str) -> Any:
    fn = getattr(module, entry, None)
    if fn is None:
        public = sorted(n for n in vars(module) if not n.startswith("_") and callable(vars(module)[n]))
        raise AttributeError(
            f"the distributed source defines no {entry!r}; callables at module scope: {public}"
        )
    if not callable(fn):
        raise TypeError(f"{entry!r} is a {type(fn).__name__}, not a callable")
    return fn


# --------------------------------------------------------------------------- #
# argument marshalling
# --------------------------------------------------------------------------- #


def materialize(name: str, value: Any) -> Any:
    """Rebuild one payload entry in this process."""
    dtype_name = TENSOR_DTYPES.get(name)
    if dtype_name is None:
        return value
    import torch

    return torch.tensor(value, dtype=getattr(torch, dtype_name))


def build_kwargs(fn: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``payload`` this entry actually accepts, as tensors.

    Candidate step functions differ in arity - a one-layer step takes ``w0``, a
    two-layer step takes ``w1_0``/``w2_0`` - so the payload is a superset and
    the signature selects from it. A required parameter the payload cannot fill
    is an error here rather than a confusing ``TypeError`` from deep inside the
    candidate.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    takes_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    kwargs: dict[str, Any] = {}
    for name, value in payload.items():
        if name in params or takes_var_kw:
            kwargs[name] = materialize(name, value)

    missing = [
        name
        for name, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if missing:
        raise TypeError(
            f"the distributed entry {getattr(fn, '__name__', '?')!r} requires "
            f"{missing}, which the O5 workload payload does not supply "
            f"(payload keys: {sorted(payload)})"
        )
    return kwargs


def summarize(out: Any, rank: int) -> dict[str, Any]:
    """One rank's result as plain floats, ready for JSON.

    Nothing tensor-shaped crosses a process boundary: the parent never
    unpickles candidate-produced objects.
    """
    import torch

    if not isinstance(out, dict):
        raise TypeError(
            f"the distributed entry returned {type(out).__name__}; a mapping with "
            f"keys {list(REQUIRED_OUTPUTS)} is required"
        )
    missing = [k for k in REQUIRED_OUTPUTS if k not in out]
    if missing:
        raise KeyError(
            f"the distributed entry returned no {missing}; O5 grades the loss series, "
            "the global grad-norm series and the final parameters"
        )
    summary: dict[str, Any] = {"rank": int(rank)}
    for key in REQUIRED_OUTPUTS:
        flat = torch.as_tensor(out[key]).detach().to(torch.float64).reshape(-1)
        summary[key] = [float(v) for v in flat]
    return summary


def call_entry(module: Any, entry: str, payload: dict[str, Any], rank: int) -> dict[str, Any]:
    """Resolve, call and summarise. Shared by both launchers on purpose."""
    fn = resolve_entry(module, entry)
    return summarize(fn(**build_kwargs(fn, payload)), rank)


# --------------------------------------------------------------------------- #
# the rank entry point
# --------------------------------------------------------------------------- #


def error_payload(rank: int, where: str, exc: BaseException) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "where": where,
        "type": type(exc).__name__,
        "message": str(exc)[:4000],
        "traceback": traceback.format_exc()[-4000:],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        logger.warning("rank result could not be written to %s: %s", path, exc)


def run_rank(
    rank: int,
    world_size: int,
    job_dir: str,
    source_path: str,
    entry: str = DEFAULT_STEP_ENTRY,
    payload: dict[str, Any] | None = None,
    backend: str = "gloo",
    init_timeout_s: float = 300.0,
) -> None:
    """Run one rank under a real process group. Called by ``spawn`` as ``fn(i, *args)``.

    Rendezvous is a ``FileStore`` under ``job_dir``: no port to collide with
    another run on the same machine, no localhost socket to be refused by a
    firewall, and it behaves identically on Windows and Linux.
    """
    import torch
    import torch.distributed as dist

    job = Path(job_dir)
    job.mkdir(parents=True, exist_ok=True)
    body = dict(payload or {})

    try:
        # One thread per rank: N ranks each grabbing every core turns a
        # measurement into a scheduling artefact, and BLAS thread counts change
        # float summation order, which is exactly the quantity O5 calibrates.
        torch.set_num_threads(1)
        torch.manual_seed(int(body.get("seed", 1234)) + int(rank))
        store = dist.FileStore(str(job / "store"), int(world_size))
        dist.init_process_group(
            backend=str(backend),
            store=store,
            rank=int(rank),
            world_size=int(world_size),
            timeout=timedelta(seconds=float(init_timeout_s)),
        )
    except Exception as exc:  # noqa: BLE001 - recorded then re-raised
        _write_json(job / f"rank{rank}.error.json", error_payload(rank, "init_process_group", exc))
        raise

    try:
        module = load_module_from_path(source_path, f"crucible_o5_candidate_rank{rank}")
        summary = call_entry(module, str(entry), body, int(rank))
        _write_json(job / f"rank{rank}.json", summary)
    except Exception as exc:  # noqa: BLE001 - recorded then re-raised
        _write_json(job / f"rank{rank}.error.json", error_payload(rank, "candidate", exc))
        raise
    finally:
        try:
            dist.destroy_process_group()
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the real failure
            logger.debug("rank %d could not destroy its process group: %s", rank, exc)


def _main() -> int:
    """Direct invocation, for debugging a rank without the parent harness."""
    payload_path = os.environ.get("CRUCIBLE_O5_PAYLOAD", "")
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8")) if payload_path else {}
    run_rank(
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
        os.environ["CRUCIBLE_O5_JOB_DIR"],
        os.environ["CRUCIBLE_O5_SOURCE"],
        os.environ.get("CRUCIBLE_O5_ENTRY", DEFAULT_STEP_ENTRY),
        payload,
        os.environ.get("CRUCIBLE_O5_BACKEND", "gloo"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "DEFAULT_STEP_ENTRY",
    "REQUIRED_OUTPUTS",
    "TENSOR_DTYPES",
    "build_kwargs",
    "call_entry",
    "error_payload",
    "load_module_from_path",
    "materialize",
    "resolve_entry",
    "run_rank",
    "summarize",
    "write_source",
]
