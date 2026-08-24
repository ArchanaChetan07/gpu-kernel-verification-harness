"""What this machine can actually do.

Everything downstream gates on this. The rule that shapes the whole module: a
probe reports what it *observed*, never what is *typical*. If triton cannot be
imported we keep the DLL error verbatim; if ncu is absent the path is None, not
an empty string; if clocks could not be locked we say so and the perf oracle
downgrades its verdict.

Probes that can crash the interpreter (triton's DLL loader, inductor's C++
backend) run in a subprocess so a segfault cannot take the parent down. Every
subprocess probe has a hard timeout and returns a structured error instead of
raising.
"""

from __future__ import annotations

import json
import logging
import os
import platform as _platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .errors import CapabilityError

logger = logging.getLogger(__name__)

_SENTINEL = "CRUCIBLE_PROBE_JSON:"
_PROBE_TIMEOUT_S = 60.0
_SMI_TIMEOUT_S = 20.0

CACHE_DIR = Path.home() / ".crucible"
CACHE_PATH = CACHE_DIR / "caps.json"

#: Capability names that may appear in ``Oracle.required_caps`` / ``missing()``.
CAP_NAMES: frozenset[str] = frozenset(
    {
        "cuda",
        "cuda_device_count",
        "device_name",
        "cuda_version",
        "triton",
        "ncu",
        "can_lock_clocks",
        "gloo",
        "nccl",
        "cxx_compiler",
        "inductor_cpu",
        "inductor_cuda",
    }
)


@dataclass(frozen=True)
class Capabilities:
    cuda: bool
    cuda_device_count: int
    device_name: str | None
    cuda_version: str | None
    triton: bool
    triton_error: str | None
    ncu: str | None
    can_lock_clocks: bool
    gloo: bool
    nccl: bool
    cxx_compiler: str | None
    inductor_cpu: bool
    inductor_cuda: bool
    torch_version: str
    platform: str
    # Diagnostics, not gates: why a false capability is false.
    details: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.details is None:
            object.__setattr__(self, "details", {})

    def missing(self, required: Iterable[str]) -> list[str]:
        """Names among ``required`` whose capability is falsy or None.

        An unknown name is a programming error and raises: silently returning
        "not missing" for a typo would let an oracle claim caps it never had.
        """
        out: list[str] = []
        for name in required:
            if name not in CAP_NAMES:
                raise CapabilityError(
                    f"unknown capability name {name!r}; valid names: {sorted(CAP_NAMES)}",
                    missing=[name],
                )
            if not getattr(self, name):
                out.append(name)
        return out

    def detail(self, name: str) -> str:
        """Human-readable reason a capability is unavailable ('' if none recorded)."""
        return self.details.get(name, "")

    def explain(self, names: Iterable[str]) -> str:
        parts = []
        for n in names:
            d = self.detail(n)
            parts.append(f"{n}: {d}" if d else n)
        return "; ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        dev = self.device_name or "no cuda device"
        return (
            f"torch {self.torch_version} on {self.platform}; {dev}; "
            f"cuda={self.cuda} triton={self.triton} ncu={'yes' if self.ncu else 'no'} "
            f"gloo={self.gloo} nccl={self.nccl} "
            f"inductor(cpu={self.inductor_cpu},cuda={self.inductor_cuda})"
        )


# --------------------------------------------------------------------------- #
# subprocess probe plumbing
# --------------------------------------------------------------------------- #


def _run_probe(source: str, timeout_s: float = _PROBE_TIMEOUT_S) -> dict[str, Any]:
    """Run a probe script in a fresh interpreter; return its JSON payload.

    The probe prints one ``CRUCIBLE_PROBE_JSON:{...}`` line. Anything else on
    stdout/stderr (DLL loader noise, warnings) is ignored except when no payload
    arrives, in which case the tail of stderr becomes the error text.
    """
    with tempfile.TemporaryDirectory(prefix="crucible-probe-") as td:
        script = Path(td) / "probe.py"
        script.write_text(source, encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            # Not -I: user site-packages may be where torch lives, and a false
            # negative here would silently disable a real capability.
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                env=env,
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"probe timed out after {timeout_s:.0f}s"}
        except OSError as exc:
            return {"ok": False, "error": f"probe could not be launched: {exc}"}

    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(_SENTINEL):
            try:
                payload = json.loads(line[len(_SENTINEL) :])
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"probe emitted malformed JSON: {exc}"}
            if isinstance(payload, dict):
                return payload
            return {"ok": False, "error": "probe payload was not an object"}

    tail = (proc.stderr or "").strip().splitlines()
    detail = " | ".join(tail[-4:]) if tail else "no output"
    return {
        "ok": False,
        "error": f"probe produced no result (exit {proc.returncode}): {detail}",
    }


_PROBE_PRELUDE = f'''
import json, sys, traceback
def emit(payload):
    sys.stdout.write("{_SENTINEL}" + json.dumps(payload, default=str) + "\\n")
    sys.stdout.flush()
'''

_TRITON_PROBE = (
    _PROBE_PRELUDE
    + '''
try:
    import triton
    import triton.language as tl
except BaseException as exc:                      # DLL load failures are not Exceptions
    emit({"ok": False, "error": "import triton failed: %s: %s" % (type(exc).__name__, exc)})
    raise SystemExit(0)

try:
    import torch
except BaseException as exc:
    emit({"ok": False, "error": "import torch failed: %s: %s" % (type(exc).__name__, exc)})
    raise SystemExit(0)

@triton.jit
def _double(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)

try:
    if not torch.cuda.is_available():
        emit({"ok": False, "error": "triton imported but no CUDA device to compile a kernel for",
              "version": getattr(triton, "__version__", "?")})
        raise SystemExit(0)
    x = torch.arange(64, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    _double[(1,)](x, y, x.numel(), BLOCK=64)
    torch.cuda.synchronize()
    good = bool(torch.equal(y, x * 2.0))
    emit({"ok": good,
          "error": None if good else "trivial kernel compiled but produced wrong values",
          "version": getattr(triton, "__version__", "?")})
except BaseException as exc:
    emit({"ok": False, "error": "kernel compile/launch failed: %s: %s" % (type(exc).__name__, exc),
          "traceback": traceback.format_exc()[-2000:]})
'''
)


def _inductor_probe_source(device: str) -> str:
    return (
        _PROBE_PRELUDE
        + f'''
DEV = "{device}"
try:
    import torch
except BaseException as exc:
    emit({{"ok": False, "error": "import torch failed: %s" % exc}})
    raise SystemExit(0)

try:
    if DEV == "cuda" and not torch.cuda.is_available():
        emit({{"ok": False, "error": "no CUDA device available"}})
        raise SystemExit(0)

    def f(x):
        return x * 2.0 + 1.0

    g = torch.compile(f, fullgraph=True)
    x = torch.randn(64, device=DEV)
    y = g(x)
    if DEV == "cuda":
        torch.cuda.synchronize()
    good = bool(torch.allclose(y, f(x)))
    emit({{"ok": good, "error": None if good else "compiled fn returned wrong values"}})
except BaseException as exc:
    import traceback as _tb
    emit({{"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
          "traceback": _tb.format_exc()[-2000:]}})
'''
    )


_TORCH_INFO_PROBE = (
    _PROBE_PRELUDE
    + '''
out = {"ok": True, "cuda": False, "count": 0, "device_name": None, "cuda_version": None,
       "torch_version": None, "gloo": False, "nccl": False, "error": None}
try:
    import torch
    out["torch_version"] = torch.__version__
    out["cuda_version"] = torch.version.cuda
    try:
        out["cuda"] = bool(torch.cuda.is_available())
        if out["cuda"]:
            out["count"] = int(torch.cuda.device_count())
            if out["count"] > 0:
                out["device_name"] = torch.cuda.get_device_name(0)
    except BaseException as exc:
        out["error"] = "cuda query failed: %s" % exc
    try:
        import torch.distributed as dist
        out["gloo"] = bool(dist.is_gloo_available())
        out["nccl"] = bool(dist.is_nccl_available())
    except BaseException as exc:
        out["error"] = (out["error"] or "") + " dist query failed: %s" % exc
except BaseException as exc:
    out["ok"] = False
    out["error"] = "import torch failed: %s" % exc
emit(out)
'''
)


# --------------------------------------------------------------------------- #
# non-python probes
# --------------------------------------------------------------------------- #

_NSIGHT_GLOBS = (
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute *",
    r"C:\Program Files (x86)\NVIDIA Corporation\Nsight Compute *",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin",
)


def _find_ncu() -> str | None:
    """ncu on PATH, else the standard Nsight Compute install locations."""
    for name in ("ncu", "ncu.bat", "ncu.exe"):
        found = shutil.which(name)
        if found:
            return found
    if os.name == "nt":
        for pattern in _NSIGHT_GLOBS:
            base = Path(pattern).parent
            glob = Path(pattern).name
            if not base.exists():
                continue
            for candidate_dir in sorted(base.glob(glob), reverse=True):
                for exe in ("ncu.bat", "ncu.exe"):
                    p = candidate_dir / exe
                    if p.exists():
                        return str(p)
    return None


def _find_cxx() -> tuple[str | None, str]:
    for name in ("cl", "g++", "gcc", "clang++", "clang"):
        found = shutil.which(name)
        if found:
            return found, ""
    return None, "none of cl/g++/gcc/clang++/clang on PATH"


def run_nvidia_smi(args: list[str], timeout_s: float = _SMI_TIMEOUT_S) -> tuple[int, str, str]:
    """(returncode, stdout, stderr). Returns a non-zero code instead of raising."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return 127, "", "nvidia-smi not on PATH"
    try:
        proc = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"nvidia-smi {' '.join(args)} timed out after {timeout_s:.0f}s"
    except OSError as exc:
        return 126, "", f"nvidia-smi could not be launched: {exc}"
    return proc.returncode, proc.stdout or "", proc.stderr or ""


_smi = run_nvidia_smi


def probe_clock_lock() -> tuple[bool, str]:
    """Can we pin the graphics clock for reproducible timing?

    Two stages, both non-destructive. First a pure query of the supported clock
    range (no privilege needed). Only if that works do we attempt a real lock at
    the reported maximum, and the reset is issued unconditionally in ``finally``
    so this probe never leaves the GPU pinned. Any non-zero exit -> False; we do
    not assume the caller is an administrator.
    """
    rc, out, err = _smi(["--query-gpu=clocks.max.graphics", "--format=csv,noheader,nounits"])
    if rc != 0:
        return False, (err or out or "nvidia-smi clock query failed").strip()[:400]
    first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    try:
        max_mhz = int(float(first.split()[0]))
    except (ValueError, IndexError):
        return False, f"could not parse max graphics clock from {first!r}"

    locked = False
    reason = ""
    try:
        rc2, out2, err2 = _smi(["-lgc", f"{max_mhz},{max_mhz}"])
        locked = rc2 == 0
        if not locked:
            reason = (err2 or out2 or f"nvidia-smi -lgc exited {rc2}").strip()[:400]
    finally:
        # Always restore, including when the lock partially applied.
        rc3, _, err3 = _smi(["-rgc"])
        if locked and rc3 != 0:
            reason = f"locked but reset failed (rc={rc3}): {err3.strip()[:200]}"
            logger.warning("nvidia-smi -rgc failed after probe lock: %s", err3.strip()[:200])
    return locked, reason


# --------------------------------------------------------------------------- #
# detect
# --------------------------------------------------------------------------- #

_MEMO: dict[str, Capabilities] = {}


def _cache_key(torch_version: str, plat: str) -> str:
    return f"torch={torch_version}|platform={plat}|py={sys.version.split()[0]}"


def _read_cache(key: str) -> Capabilities | None:
    if not CACHE_PATH.exists():
        return None
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("capability cache unreadable (%s); re-probing", exc)
        return None
    if not isinstance(raw, dict) or raw.get("key") != key:
        return None
    payload = raw.get("caps")
    if not isinstance(payload, dict):
        return None
    names = {f.name for f in fields(Capabilities)}
    if not names.issubset(payload.keys()):
        return None
    try:
        return Capabilities(**{k: payload[k] for k in names})
    except TypeError as exc:
        logger.debug("capability cache incompatible (%s); re-probing", exc)
        return None


def _write_cache(key: str, caps: Capabilities) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(
                {
                    "key": key,
                    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "caps": caps.as_dict(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("could not write capability cache to %s: %s", CACHE_PATH, exc)


def _probe_all(probe_timeout_s: float) -> Capabilities:
    details: dict[str, str] = {}
    plat = _platform.platform()

    info = _run_probe(_TORCH_INFO_PROBE, timeout_s=probe_timeout_s)
    torch_version = str(info.get("torch_version") or "unavailable")
    cuda = bool(info.get("cuda", False))
    count = int(info.get("count", 0) or 0)
    device_name = info.get("device_name") or None
    cuda_version = info.get("cuda_version") or None
    gloo = bool(info.get("gloo", False))
    nccl = bool(info.get("nccl", False))
    if info.get("error"):
        details["cuda"] = str(info["error"])[:400]
    if not cuda:
        details.setdefault("cuda", "torch.cuda.is_available() returned False")
    if not nccl:
        details.setdefault("nccl", "torch.distributed.is_nccl_available() returned False")
    if not gloo:
        details.setdefault("gloo", "torch.distributed.is_gloo_available() returned False")

    triton_res = _run_probe(_TRITON_PROBE, timeout_s=probe_timeout_s)
    triton_ok = bool(triton_res.get("ok", False))
    triton_error = None if triton_ok else str(triton_res.get("error") or "unknown triton failure")
    if triton_error:
        details["triton"] = triton_error[:1000]

    ncu = _find_ncu()
    if ncu is None:
        details["ncu"] = "ncu not on PATH and not found in standard Nsight Compute locations"

    can_lock, lock_reason = probe_clock_lock()
    if not can_lock:
        details["can_lock_clocks"] = lock_reason or "clock lock probe failed"

    cxx, cxx_reason = _find_cxx()
    if cxx is None:
        details["cxx_compiler"] = cxx_reason

    cpu_res = _run_probe(_inductor_probe_source("cpu"), timeout_s=probe_timeout_s)
    inductor_cpu = bool(cpu_res.get("ok", False))
    if not inductor_cpu:
        details["inductor_cpu"] = str(cpu_res.get("error") or "unknown inductor failure")[:1000]

    if cuda:
        cuda_res = _run_probe(_inductor_probe_source("cuda"), timeout_s=probe_timeout_s)
        inductor_cuda = bool(cuda_res.get("ok", False))
        if not inductor_cuda:
            details["inductor_cuda"] = str(cuda_res.get("error") or "unknown inductor failure")[:1000]
    else:
        inductor_cuda = False
        details["inductor_cuda"] = "skipped: no CUDA device"

    return Capabilities(
        cuda=cuda,
        cuda_device_count=count,
        device_name=device_name,
        cuda_version=cuda_version,
        triton=triton_ok,
        triton_error=triton_error,
        ncu=ncu,
        can_lock_clocks=can_lock,
        gloo=gloo,
        nccl=nccl,
        cxx_compiler=cxx,
        inductor_cpu=inductor_cpu,
        inductor_cuda=inductor_cuda,
        torch_version=torch_version,
        platform=plat,
        details=details,
    )


def detect(refresh: bool = False, probe_timeout_s: float = _PROBE_TIMEOUT_S) -> Capabilities:
    """Probe this machine. Cached in-process and in ``~/.crucible/caps.json``.

    The cache key includes the torch version, the platform string and the Python
    version, so upgrading torch invalidates stale answers automatically.
    ``refresh=True`` forces a full re-probe and rewrites the cache.
    """
    plat = _platform.platform()
    # Cheap version read for the key; the authoritative one comes from the probe.
    torch_version = _torch_version_hint()
    key = _cache_key(torch_version, plat)

    if not refresh:
        memo = _MEMO.get(key)
        if memo is not None:
            return memo
        cached = _read_cache(key)
        if cached is not None:
            _MEMO[key] = cached
            return cached

    caps = _probe_all(probe_timeout_s)
    real_key = _cache_key(caps.torch_version, caps.platform)
    _MEMO[real_key] = caps
    _MEMO[key] = caps
    _write_cache(real_key, caps)
    return caps


def _torch_version_hint() -> str:
    """Torch version without importing torch if it is already imported elsewhere."""
    mod = sys.modules.get("torch")
    if mod is not None:
        return str(getattr(mod, "__version__", "unknown"))
    try:
        from importlib.metadata import version

        return version("torch")
    except (ImportError, LookupError, OSError) as exc:
        logger.debug("could not read torch version metadata: %s", exc)
        return "unknown"


def clear_cache() -> None:
    """Drop the in-process memo and the on-disk cache (used by ``doctor --refresh``)."""
    _MEMO.clear()
    try:
        CACHE_PATH.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not remove %s: %s", CACHE_PATH, exc)


__all__ = [
    "Capabilities",
    "detect",
    "CAP_NAMES",
    "clear_cache",
    "probe_clock_lock",
    "run_nvidia_smi",
    "CACHE_PATH",
]
