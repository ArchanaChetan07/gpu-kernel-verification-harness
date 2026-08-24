"""The shared data model. Every module speaks these types.

Two design commitments live here rather than in prose:

1. ``OracleResult`` cannot carry a non-PASS verdict without a reason. A verdict
   with no explanation is indistinguishable from a fabricated one.
2. ``TaskVerdict.combine`` puts SKIP strictly above PASS. A task whose oracles
   could not run is not a passing task, and no amount of downstream reporting
   can turn it into one.
"""

from __future__ import annotations

import hashlib
import json
import platform as _platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Tier = Literal["T1", "T2", "T3", "T4", "T5", "T6"]
Domain = Literal[
    "triton",
    "pallas",
    "cuda",
    "jax",
    "pytorch",
    "distributed",
    "data_pipeline",
    "checkpointing",
]
VerdictStr = Literal["PASS", "FAIL", "SKIP", "ERROR"]

#: Worst-first. Index in this tuple is the precedence used by ``combine``.
VERDICT_PRECEDENCE: tuple[VerdictStr, ...] = ("ERROR", "FAIL", "SKIP", "PASS")


def _stable_hash(obj: Any, length: int = 16) -> str:
    payload = json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ShapeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)

    def key(self) -> str:
        """Deterministic id of the *inputs*, independent of the label."""
        return _stable_hash(self.kwargs)

    def __str__(self) -> str:
        return f"{self.name}({self.key()})"


class SeedRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_id: str
    module: str
    entry: str
    content_sha256: str


class MutationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cls: str
    site: str
    params: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class Witness(BaseModel):
    """Proof that the mutation is observable: the shape on which it breaks."""

    model_config = ConfigDict(extra="forbid")

    shape: ShapeSpec
    max_abs_err: float
    max_rel_err: float
    tolerance: float
    baseline_checksum: str
    mutant_checksum: str
    #: "performance" exists because T4 means "correct output, wrong speed": such a
    #: mutation is byte-identical to the baseline, so every correctness-based
    #: witness kind below is blind to it and the whole tier would be
    #: undiscoverable by construction.
    kind: Literal["numeric", "exception", "shape", "hang", "loss_curve", "performance"]
    detail: str = ""


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    weight: float
    anchors: dict[int, str]
    auto_probe: str | None = None
    probe_thresholds: list[tuple[float, int]] = Field(default_factory=list)
    machine_probed: bool = False

    @model_validator(mode="after")
    def _check(self) -> RubricCriterion:
        for required in (0, 3, 5):
            if required not in self.anchors:
                raise ValueError(f"criterion {self.id!r} must define anchors for scores 0, 3 and 5")
        if self.weight < 0:
            raise ValueError(f"criterion {self.id!r} has negative weight")
        bounds = [b for b, _ in self.probe_thresholds]
        if bounds != sorted(bounds):
            raise ValueError(f"criterion {self.id!r}: probe_thresholds must ascend by bound")
        if self.machine_probed and not self.auto_probe:
            raise ValueError(f"criterion {self.id!r}: machine_probed requires an auto_probe")
        return self


class RubricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criteria: list[RubricCriterion] = Field(default_factory=list)
    version: str = "1.0"

    def total_weight(self) -> float:
        return float(sum(c.weight for c in self.criteria))

    def machine_probed_share(self) -> float:
        if not self.criteria:
            return 0.0
        return sum(1 for c in self.criteria if c.machine_probed) / len(self.criteria)


class CalibrationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    k: int
    n_samples: int
    n_correct: int
    pass_at_1: float
    pass_at_k: float
    route: Literal["reject", "gold", "frontier", "escalate"]
    rationale: str
    trace_stats: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    schema_version: str = "1.0"
    created_utc: str
    seed_source: SeedRef
    domain: Domain
    failure_tier: Tier
    mutation: MutationSpec
    baseline_code: str
    mutant_code: str
    ground_truth_diff: str
    witness: Witness | None = None
    detect_shapes: list[ShapeSpec] = Field(default_factory=list)
    decoy_shapes: list[ShapeSpec] = Field(default_factory=list)
    prompt: str = ""
    oracles: list[str] = Field(default_factory=list)
    rubric: RubricSpec = Field(default_factory=RubricSpec)
    calibration: CalibrationRecord | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_prompt_leaks(self) -> Task:
        """The prompt must not hand over the answer.

        Detect shapes are the withheld grading set; leaking one into the prompt
        turns a discovery task into a lookup task.
        """
        if self.prompt:
            for shape in self.detect_shapes:
                # Short names ("s1") collide with ordinary prose; only names
                # distinctive enough to be a real leak are treated as one.
                if len(shape.name) >= 4 and shape.name in self.prompt:
                    raise ValueError(
                        f"prompt leaks detect_shape {shape.name!r}; detect shapes are withheld"
                    )
        return self

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100000,
        )

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> Task:
        text = Path(path).read_text(encoding="utf-8")
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            # Translate at the boundary. yaml.YAMLError is neither OSError nor
            # ValueError, so it escapes every caller's exception handling and
            # a single corrupt file aborts a whole bank sweep. One unreadable
            # task must be reported and skipped, never fatal.
            raise ValueError(f"{path} is not valid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path} does not contain a task mapping")
        return cls.model_validate(raw)

    @classmethod
    def from_yaml(cls, text: str) -> Task:
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"task YAML is not valid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("task YAML does not contain a mapping")
        return cls.model_validate(raw)

    def cell_id(self) -> str:
        return f"{self.failure_tier}/{self.domain}"


class OracleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oracle: str
    verdict: VerdictStr
    reason: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    duration_s: float = 0.0
    capabilities_used: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reason_required(self) -> OracleResult:
        if self.verdict != "PASS" and not self.reason.strip():
            raise ValueError(
                f"oracle {self.oracle!r}: verdict {self.verdict} requires a non-empty reason"
            )
        return self


class EnvironmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caps: dict[str, Any] = Field(default_factory=dict)
    python: str = ""
    torch: str = ""
    host: str = ""
    utc: str = ""

    @classmethod
    def capture(cls, caps: Any = None) -> EnvironmentRecord:
        """Snapshot the machine. ``caps`` is a Capabilities or a plain dict."""
        if caps is None:
            cap_dict: dict[str, Any] = {}
            torch_version = str(getattr(sys.modules.get("torch"), "__version__", "unknown"))
        elif isinstance(caps, dict):
            cap_dict = dict(caps)
            torch_version = str(cap_dict.get("torch_version", "unknown"))
        else:
            cap_dict = caps.as_dict() if hasattr(caps, "as_dict") else dict(vars(caps))
            torch_version = str(getattr(caps, "torch_version", "unknown"))
        try:
            host = socket.gethostname()
        except OSError:
            host = "unknown"
        return cls(
            caps=cap_dict,
            python=f"{sys.version.split()[0]} ({_platform.platform()})",
            torch=torch_version,
            host=host,
            utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


class TaskVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    verdict: VerdictStr
    executed: bool
    oracle_results: list[OracleResult] = Field(default_factory=list)
    environment: EnvironmentRecord = Field(default_factory=EnvironmentRecord)
    duration_s: float = 0.0
    candidate_sha256: str = ""

    @staticmethod
    def combine(results: Iterable[OracleResult | str]) -> VerdictStr:
        """ERROR > FAIL > SKIP > PASS.

        An empty result set is SKIP, not PASS: nothing was verified.
        """
        seen: set[str] = set()
        for r in results:
            seen.add(r if isinstance(r, str) else r.verdict)
        if not seen:
            return "SKIP"
        for v in VERDICT_PRECEDENCE:
            if v in seen:
                return v
        return "SKIP"

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent, sort_keys=False)

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> TaskVerdict:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def failures(self) -> list[OracleResult]:
        return [r for r in self.oracle_results if r.verdict in ("FAIL", "ERROR")]

    def skips(self) -> list[OracleResult]:
        return [r for r in self.oracle_results if r.verdict == "SKIP"]


__all__ = [
    "Tier",
    "Domain",
    "VerdictStr",
    "VERDICT_PRECEDENCE",
    "ShapeSpec",
    "SeedRef",
    "MutationSpec",
    "Witness",
    "RubricCriterion",
    "RubricSpec",
    "CalibrationRecord",
    "Task",
    "OracleResult",
    "EnvironmentRecord",
    "TaskVerdict",
    "sha256_text",
]
