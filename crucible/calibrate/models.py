"""Target models for calibration, and the offline default.

The rule this module exists to enforce: **no network call may happen unless the
caller explicitly asked for a non-stub model.** ``StubModel`` is the default, it
is fully deterministic, and it never imports an HTTP client. The real adapters
construct their client *inside* ``generate``, after checking the environment, so
merely importing this module - or resolving a model name and never calling it -
cannot open a socket. A missing API key raises ``ModelUnavailable`` with the
variable name in the message; it never falls back to the stub, because a silent
fallback would report stub pass rates under a real model's name.

Correctness is not decided here. ``grade_completion`` takes a ``verify``
callable so calibration can reuse the real oracle harness; this module never
imports ``crucible.oracles`` (that would make ``oracles -> calibrate -> oracles``
a cycle) and never guesses. If no verifier is supplied and the completion is not
a stub completion, grading raises rather than defaulting to "incorrect".
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from ..errors import CrucibleError

logger = logging.getLogger(__name__)

#: Marker the stub writes into a completion it intends to be graded correct.
STUB_CORRECT_MARKER = "CRUCIBLE-STUB-CORRECT"
#: Marker the stub writes into a completion it intends to be graded incorrect.
STUB_INCORRECT_MARKER = "CRUCIBLE-STUB-INCORRECT"
#: Prefix a real adapter uses when the provider declined to answer at all.
REFUSAL_MARKER = "CRUCIBLE-MODEL-REFUSAL"

#: Tag used to carry a task id inside a prompt for models whose ``generate``
#: signature cannot take one (the protocol only promises prompt/n/temperature).
TASK_ID_TAG = "crucible-task-id:"

#: Current default target for the Anthropic adapter.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

#: Anthropic models that reject ``temperature``/``top_p``/``top_k`` outright
#: (the request 400s). The protocol still hands us a temperature, so the adapter
#: drops it for these and says so once, rather than failing every request.
_NO_SAMPLING_PARAMS: tuple[str, ...] = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)

_DEFAULT_SYSTEM = (
    "You are repairing a defect in a machine-learning systems kernel. "
    "Return the corrected implementation in a single fenced code block. "
    "Do not explain unless asked."
)

__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "REFUSAL_MARKER",
    "STUB_CORRECT_MARKER",
    "STUB_INCORRECT_MARKER",
    "TASK_ID_TAG",
    "AnthropicModel",
    "Grade",
    "LocalModel",
    "ModelUnavailable",
    "OpenAIModel",
    "StubModel",
    "TargetModel",
    "embed_task_id",
    "extract_task_id",
    "grade_completion",
    "resolve_model",
    "stub_verify",
]


class ModelUnavailable(CrucibleError):
    """The requested target model cannot be reached from this process.

    Raised for a missing API key, a missing SDK, or a missing endpoint. It is
    never caught internally to substitute a different model.
    """


@runtime_checkable
class TargetModel(Protocol):
    """What calibration needs from a model."""

    name: str

    def generate(self, prompt: str, n: int, temperature: float) -> list[str]:
        """Return exactly ``n`` completions for ``prompt``."""
        ...


# --------------------------------------------------------------------------- #
# task-id plumbing
# --------------------------------------------------------------------------- #


def embed_task_id(task_id: str, prompt: str) -> str:
    """Prefix ``prompt`` with a machine-readable task id line."""
    return f"{TASK_ID_TAG} {task_id}\n{prompt}"


def extract_task_id(prompt: str) -> str | None:
    """The task id embedded by :func:`embed_task_id`, or None."""
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(TASK_ID_TAG):
            value = stripped[len(TASK_ID_TAG) :].strip()
            return value or None
    return None


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #

#: ``verify(completion)`` or ``verify(completion, task)``; both arities accepted.
VerifyFn = Callable[..., Any]


@dataclass(frozen=True)
class Grade:
    correct: bool
    graded_by: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"correct": self.correct, "graded_by": self.graded_by, "detail": self.detail}


def stub_verify(completion: str, task: Any = None) -> bool:
    """Read the stub's own correctness marker out of a completion.

    Raises on a completion that carries no marker: this verifier is only valid
    for stub output, and silently grading a real model's answer as incorrect
    would be a fabricated measurement.
    """
    del task
    if STUB_CORRECT_MARKER in completion:
        return True
    if STUB_INCORRECT_MARKER in completion:
        return False
    raise ValueError(
        "stub_verify was given a completion with no stub marker; "
        "supply a real verify callable for non-stub models"
    )


def _verify_arity(fn: VerifyFn) -> int:
    """How many positional arguments ``fn`` will accept (1 or 2)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 1
    positional = 0
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            return 2
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
    return 2 if positional >= 2 else 1


def grade_completion(
    completion: str,
    *,
    verify: VerifyFn | None = None,
    task: Any = None,
) -> Grade:
    """Decide whether one completion is correct.

    ``verify`` is the hook that lets calibration reuse the real oracle harness:
    pass a callable that runs the candidate through ``oracles.run_all`` and
    returns whether the resulting ``TaskVerdict`` was PASS. It is called as
    ``verify(completion, task)`` when it accepts two positional arguments and
    ``verify(completion)`` otherwise, and exceptions from it propagate - a
    broken verifier must not look like a wrong answer.

    With no ``verify``, only stub completions can be graded. Anything else
    raises.
    """
    if verify is not None:
        result = verify(completion, task) if _verify_arity(verify) >= 2 else verify(completion)
        return Grade(
            correct=bool(result),
            graded_by=getattr(verify, "__name__", verify.__class__.__name__),
            detail="graded by caller-supplied verifier",
        )

    if STUB_CORRECT_MARKER in completion or STUB_INCORRECT_MARKER in completion:
        return Grade(
            correct=stub_verify(completion),
            graded_by="stub_verify",
            detail="stub completion graded by its own marker",
        )

    raise ValueError(
        "no verify callable was supplied and the completion carries no stub marker; "
        "calibration refuses to guess whether a completion is correct"
    )


# --------------------------------------------------------------------------- #
# StubModel - the default, offline, deterministic target
# --------------------------------------------------------------------------- #


def _unit_hash(*parts: Any) -> float:
    """A deterministic float in [0, 1) from the given parts."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


@dataclass(frozen=True)
class StubModel:
    """Deterministic offline target. No network, no API key, no clock.

    The per-task pass probability is a hash of the task id (falling back to the
    prompt when no id is available), so a bank calibrates end to end with the
    same numbers on every machine and every run, and different tasks land in
    different routing bands. Temperature is folded into the per-sample draw -
    it changes *which* samples pass, not how many are expected to.
    """

    seed: int = 1234
    name: str = "stub"
    p_min: float = 0.0
    p_max: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_min <= self.p_max <= 1.0:
            raise ValueError(f"require 0 <= p_min <= p_max <= 1: {self.p_min}, {self.p_max}")

    def pass_probability(self, task_id: str) -> float:
        """The task's latent pass rate. Pure function of ``(seed, task_id)``."""
        u = _unit_hash(self.seed, "task", task_id)
        return self.p_min + (self.p_max - self.p_min) * u

    def generate(
        self,
        prompt: str,
        n: int = 8,
        temperature: float = 0.8,
        *,
        task_id: str | None = None,
    ) -> list[str]:
        if int(n) < 0:
            raise ValueError(f"n must be >= 0: {n}")
        key = task_id or extract_task_id(prompt) or prompt
        p = self.pass_probability(key)
        out: list[str] = []
        for i in range(int(n)):
            u = _unit_hash(self.seed, "sample", key, i, f"{float(temperature):.6f}")
            correct = u < p
            marker = STUB_CORRECT_MARKER if correct else STUB_INCORRECT_MARKER
            out.append(
                f"# {marker}\n"
                f"# stub completion {i} for {key} (T={float(temperature):.2f}, "
                f"draw={u:.6f} vs p={p:.6f})\n"
                "def solution():  # offline stub, not a real answer\n"
                "    raise NotImplementedError\n"
            )
        return out


# --------------------------------------------------------------------------- #
# real adapters - constructed lazily, never on import
# --------------------------------------------------------------------------- #


def _require_key(env_var: str, provider: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise ModelUnavailable(
            f"{provider} was requested but {env_var} is not set in the environment; "
            "export the key or calibrate with --model stub (offline)",
            env_var=env_var,
            provider=provider,
        )
    return key


@dataclass
class AnthropicModel:
    """Anthropic Messages API target. Requires ``ANTHROPIC_API_KEY``."""

    model: str = DEFAULT_ANTHROPIC_MODEL
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 4096
    timeout_s: float = 600.0
    system: str = _DEFAULT_SYSTEM
    _client: Any = field(default=None, repr=False, compare=False)

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def accepts_temperature(self) -> bool:
        """False for models that reject sampling parameters with a 400."""
        return not any(self.model.startswith(prefix) for prefix in _NO_SAMPLING_PARAMS)

    def client(self) -> Any:
        if self._client is not None:
            return self._client
        key = _require_key(self.api_key_env, "AnthropicModel")
        try:
            import anthropic
        except ImportError as exc:
            raise ModelUnavailable(
                "AnthropicModel needs the 'anthropic' package; pip install anthropic",
                provider="anthropic",
            ) from exc
        self._client = anthropic.Anthropic(api_key=key, timeout=self.timeout_s)
        return self._client

    def generate(self, prompt: str, n: int = 8, temperature: float = 0.8) -> list[str]:
        client = self.client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.accepts_temperature():
            kwargs["temperature"] = float(temperature)
        else:
            logger.info(
                "%s does not accept sampling parameters; temperature=%.3f ignored",
                self.model,
                float(temperature),
            )

        out: list[str] = []
        for _ in range(int(n)):
            message = client.messages.create(**kwargs)
            out.append(self._text_of(message))
        return out

    @staticmethod
    def _text_of(message: Any) -> str:
        """Response text, with a refusal surfaced rather than silently empty."""
        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            return f"{REFUSAL_MARKER} category={category}"
        parts: list[str] = []
        for block in getattr(message, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "")))
        return "\n".join(parts)


@dataclass
class OpenAIModel:
    """OpenAI chat-completions target. Requires ``OPENAI_API_KEY``."""

    model: str = DEFAULT_OPENAI_MODEL
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 4096
    timeout_s: float = 600.0
    system: str = _DEFAULT_SYSTEM
    base_url: str | None = None
    _client: Any = field(default=None, repr=False, compare=False)

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def client(self) -> Any:
        if self._client is not None:
            return self._client
        key = _require_key(self.api_key_env, type(self).__name__)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelUnavailable(
                f"{type(self).__name__} needs the 'openai' package; pip install openai",
                provider="openai",
            ) from exc
        kwargs: dict[str, Any] = {"api_key": key, "timeout": self.timeout_s}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, prompt: str, n: int = 8, temperature: float = 0.8) -> list[str]:
        client = self.client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
            n=int(n),
            temperature=float(temperature),
            max_tokens=self.max_tokens,
        )
        return [str(getattr(choice.message, "content", "") or "") for choice in response.choices]


@dataclass
class LocalModel(OpenAIModel):
    """An OpenAI-compatible server you host (vLLM, TGI, llama.cpp, ...).

    Still a network call, so it is still opt-in: the base URL must be supplied
    explicitly or through ``CRUCIBLE_LOCAL_BASE_URL``.
    """

    model: str = "local"
    api_key_env: str = "CRUCIBLE_LOCAL_API_KEY"
    base_url_env: str = "CRUCIBLE_LOCAL_BASE_URL"

    @property
    def name(self) -> str:
        return f"local:{self.model}"

    def client(self) -> Any:
        if self._client is not None:
            return self._client
        url = self.base_url or os.environ.get(self.base_url_env, "").strip()
        if not url:
            raise ModelUnavailable(
                f"LocalModel needs a base URL; set {self.base_url_env} or pass base_url=",
                provider="local",
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelUnavailable(
                "LocalModel speaks the OpenAI wire format; pip install openai",
                provider="local",
            ) from exc
        # Local servers commonly ignore the key but the client requires one.
        key = os.environ.get(self.api_key_env, "").strip() or "not-used"
        self._client = OpenAI(api_key=key, base_url=url, timeout=self.timeout_s)
        return self._client


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #


def resolve_model(spec: str | None = None, *, seed: int = 1234) -> TargetModel:
    """Turn a ``--model`` string into a target. Unknown specs raise.

    Recognised forms:

    * ``""`` / ``"stub"`` / ``"stub:<anything>"`` -> ``StubModel`` (offline)
    * ``"anthropic:<id>"`` or an id starting ``claude`` -> ``AnthropicModel``
    * ``"openai:<id>"`` or an id starting ``gpt-``/``o1``/``o3`` -> ``OpenAIModel``
    * ``"local:<id>"`` -> ``LocalModel``

    Nothing here opens a connection; the client is built on first ``generate``.
    """
    raw = (spec or "stub").strip()
    low = raw.lower()

    if low in ("", "stub") or low.startswith("stub:"):
        return StubModel(seed=seed)
    if low.startswith("anthropic:"):
        return AnthropicModel(model=raw.split(":", 1)[1] or DEFAULT_ANTHROPIC_MODEL)
    if low.startswith("claude"):
        return AnthropicModel(model=raw)
    if low.startswith("openai:"):
        return OpenAIModel(model=raw.split(":", 1)[1] or DEFAULT_OPENAI_MODEL)
    if low.startswith(("gpt-", "o1", "o3", "o4")):
        return OpenAIModel(model=raw)
    if low.startswith("local:"):
        return LocalModel(model=raw.split(":", 1)[1] or "local")

    raise ValueError(
        f"unknown model spec {raw!r}; expected 'stub', 'anthropic:<id>', "
        "'openai:<id>', 'local:<id>', or a bare claude-*/gpt-* id"
    )
