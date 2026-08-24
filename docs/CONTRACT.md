# CRUCIBLE — Internal interface contract (v1.0)

**This file is normative.** Every module is written against it by a different author.
Do not change a shared type without changing this file first.

Repo root: `crucible/`. Import root: `crucible` (the inner package).
Python 3.11, torch 2.6.0+cu124, pydantic v2, numpy, scipy, typer, rich, jinja2, pyyaml.

---

## 0. The one invariant

> A task ships only if its **reference solution was executed** and passed every
> **applicable** oracle.

Therefore `SKIP` is never `PASS`. Any oracle that cannot run on this machine
returns `verdict="SKIP"` with a non-empty `reason` and the task's overall verdict
becomes `SKIP` (not `PASS`). The headline metric counts `PASS` only.

Corollary: **no oracle may fabricate evidence.** If `ncu` is absent, the counter
dict is absent — not zero. If clocks could not be locked, `clock_locked=False`
is recorded and the perf verdict is downgraded, not silently reported.

---

## 1. Layout

```
crucible/
  pyproject.toml
  README.md
  crucible/
    __init__.py        errors.py      config.py
    capabilities.py    schema.py      taxonomy.py
    runner/     __init__.py  sandbox.py  determinism.py
    seeds/      __init__.py  registry.py  attention.py  reduction.py  matmul.py
                quant.py  collectives.py  dataloader.py  checkpointing.py  autotune.py
    mutate/     __init__.py  astutil.py  classes.py  witness.py  engine.py
    oracles/    __init__.py  base.py  tolerance.py  o1_numerics.py  o2_perf.py
                o3_anticheat.py  o4_compile.py  o5_nrank.py  o5_worker.py
    calibrate/  __init__.py  passk.py  models.py  irt.py  router.py
    rubric/     __init__.py  spec.py  autoprobe.py  irr.py  sequences.py  gate.py
    redteam/    __init__.py  attacks.py  suite.py
    report/     __init__.py  metrics.py  coverage.py  dashboard.py
    cli.py
  tests/   docs/   bank/
```

---

## 2. `crucible/capabilities.py`

```python
@dataclass(frozen=True)
class Capabilities:
    cuda: bool
    cuda_device_count: int
    device_name: str | None
    cuda_version: str | None
    triton: bool                # importable AND a trivial kernel compiles
    triton_error: str | None
    ncu: str | None             # path to ncu, else None
    can_lock_clocks: bool       # nvidia-smi -lgc succeeded in a dry probe
    gloo: bool
    nccl: bool
    cxx_compiler: str | None    # cl / gcc on PATH, else None
    inductor_cpu: bool          # torch.compile on CPU produced a compiled fn
    inductor_cuda: bool
    torch_version: str
    platform: str

    def missing(self, required: Iterable[str]) -> list[str]: ...
        # returns names of falsy/None attrs among `required`

def detect(refresh: bool = False) -> Capabilities: ...   # cached in-process +
                                                          # ~/.crucible/caps.json
CAP_NAMES: frozenset[str]        # every valid capability string
```

Probes must be cheap, never raise, and time out (compiler probes ≤ 60 s).
`triton` probe = import **and** JIT-compile a 3-line kernel; on this machine it
fails with a DLL error, which is recorded verbatim in `triton_error`.

---

## 3. `crucible/schema.py` — pydantic v2 models

```python
Tier   = Literal["T1","T2","T3","T4","T5","T6"]
Domain = Literal["triton","pallas","cuda","jax","pytorch","distributed",
                 "data_pipeline","checkpointing"]
VerdictStr = Literal["PASS","FAIL","SKIP","ERROR"]

class ShapeSpec(BaseModel):
    name: str                  # stable id, e.g. "seq1023_bf16"
    kwargs: dict[str, Any]     # passed verbatim to SeedSpec.make_inputs
    def key(self) -> str       # deterministic hash of kwargs

class SeedRef(BaseModel):
    seed_id: str
    module: str
    entry: str
    content_sha256: str

class MutationSpec(BaseModel):
    cls: str                   # mutation class id, see §6
    site: str                  # "attention.py:88" or "func:blocked_attention#3"
    params: dict[str, Any] = {}
    description: str = ""

class Witness(BaseModel):
    shape: ShapeSpec
    max_abs_err: float
    max_rel_err: float
    tolerance: float
    baseline_checksum: str
    mutant_checksum: str
    kind: Literal["numeric","exception","shape","hang","loss_curve"]
    detail: str = ""

class RubricCriterion(BaseModel):
    id: str
    weight: float
    anchors: dict[int, str]          # score -> anchor text (0,3,5 minimum)
    auto_probe: str | None = None    # e.g. "oracle.O1.max_rel_err"; see §10
    probe_thresholds: list[tuple[float, int]] = []   # [(bound, score)] ascending
    machine_probed: bool = False

class RubricSpec(BaseModel):
    criteria: list[RubricCriterion]
    version: str = "1.0"

class CalibrationRecord(BaseModel):
    model: str
    k: int
    n_samples: int
    n_correct: int
    pass_at_1: float
    pass_at_k: float
    route: Literal["reject","gold","frontier","escalate"]
    rationale: str
    trace_stats: dict[str, Any] = {}

class Task(BaseModel):
    task_id: str
    schema_version: str = "1.0"
    created_utc: str
    seed_source: SeedRef
    domain: Domain
    failure_tier: Tier
    mutation: MutationSpec
    baseline_code: str
    mutant_code: str
    ground_truth_diff: str        # unified diff mutant -> baseline (i.e. the FIX)
    witness: Witness | None
    detect_shapes: list[ShapeSpec]   # withheld from the prompt; grading uses these
    decoy_shapes: list[ShapeSpec]    # given in the prompt; bug does NOT reproduce
    prompt: str
    oracles: list[str]               # ["O1","O3","O4"]
    rubric: RubricSpec
    calibration: CalibrationRecord | None = None
    provenance: dict[str, Any] = {}

    def to_yaml(self) -> str / def save(path) / @classmethod load(path)

class OracleResult(BaseModel):
    oracle: str
    verdict: VerdictStr
    reason: str = ""              # REQUIRED non-empty unless verdict == "PASS"
    evidence: dict[str, Any] = {}
    duration_s: float = 0.0
    capabilities_used: list[str] = []

class EnvironmentRecord(BaseModel):
    caps: dict[str, Any]; python: str; torch: str; host: str; utc: str

class TaskVerdict(BaseModel):
    task_id: str
    verdict: VerdictStr
    executed: bool                # did the candidate actually run on hardware
    oracle_results: list[OracleResult]
    environment: EnvironmentRecord
    duration_s: float
    candidate_sha256: str

    @staticmethod
    def combine(results) -> VerdictStr:
        # ERROR if any ERROR; FAIL if any FAIL; SKIP if any SKIP; else PASS
```

Validator: `OracleResult` rejects empty `reason` when verdict != "PASS".

---

## 4. `crucible/taxonomy.py`

```python
TIERS: dict[Tier, TierInfo]      # failure_mode, signal_available, model_skill, data_value
DOMAINS: tuple[Domain, ...]      # the 8 above
CELLS: list[tuple[Tier, Domain]] # 48
def cell_id(tier, domain) -> str          # "T5/triton"
def coverage(tasks) -> CoverageGrid       # counts + fill rate + gaps
@dataclass CoverageGrid: counts: dict[str,int]; target_per_cell: int = 5
    filled(), gaps(), fill_rate(), silent_share()  # (T5+T6)/total
```

---

## 5. `crucible/seeds/` — known-good baselines

Every seed module defines module-level `SEED: SeedSpec` and registers via
`@register` in `registry.py`.

```python
@dataclass
class SeedSpec:
    id: str                       # "attention.blocked_fwd"
    domain: Domain
    tiers: tuple[Tier, ...]       # tiers this seed can express
    description: str
    entry: str                    # function name the candidate must define
    source: str                   # the known-good source of `entry` (text)
    make_inputs: Callable[[ShapeSpec, torch.device, torch.Generator], dict]
    reference: Callable[..., Any] # independent ground truth (torch native op)
    shape_sweep: list[ShapeSpec]  # adversarial, not happy-path
    accum_depth: Callable[[ShapeSpec], int]
    bytes_moved: Callable[[ShapeSpec], int]
    flops: Callable[[ShapeSpec], int]
    denylist: tuple[str, ...]     # symbols banned in a candidate solution
    compare: Callable[[Any, Any], CompareResult] | None = None  # default: tensor cmp
    supports_cpu: bool = True
```

Rules:
- `source` must be **pure torch**, executable on CPU and CUDA, and must
  *structurally* contain the mutation sites it advertises (explicit block loops,
  an explicit accumulator dtype, an explicit empty-block guard, an explicit
  store). It simulates kernel structure; it is not a wrapper around a fused op.
- `reference` must NOT be the same code path as `source` (no self-consistency).
- `make_inputs` is a pure function of (shape, device, generator) — reproducible.
- Seeds carrying real Triton/CUDA text for AST mutation put it in
  `SeedSpec.source` only if `supports_cpu=False`; those seeds' execution oracles
  will legitimately SKIP on this machine.

Coverage required across the 8 domains: attention (triton), reduction (triton),
matmul (cuda), quant (cuda), collectives (distributed), dataloader
(data_pipeline), checkpointing (checkpointing), autotune (pytorch/jax).

---

## 6. `crucible/mutate/` — the mutation engine

Twelve mutation classes, ids exactly:

| id | change | tier |
|---|---|---|
| `boundary_mask` | `offs < N` → `offs <= N` | T5 |
| `accum_dtype` | fp32 accumulator → fp16/bf16 | T5 |
| `shmem_sizing` | buffer sized by logical count, indexed by thread/lane id | T2/T5 |
| `missing_barrier` | drop a sync/clone boundary → aliasing race | T5 |
| `cross_block_reduction` | per-block partial treated as total | T5 |
| `empty_input_guard` | remove zero-length-block skip | T2 |
| `uninit_output` | drop the final store on one path | T5 |
| `layout_conflict` | force a more-replicated layout at a conflict | T5 |
| `grad_norm_scope` | global norm → per-shard norm under FSDP | T6 |
| `collective_ordering` | reorder all-reduce across a conditional branch | T3 |
| `resume_fidelity` | optimizer state saved fp32, restored bf16 | T6 |
| `autotune_staleness` | reuse cached config after a shape-class change | T4 |

```python
class MutationClass(Protocol):
    id: str; tier: Tier; description: str
    def sites(self, tree: ast.AST, src: str) -> list[Site]
    def apply(self, tree: ast.AST, site: Site) -> ast.AST   # returns a NEW tree
```

`astutil.py`: parse/unparse with `ast.parse` + `ast.unparse`, site addressing by
`(lineno, col_offset, node_type, ordinal)`, unified-diff helper.

`witness.py`:
```python
def search(seed, mutant_src, sweep, caps, cfg) -> WitnessSearchResult
# For each shape in the sweep (ordered small->large), run baseline and mutant in
# the sandbox on identical seeded inputs and compare against derived tolerance.
# Returns first breaking shape as the witness, plus every passing shape as decoys.
# kind: "numeric" | "exception" | "shape" | "hang" | "loss_curve"
@dataclass WitnessSearchResult:
    witness: Witness | None; decoys: list[ShapeSpec]; detects: list[ShapeSpec]
    n_evaluated: int; discarded_reason: str | None
```
**A mutation with no witness is discarded** (`discarded_reason` set) — it is
semantically neutral or untested-by-construction. This is the structural
guarantee that every shipped task has a real, reachable answer.

`engine.py`:
```python
def generate(seed_ids, classes, caps, cfg, out_dir, limit) -> GenerationReport
# seed -> parse -> for each class: sites -> apply -> witness search ->
# admit or discard -> build Task (prompt, rubric, oracles, ground_truth_diff)
```
`task_id` format: `mut-{seed_short}-{cls}-{4-hex}` (hex = sha256 of mutant src).
The prompt must include `decoy_shapes` and must NOT include `detect_shapes`,
the mutation class, or the diff.

---

## 7. `crucible/oracles/base.py`

```python
@dataclass
class OracleContext:
    task: Task
    candidate_src: str          # code under test (reference solution by default)
    seed: SeedSpec
    caps: Capabilities
    workdir: Path
    cfg: Config
    rng_seed: int
    device: str                 # "cpu" | "cuda"

class Oracle(Protocol):
    id: str; name: str
    required_caps: tuple[str, ...]
    def applies_to(self, task: Task) -> bool: ...
    def run(self, ctx: OracleContext) -> OracleResult: ...

ORACLES: dict[str, Oracle]      # "O1".."O5"
def run_all(ctx, ids=None) -> list[OracleResult]
```
`run_all` wraps each oracle: missing caps → `SKIP` with
`"requires <caps>; missing <caps> (<detail>)"`; uncaught exception → `ERROR`
with the traceback in `evidence["traceback"]`; per-oracle timeout from config.

### `tolerance.py` — the error-budget model (no hardcoded 1e-3)
```python
EPS = {"fp32": 2**-24, "tf32": 2**-11, "bf16": 2**-8, "fp16": 2**-11, "fp8": 2**-4}
def derive(dtype, accum_depth, mode="stochastic", safety=4.0) -> Tolerance
# rel = safety * growth(accum_depth) * EPS[dtype];  growth = sqrt(K) stochastic,
# K deterministic (worst case).  abs = rel * scale, scale from data magnitude.
@dataclass Tolerance: rel: float; abs: float; dtype: str; accum_depth: int;
                      mode: str; formula: str    # human-readable derivation
```
Every numeric verdict must print the formula it used.

### O1 `o1_numerics.py` — required_caps: ()
Differential vs `seed.reference` across the **adversarial** sweep
(`seq_len ∈ {1,127,128,129,1023,4096}`, empty blocks, single-element batch,
head_dim at/above block size, non-contiguous strides, dtypes fp32/bf16/fp16).
Evidence: `max_abs_err`, `max_rel_err`, `first_break_shape`, `per_shape` table,
`tolerance` (with formula). FAIL if any shape exceeds tolerance.

### O2 `o2_perf.py` — required_caps: ("cuda",)
Lock clocks (`nvidia-smi -lgc`, record success), purge autotune cache, warmup
discarded, ≥30 timed reps, **median + bootstrap CI (2000 resamples)**. Roofline
denominator uses *measured achievable* bandwidth (STREAM-triad probe) and
achievable FLOP/s, not spec sheet. Attach `ncu` counters when available
(achieved occupancy, DRAM throughput, spill loads/stores). A speedup without a
CI is not a claim: if the CI straddles 1.0, report `inconclusive`, not a win.

### O3 `o3_anticheat.py` — required_caps: ()
Five independent checks, each individually reported:
`static_denylist` (AST: banned imports/attribute chains/calls — a *string*
denylist is not acceptable), `randomized_inputs` (fresh seed per grading
invocation; rerun and require agreement), `held_out_shapes` (grade on
`detect_shapes` the solution never saw), `output_liveness` (consume + checksum
the result; verify that perturbing the input changes the checksum, so DCE cannot
elide the work), `timing_sanity` (flag any time below
`bytes_moved / achievable_bandwidth`). Any check firing → FAIL.

### O4 `o4_compile.py` — required_caps: ()  (per-check caps internally)
`torch.compile(fullgraph=True)` graph-break detection via
`torch._dynamo.explain`; Triton register-spill check (SKIP with
`caps.triton_error` here); Pallas TPU/GPU portability (SKIP — no jax/TPU);
emitted IR capture (inductor output code) and diff vs baseline. Each sub-check
reports its own SKIP reason; the oracle SKIPs only if *no* sub-check could run.

### O5 `o5_nrank.py` + `o5_worker.py` — required_caps: ("gloo",)
Tiny model (2 layers, ~1M params), 50 steps, fixed seed, deterministic data.
Run 1 rank, then N ranks (default 2 and 4) with **gloo** via
`torch.multiprocessing.spawn` + `FileStore`/TCP on localhost (NCCL is
unavailable on Windows — do not use it). Compare loss curves.
**Tolerance is empirically calibrated, not assumed:** first run the 1-rank
config twice under different reduction orders to measure the float
non-associativity noise floor, then require
`max|loss_N - loss_1| <= k * noise_floor` (k from config, default 3), and
report the floor. Catches per-shard grad norm, dropped grad sync, wrong loss
reduction across DP, mis-sharded optimizer state on resume.
Must be Windows-safe: worker entry importable at module scope, `if __name__ ==
"__main__"` guard, no fork assumptions, hard timeout, guaranteed process reap.

---

## 8. `crucible/runner/`
`sandbox.py`: execute candidate source in a **subprocess** (`sys.executable -c`),
JSON in / JSON out over a temp file, hard timeout, memory-lenient, captured
stderr, non-zero exit → structured `SandboxError`. Never `exec()` untrusted code
in-process. Tensors cross the boundary as `.npy` files or checksums, not pickle
of arbitrary objects.
`determinism.py`: `seed_everything(n)`, `deterministic_ctx()`
(`torch.use_deterministic_algorithms`, cuBLAS workspace env var), autotune-cache
purge, `lock_clocks()/unlock_clocks()` context manager reporting success.

---

## 9. `crucible/calibrate/`
`passk.py`: the **unbiased** Chen et al. estimator
`pass@k = 1 - C(n-c, k)/C(n, k)`, computed in log space; never `(c/n)**k`.
`router.py`: `route(pass1, passk) -> ("reject"|"gold"|"frontier"|"escalate", rationale)`
per the proposal's table (>0.9 reject; [0.1,0.7] gold; <0.1 with pass@8>0 frontier;
pass@8==0 escalate).
`models.py`: `TargetModel` protocol (`generate(prompt, n, temperature) -> list[str]`)
with `StubModel` (deterministic, offline, **the default**), `AnthropicModel`,
`OpenAIModel`, `LocalModel`. **No network call may happen unless the user passes
an explicit `--model` that is not `stub`.**
`irt.py`: 2PL fit (difficulty b, discrimination a) by MAP over the bank with
scipy; item-response curves; report discrimination ranking. Contamination
detector: `pass@1 ≈ 1.0` **and** a long, confident, non-exploratory trace.

---

## 10. `crucible/rubric/`
`spec.py`: build/validate `RubricSpec`; every criterion that *can* be
machine-probed **must** be (`machine_probed=True`), removing it from human
variance.
`autoprobe.py`: resolve `auto_probe` strings like `oracle.O1.max_rel_err`
against a `list[OracleResult]`, map through `probe_thresholds` to a score.
Unresolvable probe → explicit error, never a default score.
`irr.py`: **Krippendorff's α** from a ratings matrix (raters × items, NaN =
missing) with `nominal|ordinal|interval|ratio` difference functions, via the
coincidence matrix. Bootstrap CI. Verified against a published worked example
in the tests.
`sequences.py`: anytime-valid confidence sequences — `HoeffdingCS`,
`EmpiricalBernsteinCS`, `BettingCS` (port the API from
`certified-sparse-attention/csa/verify.py`, credited in the docstring) applied
to the per-criterion **disagreement stream** so α can be monitored continuously
as ratings accumulate without a peeking problem.
`gate.py`: α ≥ 0.67 → keep; below → flag criterion for rewrite with the
disagreeing item pairs attached as evidence.

---

## 11. `crucible/redteam/`
`attacks.py`: named attack solutions, each `Attack(id, description,
expected_catcher, applies_to(seed), source(task))` covering at minimum:
`cublas_smuggling` (call the fused op), `memoize_inputs`, `shape_hardcode`
(`BLOCK_M=128`), `dce_elision` (never consume the output),
`tolerance_gaming` (right shape, wrong values, fast), `wrong_dtype_speedwin`,
`seed_pinning` (assume the fixed seed), `reference_import` (import the baseline).
`suite.py`: run every applicable attack against the harness; a template ships
only if catch rate == 100%. Report which oracle caught each attack; an attack
caught by *no* oracle is a grader defect and fails the suite loudly.

---

## 12. `crucible/report/`
`metrics.py`: the seven headline metrics from the proposal §10, computed from a
directory of tasks + verdicts. Headline: **% of shipped tasks whose reference
solution was executed and passed all oracles**.
`coverage.py`: 48-cell grid fill rates, gaps, T5+T6 share.
`dashboard.py`: self-contained HTML (jinja2, inline CSS, no CDN), light+dark,
coverage heatmap, metric tiles, per-task table, IRR panel.

---

## 13. `crucible/cli.py` (typer)
```
crucible doctor                     # capability report — what can run here
crucible seeds list|show ID
crucible mutate --seed ID --classes a,b --limit N --out bank/
crucible verify TASK.yaml [--candidate FILE] [--oracles O1,O3] -o verdict.json
crucible verify-bank bank/ -o verdicts/
crucible calibrate bank/ --model stub --k 8
crucible irr ratings.csv [--metric ordinal]
crucible redteam [--seed ID]
crucible coverage bank/
crucible report bank/ verdicts/ -o report.html
```
Exit codes: 0 ok, 1 FAIL, 2 SKIP-blocked, 3 ERROR. Rich output; `--json` for
machine consumption everywhere.

---

## 14. Style rules for every author
- Type hints on public functions; `from __future__ import annotations`.
- No bare `except:`; catch specific exceptions and record them.
- Comments explain *why*, not *what*. Match the density of the surrounding code.
- No emoji in code or output. No `print` outside `cli.py`/`report/` — use the
  module logger.
- Windows-safe: `pathlib`, no `/tmp`, no `fork`, no `signal.SIGALRM`,
  `encoding="utf-8"` on every file open.
- Every module ships `tests/test_<module>.py` with real assertions. A test that
  merely imports the module is not a test.
- Do not edit files outside your assigned list.
