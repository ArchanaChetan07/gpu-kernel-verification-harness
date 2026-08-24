# CRUCIBLE

An execution-grounded task foundry for ML-systems training data.

Implementation of `CRUCIBLE_Proposal.pdf`. The product is not tasks — it is the
instrumentation that turns task quality into a number:

> **What fraction of shipped reference solutions was ever executed?**

Today that number is unmeasured almost everywhere. It should be 100%.

---

## The one invariant

A task ships only if its reference solution **was executed** and passed every
**applicable** oracle.

So `SKIP` is never `PASS`. An oracle that cannot run on the current machine
returns a skip with its real reason, that task counts *against* the headline
metric, and the run summary prints exactly which claims went unverified. No
oracle fabricates evidence: if the profiler is absent, the counters are absent —
not zero.

This matters because the failure mode the project targets is invisible. An
unexecuted wrong solution and an executed right solution look identical in the
deliverable; the defect surfaces months later as model behavior.

---

## Architecture

```
upstream repos  ->  L1  MUTATION ENGINE          -> task + known ground-truth diff
(Triton, vLLM,      |    semantic bug injection      + a witness proving it is real
 JAX, PyTorch)      v
                    L2  FIVE-ORACLE VERIFIER     -> PASS/FAIL/SKIP + evidence JSON
                    |    O1 numerics   O2 perf
                    |    O3 anti-cheat O4 compile
                    |    O5 N-rank equivalence
                    v
                    L3  DIFFICULTY CALIBRATOR    -> ship / reject / escalate
                    |    pass@k vs target model
                    v
                    L4  RUBRIC ENGINE + IRR      -> Krippendorff alpha per criterion
                    |
                    v
                    L5  REWARD-HACK RED TEAM     -> grader regression suite
```

**L0 — the organizing idea.** Every task family is stratified by *how
observable the failure is*, T1 (compile error, the message is the answer)
through T6 (silent convergence divergence, the loss curve looks fine). Coverage
is a 6×8 grid of 48 cells and every cell's fill rate is a number on the
dashboard. Frontier models are already good at T1–T2; the value and the real
cost are at T5–T6.

**L1 — mutation, not mining.** Mining public repos is linear and contaminated.
Taking a *known-correct* kernel and applying a semantically meaningful mutation
yields a task whose ground truth is the inverse diff and which has never existed
in any repository. A mutation is admitted only if the engine can produce a
concrete **witness** — an input on which the mutant demonstrably differs from
the baseline. No witness means the mutation is semantically neutral or
untested-by-construction, and it is discarded. That is the structural guarantee
that every shipped task has a real, reachable answer.

---

## Quick start

```bash
pip install -e .
crucible doctor
```

`doctor` reports what this machine can honestly verify, oracle by oracle. On a
box without a working Triton or a pre-Ampere GPU it will tell you so, and which
checks will consequently skip.

```bash
crucible mutate --seed all --classes all --limit 50 --out bank/
crucible verify-bank bank/ -o verdicts/
crucible report bank/ verdicts/ -o report.html
```

Or the whole pipeline with preflight, resume, and a cost line:

```bash
./scripts/pipeline.sh --smoke     # ~5 min shakeout
./scripts/pipeline.sh             # full run
```

## Running on a rented GPU

See [docs/RUNNING_ON_RENTED_GPU.md](docs/RUNNING_ON_RENTED_GPU.md). Short
version — Modal, per-second billing:

```bash
modal run deploy/modal_app.py::doctor
modal run deploy/modal_app.py::smoke
modal run deploy/modal_app.py::full --limit 200
```

**Minimum useful hardware is sm_80 (Ampere).** The highest-value tier is silent
numerics, and a reduced-precision accumulation bug only produces a witness where
bf16 accumulation is real. On Turing and older those mutations are discarded as
witness-free and the tasks are never manufactured.

---

## Layout

| Path | What |
|---|---|
| `crucible/capabilities.py` | probes what this machine can verify; every skip reason traces here |
| `crucible/schema.py` | Task, Witness, OracleResult, TaskVerdict |
| `crucible/taxonomy.py` | T1–T6 × 8 domains, the 48-cell coverage grid |
| `crucible/seeds/` | known-good baselines carrying real mutation sites |
| `crucible/mutate/` | 12 mutation classes, witness search, task generation |
| `crucible/oracles/` | O1 numerics, O2 perf, O3 anti-cheat, O4 compile, O5 N-rank |
| `crucible/calibrate/` | unbiased pass@k, routing, 2PL IRT over the bank |
| `crucible/rubric/` | Krippendorff α, anytime-valid confidence sequences, the 0.67 gate |
| `crucible/redteam/` | attack solutions the grader must catch |
| `crucible/report/` | headline metrics, coverage grid, self-contained HTML |
| `docs/CONTRACT.md` | the normative interface contract |

The confidence-sequence machinery in `crucible/rubric/sequences.py` is ported
from the author's `certified-sparse-attention` project.

---

## Design notes worth knowing

**Tolerance is derived, never hardcoded.** `1e-3` is not a tolerance, it is a
guess. O1 derives one from an error budget — accumulation depth × dtype epsilon
× a safety factor — and prints the formula it used alongside every verdict.

**Seeds simulate kernel structure in pure torch.** Explicit tile loops, explicit
accumulator dtypes, explicit boundary masks, explicit stores. This keeps the
mutation sites structurally real and executable on CPU and GPU alike, instead of
being blocked on a toolchain. Each seed's ground truth is an *independent* torch
native op — never the seed compared against itself.

**O5 calibrates its own tolerance.** Before comparing anything, it measures the
float non-associativity noise floor by running the 1-rank config twice under
different summation orders, then gates on a multiple of the measured floor. The
proposal flags this as the least de-risked component; measuring the floor rather
than assuming a constant is the de-risking.

**Anti-cheat is AST-based, not string matching.** Alias-resolved imports,
dotted attribute chains, dynamic `getattr`, and `__import__` are all caught. The
uninitialized-output check runs the same call into two differently pre-filled
buffers and requires identical results — the concrete defect from the
proposal's audit where a kernel never wrote its output and returned
uninitialized memory.
