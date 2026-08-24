# Running CRUCIBLE on a rented GPU

Everything here assumes you are paying by the hour. The operating principle is
that **an unverified claim must cost you nothing to discover** — so the run
aborts before the expensive stage when the pod cannot honestly execute it.

---

## 1. Which GPU, and why

The choice is driven by one thing: the highest-value task tier in this project
is **silent numerics (T5)**, and a bf16/fp16 accumulation bug only produces a
witness on hardware where reduced-precision accumulation is *real*. On a
pre-Ampere card those mutations are discarded as witness-free and the tasks are
never manufactured at all.

| GPU | sm | Use it for | Rough $/hr |
|---|---|---|---|
| **H100 80GB** | 90 | the headline run; perf claims you intend to defend | $2.00–3.50 |
| **A100 80GB** | 80 | same coverage, cheaper, slower | $1.20–2.00 |
| **L40S 48GB** | 89 | best value for O1/O3/O4; weak for O5 scaling | $0.80–1.30 |
| **A10 / RTX 4090** | 86/89 | development and smoke runs | $0.30–0.75 |
| T4 / T1000 / any Turing | 75 | **not sufficient** — no real bf16 accumulation | — |

Minimum bar is **sm_80 (Ampere)**. Above that, more memory buys longer sequence
lengths in the adversarial sweep, not different conclusions.

**Multi-GPU is optional.** O5's most valuable failure classes — per-shard grad
norm, dropped all-reduce, wrong DP loss reduction, mis-sharded optimizer state
— are semantic and reproduce on gloo/CPU with multiple processes on one device.
Rent 2+ GPUs only for the T3 ring-ordering and deadlock class.

---

## 2. Fastest path: Modal (recommended)

Per-second billing, so a misconfigured image costs cents.

```bash
pip install modal && modal setup
modal run deploy/modal_app.py::doctor
```

`doctor` prints what this card can actually verify. Read it before spending
anything. Then:

```bash
modal run deploy/modal_app.py::smoke
```

Six tasks end to end, roughly five minutes. **This is the step that tells you
what the full run will cost** — multiply the per-task wall clock by your bank
size before committing. Then:

```bash
modal run deploy/modal_app.py::full --limit 200
modal volume get crucible-runs /full ./runs
```

---

## 3. RunPod / Lambda / Vast / any Ubuntu pod

```bash
# on the pod, having copied the repo up (scp, git clone, or runpodctl)
cd crucible
./scripts/bootstrap_pod.sh
./scripts/pipeline.sh --smoke
./scripts/pipeline.sh
tar czf crucible-run.tar.gz runs/
```

`bootstrap_pod.sh` is idempotent and deliberately **does not replace a working
torch**. Rented images ship torch built against their driver; swapping it for a
PyPI default-CUDA wheel is the standard way to break a working pod, and it also
invalidates the tolerance calibration, which is per-torch-build.

## 4. Docker (most reproducible)

```bash
docker build -t crucible .
docker run --gpus all --cap-add=SYS_ADMIN -v "$PWD/runs:/opt/crucible/runs" \
    --entrypoint bash crucible scripts/pipeline.sh --smoke
```

`--cap-add=SYS_ADMIN` is required for Nsight Compute counters. Without it O2
records `ERR_NVGPUCTRPERM` as its reason and **omits** the counters rather than
reporting zeros.

---

## 5. What the run does, and where the money goes

| Stage | Layer | Needs GPU | Notes |
|---|---|---|---|
| preflight | — | no | aborts in ~10s if caps are missing |
| mutate | L1 | **yes** | witness search executes both variants; bf16 fidelity matters here |
| verify-bank | L2 | **yes** | the product: every reference solution executed |
| calibrate | L3 | no | offline stub by default, no network, no API key |
| redteam | L5 | partial | regression tests the grader itself |
| coverage + report | L0/L4 | no | 48-cell grid, headline metric, HTML |

Cost is dominated by `verify-bank`. Derive your estimate from the smoke run
rather than from this page — that is the whole reason smoke exists. Per the
proposal's own scope note, **estimate before committing, and select oracles per
task rather than running all five always**:

```bash
crucible verify-bank bank/ --oracles O1,O3 -o verdicts/     # numerics + anti-cheat only
```

---

## 6. Two expectations to set before you read the output

**Clock locking will probably fail.** `nvidia-smi -lgc` needs privileges you do
not have on most cloud pods. That is handled, not ignored: the run records
`clock_locked=false`, widens the reported confidence interval, and the perf
claim visibly weakens in the evidence. An unlocked-clock measurement is a
weaker claim and it is supposed to look weaker.

**`SKIP` is not `PASS`.** Any oracle that cannot run emits a skip with its real
reason, and those tasks count *against* the headline metric, never toward it.
The end-of-run summary prints a `NOT VERIFIED` block listing exactly which
claims this pod could not check. That block is the most useful thing the run
produces — it is the number the proposal says nobody currently has.

---

## 7. Reading the result

```
HEADLINE - reference solutions executed and passed every oracle:
  187/200 = 93.5%
```

That is the product. Then `runs/<id>/report.html` for the 48-cell coverage
heatmap, the per-oracle skip breakdown, difficulty routing, and IRR per rubric
criterion against the 0.67 gate.

Target numbers are in the proposal §10: headline 100%, gold difficulty band
>60%, Krippendorff α ≥ 0.7 on 100% of shipped criteria, red-team catch rate
100%, ≥5 tasks per taxonomy cell, T5+T6 share ≥35%.
