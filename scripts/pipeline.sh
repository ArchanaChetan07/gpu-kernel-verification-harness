#!/usr/bin/env bash
# CRUCIBLE end-to-end foundry run, sized for a metered GPU.
#
# The pod is billed by the hour, so this script is built around three rules:
#   1. Fail before the expensive part. Preflight asserts the capabilities each
#      stage needs and aborts if the pod cannot honestly run them.
#   2. Never redo finished work. Every stage is skipped when its output already
#      exists, so a disconnect costs the current stage, not the whole run.
#   3. Report what was NOT verified. A run that skipped four oracles is not a
#      successful run, and the summary says so out loud.
#
# Usage:
#   ./scripts/pipeline.sh              full run
#   ./scripts/pipeline.sh --smoke      ~5 min shakeout, do this first
#   FORCE=1 ./scripts/pipeline.sh      ignore existing outputs and redo
#
# Knobs (environment):
#   OUT              run directory                  (default runs/<utc>)
#   SEEDS            comma list or "all"            (default all)
#   CLASSES          mutation classes or "all"      (default all)
#   LIMIT            max tasks to generate          (default 200)
#   MODEL            calibration target model       (default stub, no network)
#   K                pass@k samples                 (default 8)
#   RATE             USD/hour, for the cost line    (default 2.50)
#   ALLOW_DEGRADED   1 = run even if caps missing   (default 0)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"

SMOKE=0
[[ "${1:-}" == "--smoke" ]] && SMOKE=1

OUT="${OUT:-runs/$(date -u +%Y%m%dT%H%M%SZ)}"
SEEDS="${SEEDS:-all}"
CLASSES="${CLASSES:-all}"
LIMIT="${LIMIT:-200}"
MODEL="${MODEL:-stub}"
K="${K:-8}"
RATE="${RATE:-2.50}"
FORCE="${FORCE:-0}"
ALLOW_DEGRADED="${ALLOW_DEGRADED:-0}"

if [[ "$SMOKE" == "1" ]]; then
  LIMIT=6
  K=2
  OUT="${OUT}-smoke"
fi

BANK="$OUT/bank"
VERDICTS="$OUT/verdicts"
LOGS="$OUT/logs"
mkdir -p "$BANK" "$VERDICTS" "$LOGS"

START=$(date +%s)
step() { printf '\n=== [%s] %s\n' "$(date -u +%H:%M:%S)" "$1"; }
have() { [[ "$FORCE" != "1" && -e "$1" ]]; }

# --- Stage 0: preflight -----------------------------------------------------
# The single most expensive mistake on a rented box is discovering after the
# run that the oracle you paid for could not execute. Find out in 10 seconds.
step "Preflight: capability probe"
crucible doctor --json > "$OUT/capabilities.json"
crucible doctor || true

python - "$OUT/capabilities.json" "$ALLOW_DEGRADED" <<'PY'
import json, sys, pathlib

caps = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
allow_degraded = sys.argv[2] == "1"

# Flattened lookup so this survives either a flat or a nested doctor payload.
flat = {}
def walk(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            walk(v, f"{prefix}{k}.")
        else:
            flat[k] = v
            flat[f"{prefix}{k}"] = v
walk(caps)

# What each stage genuinely needs. Absence is not fatal for all of them: the
# run degrades to a smaller set of honest verdicts rather than faking the rest.
wanted = {
    "cuda":   ("O1 on real reduced precision, O2 performance", True),
    "triton": ("O4 lowering and register-spill checks",        False),
    "gloo":   ("O5 N-rank equivalence",                        False),
    "ncu":    ("O2 hardware counters",                         False),
}
missing_hard, missing_soft = [], []
for cap, (why, hard) in wanted.items():
    if not flat.get(cap):
        (missing_hard if hard else missing_soft).append(f"{cap} -> {why}")

if missing_soft:
    print("\nDEGRADED: these oracles will SKIP and their tasks will NOT count as verified:")
    for m in missing_soft:
        print("  -", m)

if missing_hard:
    print("\nBLOCKED: this pod cannot run the stages you are paying for:")
    for m in missing_hard:
        print("  -", m)
    if not allow_degraded:
        print("\nStop the pod, or re-run with ALLOW_DEGRADED=1 to proceed anyway.")
        sys.exit(2)

dev = flat.get("device_name") or "unknown"
print(f"\nPreflight OK on {dev}. bf16 fidelity matters here: a mutation whose")
print("witness only appears under real reduced-precision accumulation is")
print("discarded as witness-free on hardware without tensor cores.")
PY

# --- Stage 1: manufacture tasks --------------------------------------------
# Runs on the GPU on purpose. Witness search executes both baseline and mutant,
# and a bf16 accumulation bug only produces a witness where bf16 is real.
step "L1 mutation engine: manufacturing tasks"
if have "$BANK/.done"; then
  echo "skip (bank exists, FORCE=1 to redo)"
else
  crucible mutate --seed "$SEEDS" --classes "$CLASSES" --limit "$LIMIT" \
      --out "$BANK" 2>&1 | tee "$LOGS/mutate.log"
  touch "$BANK/.done"
fi
N_TASKS=$(find "$BANK" -name '*.yaml' | wc -l | tr -d ' ')
echo "bank holds $N_TASKS tasks"

# --- Stage 2: verify every reference solution -------------------------------
# This is the product. Everything else is instrumentation around it.
step "L2 five-oracle verifier: executing every reference solution"
if have "$VERDICTS/.done"; then
  echo "skip (verdicts exist, FORCE=1 to redo)"
else
  crucible verify-bank "$BANK" -o "$VERDICTS" 2>&1 | tee "$LOGS/verify.log"
  touch "$VERDICTS/.done"
fi

# --- Stage 3: difficulty calibration ----------------------------------------
# Default model is the offline stub, so this makes no network call unless you
# pass a real --model. Routing thresholds are only meaningful against the
# actual target model; against a proxy they must be re-fit.
step "L3 difficulty calibrator: pass@k against $MODEL"
if have "$OUT/calibration.json"; then
  echo "skip"
else
  crucible calibrate "$BANK" --model "$MODEL" --k "$K" --json \
      > "$OUT/calibration.json" 2>"$LOGS/calibrate.log" || \
      echo "calibration incomplete, see $LOGS/calibrate.log"
fi

# --- Stage 4: red team ------------------------------------------------------
# Regression testing for the grader itself. An attack caught by no oracle is a
# grader defect, and the suite is supposed to say so loudly.
step "L5 reward-hack red team: attacking the grader"
if have "$OUT/redteam.json"; then
  echo "skip"
else
  crucible redteam --json > "$OUT/redteam.json" 2>"$LOGS/redteam.log" || \
      echo "red team reported failures, see $OUT/redteam.json"
fi

# --- Stage 5: coverage + report ---------------------------------------------
step "L0 coverage grid and report"
crucible coverage "$BANK" --json > "$OUT/coverage.json" 2>"$LOGS/coverage.log" || true
crucible coverage "$BANK" || true
crucible report "$BANK" "$VERDICTS" -o "$OUT/report.html" 2>&1 | tee "$LOGS/report.log"

# --- Summary ----------------------------------------------------------------
ELAPSED=$(( $(date +%s) - START ))
step "Run complete"
python - "$OUT" "$ELAPSED" "$RATE" <<'PY'
import json, pathlib, sys

out, elapsed, rate = pathlib.Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
hrs = elapsed / 3600.0
print(f"wall clock : {elapsed//60}m {elapsed%60}s")
print(f"est. cost  : ${hrs*rate:,.2f} at ${rate:.2f}/hr")

verdicts = sorted((out / "verdicts").glob("*.json"))
counts, skip_reasons = {}, {}
for p in verdicts:
    try:
        v = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        continue
    counts[v.get("verdict", "?")] = counts.get(v.get("verdict", "?"), 0) + 1
    for r in v.get("oracle_results", []):
        if r.get("verdict") == "SKIP":
            key = f"{r.get('oracle')}: {r.get('reason', '')[:70]}"
            skip_reasons[key] = skip_reasons.get(key, 0) + 1

total = sum(counts.values())
if total:
    passed = counts.get("PASS", 0)
    print(f"\nHEADLINE - reference solutions executed and passed every oracle:")
    print(f"  {passed}/{total} = {100.0*passed/total:.1f}%")
    print(f"  breakdown: {counts}")
    if skip_reasons:
        # The most useful panel in the whole run: it is the list of claims this
        # pod could not verify, which is precisely what a silent harness hides.
        print("\nNOT VERIFIED (these count against the headline, not toward it):")
        for k, n in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4d}  {k}")
else:
    print("\nHEADLINE: unmeasured - no verdicts were produced.")

print(f"\nartifacts: {out}")
print(f"report   : {out/'report.html'}")
PY
