#!/usr/bin/env bash
#
# v3.1 overnight launcher — three-phase sealed-gate + extended companions.
#
# Design:
#   Phase 1 (v3_1_primaries)  — MUST succeed; sealed E18 + E17b gate authority.
#                               If this fails, the whole script exits non-zero
#                               and Phase 2 / 3 are skipped (no point capturing
#                               companion data without a sealed verdict).
#   Phase 2 (v3_1_qwen3)      — best-effort; failure is logged and the script
#                               continues to Phase 3.
#   Phase 3 (v3_1_gemmas)     — best-effort; failure is logged and the script
#                               still reports Phase-1-pass if Phase 1 succeeded.
#
# Sealed parameters (from pri-v3-plan.md §Amendments 2026-04-23 + 2026-04-24):
#   rank pinned at r = 1 (plan §E18 sealed block)
#   seed = 20260423
#   n_samples_per_cell = 50
#   E17b capture ON by default (Config.v3_capture_raw = True)
#   behavioral gate threshold = 80% control accuracy at n = 20
#   no-silent-override clause: this script does NOT pass --skip-gate.
#
# Usage:
#   nohup bash scripts/run_v3_1_overnight.sh >/dev/null 2>&1 &
#   # or foreground (progress to terminal):
#   bash scripts/run_v3_1_overnight.sh
#
# Outputs:
#   experiments/v3-main-run/<YYYY-MM-DD>/run-NN/  — per-phase parquets
#   logs/v3_1_<timestamp>.log                     — merged stdout+stderr
#   logs/v3_1_<timestamp>.done                    — sentinel file with summary
#
# Expected runtimes on Mac mini M4 (approximate):
#   Phase 1 (3 primaries + raw-W_u SVD precompute) : ~60-80 min
#   Phase 2 (Qwen3 alone)                           : ~15-20 min
#   Phase 3 (Gemma 3-1B + 3-4B)                     : ~40-60 min
#   Total if all three complete sequentially        : ~2-2.5 hours

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Sealed parameters — do NOT edit here to bypass the plan. Any change requires
# a new Amendments entry in wiki/pri-v3-plan.md filed before launch.
readonly SEED=20260423
readonly N_PER_CELL=50
readonly MAX_GEN_TOKENS=14

TS="$(date -u +%Y%m%d-%H%M%SZ)"
mkdir -p logs
LOG="logs/v3_1_${TS}.log"
DONE="logs/v3_1_${TS}.done"

# Venv
if [[ ! -f .venv/bin/activate ]]; then
  echo "FATAL: .venv/bin/activate not found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" | tee -a "$LOG"
  exit 2
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Prevent macOS sleep while the pipeline runs. caffeinate backgrounds itself
# and we reap it at exit (traps handle abnormal exits too). Without this a
# Mac mini M4 can suspend during low-CPU SVD waits.
caffeinate -disu &
CAF_PID=$!
cleanup() {
  if [[ -n "${CAF_PID:-}" ]] && kill -0 "$CAF_PID" 2>/dev/null; then
    kill "$CAF_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

banner() {
  local msg="$1"
  {
    echo ""
    echo "============================================================================"
    echo "  $msg   [$(date -u +%Y-%m-%dT%H:%M:%SZ)]"
    echo "============================================================================"
    echo ""
  } | tee -a "$LOG"
}

phase_run() {
  # phase_run <label> <scope>
  local label="$1"
  local scope="$2"
  local start end elapsed rc
  start=$(date +%s)
  banner "BEGIN $label — scope=$scope seed=$SEED n=$N_PER_CELL"
  .venv/bin/python scripts/run_v3_main.py \
    --scope "$scope" \
    --n-per-cell "$N_PER_CELL" \
    --seed "$SEED" \
    --max-gen-tokens "$MAX_GEN_TOKENS" \
    >>"$LOG" 2>&1
  rc=$?
  end=$(date +%s)
  elapsed=$(( end - start ))
  if [[ $rc -eq 0 ]]; then
    banner "END   $label — PASS (${elapsed}s)"
  else
    banner "END   $label — FAIL rc=$rc (${elapsed}s)"
  fi
  return $rc
}

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
banner "v3.1 OVERNIGHT LAUNCH — $TS"
{
  echo "repo:        $REPO"
  echo "seed:        $SEED"
  echo "n/cell:      $N_PER_CELL"
  echo "max_gen:     $MAX_GEN_TOKENS"
  echo "log:         $LOG"
  echo "sentinel:    $DONE"
  echo "caffeinate:  PID $CAF_PID"
  echo "python:      $(.venv/bin/python --version 2>&1)"
  echo "git head:    $(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo')"
  echo "git branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
} | tee -a "$LOG"

PHASE1_RC=0
PHASE2_RC=0
PHASE3_RC=0
PHASE2_STATUS="SKIPPED"
PHASE3_STATUS="SKIPPED"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — sealed gate authority. MUST succeed.
# ──────────────────────────────────────────────────────────────────────────────
phase_run "PHASE 1 (sealed gate — primaries)" "v3_1_primaries"
PHASE1_RC=$?
if [[ $PHASE1_RC -ne 0 ]]; then
  banner "PHASE 1 FAILED — skipping Phases 2 and 3 (sealed gate not achieved)"
  echo "result=phase1_failed phase1_rc=$PHASE1_RC" > "$DONE"
  echo "" >> "$DONE"
  echo "Phase 1 failed with exit code $PHASE1_RC. Phase 2 and 3 skipped." >> "$DONE"
  echo "Log: $LOG" >> "$DONE"
  exit $PHASE1_RC
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — cross-generation companion (Qwen3). Best-effort.
# ──────────────────────────────────────────────────────────────────────────────
phase_run "PHASE 2 (cross-gen — Qwen3)" "v3_1_qwen3"
PHASE2_RC=$?
if [[ $PHASE2_RC -eq 0 ]]; then
  PHASE2_STATUS="PASS"
else
  PHASE2_STATUS="FAIL rc=$PHASE2_RC"
  banner "PHASE 2 FAILED — continuing to Phase 3 (best-effort independence)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — within-family scale companion (Gemma). Best-effort.
# ──────────────────────────────────────────────────────────────────────────────
phase_run "PHASE 3 (scale — Gemma 1B + 4B)" "v3_1_gemmas"
PHASE3_RC=$?
if [[ $PHASE3_RC -eq 0 ]]; then
  PHASE3_STATUS="PASS"
else
  PHASE3_STATUS="FAIL rc=$PHASE3_RC"
  banner "PHASE 3 FAILED — primary gate (Phase 1) already passed, check log"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Final summary + sentinel
# ──────────────────────────────────────────────────────────────────────────────
OVERALL="primary_pass"
if [[ $PHASE2_RC -ne 0 || $PHASE3_RC -ne 0 ]]; then
  OVERALL="primary_pass_partial_extended"
fi

{
  echo "result=$OVERALL"
  echo "phase1=PASS rc=$PHASE1_RC"
  echo "phase2=$PHASE2_STATUS"
  echo "phase3=$PHASE3_STATUS"
  echo ""
  echo "Sealed E18 + E17b gate authority was Phase 1. Phase 1 passed."
  echo "Phase 2 (Qwen3 cross-gen companion): $PHASE2_STATUS"
  echo "Phase 3 (Gemma scale companion):     $PHASE3_STATUS"
  echo ""
  echo "Log: $LOG"
  echo "Artifacts: experiments/v3-main-run/$(date -u +%Y-%m-%d)/run-NN/"
  echo ""
  echo "Next steps:"
  echo "  1. Read Phase 1 parquets for E18 + E17b AUROCs at final/step=1/rank=1."
  echo "  2. Bootstrap sample-level CIs (1000 resamples) per sealed spec."
  echo "  3. Report Qwen3 + Gemma as descriptive companion data (cannot satisfy"
  echo "     or invalidate the sealed gates per primary-scoped authority)."
} | tee "$DONE"

banner "v3.1 RUN COMPLETE — $OVERALL"

# Exit code reflects the sealed-gate phase only — Phase 2/3 failures are
# reported in the sentinel file but do not fail the overall run, since the
# sealed gate authority is carried entirely by Phase 1.
exit $PHASE1_RC
