#!/bin/bash
# Post-hoc t=0 answer-recoverability sensitivity audit over the locked
# 10-model step-0 panel. READ-ONLY w.r.t. the locked result: reads the frozen
# slice + frozen per-model specs/canaries/readouts, writes only into a fresh
# audit dir (never into the frozen run dir).
#
# Usage:
#   scripts/run_t0_recoverability_audit.sh <frozen_run_dir> <out_dir> [data.jsonl] [log]
#
# One subprocess per model (MLX buffer-cache thrash safety), same locked
# model order as run_step0_belief_panel.sh.
set -euo pipefail

FROZEN_RUN="$1"
OUT_DIR="$2"
DATA="${3:-experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl}"
LOG="${4:-/dev/stdout}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [ ! -d "$FROZEN_RUN" ]; then
  echo "[t0-audit] frozen run dir not found: $FROZEN_RUN" >&2
  exit 1
fi
if [ ! -f "$DATA" ]; then
  echo "[t0-audit] data slice not found: $DATA" >&2
  exit 1
fi
if [ "$(cd "$FROZEN_RUN" && pwd)" = "$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)" ]; then
  echo "[t0-audit] out_dir must differ from the frozen run dir" >&2
  exit 1
fi

MODELS=(
  "mlx-community/Qwen3-1.7B-4bit"
  "mlx-community/Llama-3.2-3B-Instruct-4bit"
  "mlx-community/gemma-3-4b-it-4bit"
  "mlx-community/Phi-3.5-mini-instruct-4bit"
  "mlx-community/Phi-4-mini-instruct-4bit"
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  "mlx-community/Qwen2.5-7B-Instruct-4bit"
  "mlx-community/Qwen3-8B-4bit"
  "mlx-community/Llama-3.1-8B-Instruct-4bit"
  "mlx-community/Mistral-Nemo-Instruct-2407-4bit"
)
N=${#MODELS[@]}
FAILURES=0

{
  echo "[t0-audit] frozen_run=$FROZEN_RUN"
  echo "[t0-audit] data=$DATA"
  echo "[t0-audit] out_dir=$OUT_DIR"
  echo "[t0-audit] models=$N"
  echo "[t0-audit] start: $(date)"
} | tee -a "$LOG"

for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"
  NAME="${M##*/}"
  IDX=$((i+1))
  PER_LOG="$OUT_DIR/${NAME}_t0_audit.log"
  AUDIT_JSON="$OUT_DIR/${NAME}_t0_audit.json"
  rm -f "$AUDIT_JSON"

  { echo ""; echo "[t0-audit] ($IDX/$N) $M started at $(date)"; } | tee -a "$LOG"

  if PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u scripts/audit_t0_recoverability.py \
      --model "$M" \
      --data "$DATA" \
      --frozen-run "$FROZEN_RUN" \
      --out-dir "$OUT_DIR" \
      > "$PER_LOG" 2>&1; then
    if [ ! -s "$AUDIT_JSON" ]; then
      FAILURES=$((FAILURES + 1))
      echo "[t0-audit] ($IDX/$N) $NAME status=failed (no artifact) — see $PER_LOG" | tee -a "$LOG"
    else
      echo "[t0-audit] ($IDX/$N) $NAME status=ok  json=$AUDIT_JSON" | tee -a "$LOG"
      tail -2 "$PER_LOG" | sed 's/^/[t0-audit]   /' | tee -a "$LOG"
    fi
  else
    RC=$?
    FAILURES=$((FAILURES + 1))
    echo "[t0-audit] ($IDX/$N) $NAME status=failed (exit $RC) — see $PER_LOG" | tee -a "$LOG"
  fi
done

# Panel roll-up across whatever model audits succeeded.
"$PYTHON_BIN" - "$OUT_DIR" "${MODELS[@]}" <<'PY' | tee -a "$LOG"
import csv, json, sys
from pathlib import Path

out_dir = Path(sys.argv[1]).expanduser().resolve()
models = sys.argv[2:]
fields = [
    "model", "sensitivity_verdict", "n_total",
    "literal_above_floor_coverage", "semantic_above_floor_coverage",
    "recovered_only_by_semantic_coverage",
    "frac_answerlike_exceeds_literal",
    "frac_answerlike_materially_exceeds_literal",
    "frac_top1_nonliteral", "frac_top1_answerlike_nonliteral",
    "mean_literal_decidedness", "mean_semantic_decidedness",
    "gate_max_abs_prob_drift_vs_locked_csv",
    "gate_max_abs_prob_drift_vs_frozen_canary",
]
rows = []
for m in models:
    name = m.split("/")[-1]
    jp = out_dir / f"{name}_t0_audit.json"
    if not jp.exists():
        rows.append({"model": m, "sensitivity_verdict": "MISSING"})
        continue
    s = json.loads(jp.read_text())
    g = s.get("integrity_gate", {})
    rows.append({
        "model": m,
        "sensitivity_verdict": s["sensitivity_verdict"],
        "n_total": s["n_total"],
        "literal_above_floor_coverage": round(s["literal_above_floor_coverage"], 4),
        "semantic_above_floor_coverage": round(s["semantic_above_floor_coverage"], 4),
        "recovered_only_by_semantic_coverage": round(s["recovered_only_by_semantic_coverage"], 4),
        "frac_answerlike_exceeds_literal": round(s["frac_answerlike_exceeds_literal"], 4),
        "frac_answerlike_materially_exceeds_literal": round(s["frac_answerlike_materially_exceeds_literal"], 4),
        "frac_top1_nonliteral": round(s["frac_top1_nonliteral"], 4),
        "frac_top1_answerlike_nonliteral": round(s["frac_top1_answerlike_nonliteral"], 4),
        "mean_literal_decidedness": round(s["mean_literal_decidedness"], 6),
        "mean_semantic_decidedness": round(s["mean_semantic_decidedness"], 6),
        "gate_max_abs_prob_drift_vs_locked_csv": g.get("max_abs_prob_drift_vs_locked_csv"),
        "gate_max_abs_prob_drift_vs_frozen_canary": g.get("max_abs_prob_drift_vs_frozen_canary"),
    })
csv_path = out_dir / "t0_audit_panel_summary.csv"
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
(out_dir / "t0_audit_panel_summary.json").write_text(
    json.dumps({"rows": rows}, indent=2) + "\n"
)
print(f"[t0-audit] wrote {csv_path}")
print(f"[t0-audit] wrote {out_dir / 't0_audit_panel_summary.json'}")
PY

echo "" | tee -a "$LOG"
if [ "$FAILURES" -gt 0 ]; then
  echo "[t0-audit] INCOMPLETE: $FAILURES/$N models failed at $(date)" | tee -a "$LOG"
  exit 1
fi
echo "[t0-audit] all $N models finished at $(date)" | tee -a "$LOG"
