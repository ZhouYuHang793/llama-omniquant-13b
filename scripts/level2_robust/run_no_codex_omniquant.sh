#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-report-only}"       # evidence-only | report-only | full
PROFILE="${2:-balanced}"       # balanced | aggressive
ROOT="${ROOT:-/root/autodl-tmp/omniquant_final}"
AGENT="$ROOT/HIGH_SCORE_AGENT"
ENV_DIR="$ROOT/env"
WORK="$ROOT/agent_workspace"
EVIDENCE="$WORK/evidence"
FIGURES="$WORK/figures"
REPORT="$ROOT/report"
SUBMISSION="$ROOT/submission"
GPU_APPROVED="${GPU_APPROVED:-0}"

mkdir -p "$WORK" "$EVIDENCE" "$FIGURES" "$REPORT" "$SUBMISSION"

echo "==== OmniQuant No-Codex Pipeline ===="
echo "MODE=$MODE PROFILE=$PROFILE GPU_APPROVED=$GPU_APPROVED"
echo "ROOT=$ROOT"
echo "START=$(date --iso-8601=seconds)"

[[ -x "$ENV_DIR/bin/python" ]] || { echo "[FATAL] env not found: $ENV_DIR"; exit 2; }
[[ -d "$AGENT" ]] || { echo "[FATAL] HIGH_SCORE_AGENT not installed: $AGENT"; exit 2; }

source "$ENV_DIR/bin/activate"

echo "[1] collect evidence"
python "$AGENT/scripts/collect_evidence.py" --root "$ROOT" --out "$EVIDENCE"

if [[ "$MODE" == "evidence-only" ]]; then
    echo "[DONE] Evidence: $EVIDENCE/evidence_index.md"
    exit 0
fi

if [[ "$MODE" == "full" ]]; then
    [[ "$GPU_APPROVED" == "1" ]] || { echo "[FATAL] full mode needs GPU_APPROVED=1"; exit 3; }
    echo "[2] run adaptive experiments"
    MAX_GPU_HOURS_PER_CASE="${MAX_GPU_HOURS_PER_CASE:-1.5}"     MAX_NEW_RUNS="${MAX_NEW_RUNS:-5}"     W2_BAD_PPL_THRESHOLD="${W2_BAD_PPL_THRESHOLD:-100}"     GPU_APPROVED=1     bash "$AGENT/scripts/adaptive_experiments.sh" "$PROFILE" | tee "$WORK/no_codex_adaptive_experiments.log"
    echo "[3] re-collect evidence"
    python "$AGENT/scripts/collect_evidence.py" --root "$ROOT" --out "$EVIDENCE"
else
    echo "[2] skip GPU experiments"
fi

echo "[4] generate figures"
python "$AGENT/scripts/generate_figures.py" --evidence "$EVIDENCE/evidence.json" --out "$FIGURES" || true

echo "[5] write report draft"
python "$AGENT/scripts/write_no_codex_report.py" --root "$ROOT" --evidence "$EVIDENCE/evidence.json" --figures "$FIGURES" --out "$REPORT/final_report.md"

echo "[6] validate and package"
python "$AGENT/scripts/validate_report.py" --report "$REPORT/final_report.md" --evidence "$EVIDENCE/evidence.json" --out "$REPORT/validation_report.md" || true

cat > "$REPORT/reviewer_report.md" <<'EOF'
# No-Codex 审阅说明
本报告由 No-Codex 流水线根据真实 evidence 自动生成。提交前请人工检查：
1. 替换团队成员真实姓名和真实分工；
2. 按老师要求转成 Word/PDF；
3. 核对所有数字与 evidence_index.md；
4. 保留并解释失败实验和高 PPL，不要删除。
EOF

cat > "$REPORT/change_log.md" <<'EOF'
# 修改记录
- 使用 No-Codex 流水线生成证据索引、图表、报告初稿和提交包。
- 所有实验数字来自 evidence.json 或日志。
EOF

python "$AGENT/scripts/build_submission.py" --root "$ROOT" --out "$SUBMISSION"

echo "==== FINISHED ===="
echo "REPORT=$REPORT/final_report.md"
echo "VALIDATION=$REPORT/validation_report.md"
echo "EVIDENCE=$EVIDENCE/evidence_index.md"
echo "SUBMISSION=$SUBMISSION/OmniQuant_Final_Submission.zip"
