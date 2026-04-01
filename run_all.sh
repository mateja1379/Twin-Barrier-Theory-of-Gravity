#!/usr/bin/env bash
# run_all.sh — Full 7-stage validation of the RS-II braneworld solver.
#
# Usage:  bash run_all.sh
#
# Requires: JAX with CUDA, GPU with >=8 GB VRAM, Python >=3.10.

set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "  Einstein-DeTurck RS-II Braneworld Validation Suite"
echo "============================================================"
echo ""

# ── Environment check ────────────────────────────────────────────
echo "[0/4] Checking JAX environment..."
python env_check.py
echo ""

# ── Stage 1: Background metric ──────────────────────────────────
echo "[1/4] Stage 1: Background metric verification..."
python background_test.py
echo ""

# ── Stages 3–7: KK validation pipeline ──────────────────────────
echo "[2/4] Stages 3–6: KK spectrum & stability validation..."
python run_validation.py
echo ""

# ── Stage 7 (fast): PPN check ───────────────────────────────────
echo "[3/4] Stage 7: PPN relativistic check (fast version)..."
python run_stage7_fast.py
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "[4/4] Collecting results..."
echo ""
echo "============================================================"
echo "  RESULTS SUMMARY"
echo "============================================================"

for f in stage1_report.txt stage3_report.txt stage4_report.txt \
         stage5_report.txt stage6_report.txt stage7_report.txt; do
    if [ -f "$f" ]; then
        head -1 "$f"
    fi
done

echo ""
echo "Detailed results in: stage{1..7}_results.json"
echo "============================================================"
