#!/usr/bin/env bash
# ESWA replication chain (Git Bash / Linux / macOS). Run from repo root: bash scripts/run_all_eswa.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export RSE_RESULTS_DIR="${RSE_RESULTS_DIR:-data_new/results}"

echo "[1/8] Aggregate calibration_sweep from calib_*.json"
python experiments/aggregate_calibration_sweep.py

# Uncomment when Ollama is ready (long runs):
# python experiments/09_calibration_sweep.py --repeats 5 --threshold 0.05 --run-group calib_v2
# Full benchmark including 07/08 (omit --skip-methods):
# python experiments/06_cross_model_suite.py --families location emotion color --baseline-repeats 20 --compare-repeats 10 --quality-repeats 5 --run-group eswa_benchmark_v1 --prompt-difficulty d2
# Faster without 07/08: add --skip-methods to the line above, then run method layer:
# python experiments/10_run_method_layer.py --repeat 10 --run-group eswa_methods_v1
# python experiments/06_cross_model_suite.py --families location --baseline-repeats 10 --compare-repeats 0 --quality-repeats 0 --skip-methods --ablation-repeats 10 --run-group eswa_ablation_v1
# python experiments/11_std_signal_ablation.py
# python experiments/12_asc_threshold_sweep.py
# python experiments/13_held_out_baseline.py --repeats 1

echo "[2/8] Build paper artifacts"
python experiments/04_build_paper_artifacts.py --merge-all-manifests --figure-models qwen3.5:4b gemma4:e4b openbmb/minicpm-v4.5:8b ministral-3:8b

echo "[3/8] Unit tests"
python -m pytest

echo "Done."
