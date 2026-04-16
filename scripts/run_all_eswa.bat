@echo off
REM One-command ESWA replication chain (from project root).
REM Uncomment Ollama-heavy lines only when models are pulled and GPU/server are ready.

cd /d "%~dp0.."
set RSE_RESULTS_DIR=data_new\results

echo [1/8] Aggregate calibration_sweep from calib_*.json
python experiments\aggregate_calibration_sweep.py
if errorlevel 1 exit /b 1

REM Full calibration (hours; requires Ollama):
REM python experiments\09_calibration_sweep.py --repeats 5 --threshold 0.05 --run-group calib_v2

REM Full benchmark 3x3, baseline 20 / compare 10 / quality 5 + detector/adaptive (07/08) for paper figures (very long):
REM python experiments\06_cross_model_suite.py --families location emotion color --baseline-repeats 20 --compare-repeats 10 --quality-repeats 5 --run-group eswa_benchmark_v1 --prompt-difficulty d2
REM Faster variant without 07/08: add --skip-methods to the line above, then uncomment method layer below.

REM Method layer repeat=10 (only if 06 used --skip-methods; also calls 07/08):
REM python experiments\10_run_method_layer.py --repeat 10 --run-group eswa_methods_v1

REM Ablation density+channel (long):
REM python experiments\06_cross_model_suite.py --families location --baseline-repeats 10 --compare-repeats 0 --quality-repeats 0 --skip-methods --ablation-repeats 10 --run-group eswa_ablation_v1

echo [2/8] Optional: STD signal ablation (uncomment)
REM python experiments\11_std_signal_ablation.py

echo [3/8] Optional: ASC threshold sweep (uncomment)
REM python experiments\12_asc_threshold_sweep.py

echo [4/8] Held-out baseline (requires Ollama; uncomment for smoke: --repeats 1)
REM python experiments\13_held_out_baseline.py --repeats 1

echo [5/8] Build paper artifacts (figures + JSON tables)
python experiments\04_build_paper_artifacts.py --merge-all-manifests --figure-models qwen3.5:4b gemma4:e4b openbmb/minicpm-v4.5:8b ministral-3:8b
if errorlevel 1 exit /b 1

echo [6/8] Unit tests
python -m pytest
if errorlevel 1 exit /b 1

echo [7/8] Optional: compile ESWA manuscript (requires elsarticle.cls)
REM pushd Docs\Paper
REM latexmk -pdf eswa_manuscript.tex
REM popd

echo [8/8] Done.
exit /b 0
