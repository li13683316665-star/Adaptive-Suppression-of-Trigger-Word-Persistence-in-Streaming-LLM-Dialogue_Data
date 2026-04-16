@echo off
REM Incremental full paper data for ministral-3:8b only: 06 (incl. ablations) + held-out.
REM Stops nothing; run stop_quartet_run.bat first if a quartet job is still active.
setlocal
cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%t
set RG=ministral_incr_full_eswa_%TS%
echo RUN_GROUP=%RG%
echo Step 1: cross-model suite (baseline, compare, quality, detector, adaptive, ablation)
python experiments\06_cross_model_suite.py --models ministral-3:8b --families location emotion color --baseline-repeats 20 --compare-repeats 10 --quality-repeats 5 --prompt-difficulty d2 --run-group %RG%_suite
if errorlevel 1 exit /b 1
echo Step 2: held-out families animal + weather
python experiments\13_held_out_baseline.py --models ministral-3:8b --repeats 5 --run-group %RG%_heldout
if errorlevel 1 exit /b 1
echo Done. Merge: python experiments\04_build_paper_artifacts.py --merge-all-manifests --figure-models qwen3.5:4b gemma4:e4b openbmb/minicpm-v4.5:8b ministral-3:8b
endlocal
