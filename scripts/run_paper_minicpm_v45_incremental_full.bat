@echo off
REM Incremental full paper data for openbmb/minicpm-v4.5:8b only (06 + held-out).
REM Same protocol as ministral incremental: d2, 20/10/5, incl. ablations and 07/08.
setlocal
cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%t
set RG=minicpm_v45_incr_full_eswa_%TS%
set MODEL=openbmb/minicpm-v4.5:8b
echo RUN_GROUP=%RG%
echo MODEL=%MODEL%
python experiments\06_cross_model_suite.py --models "%MODEL%" --families location emotion color --baseline-repeats 20 --compare-repeats 10 --quality-repeats 5 --prompt-difficulty d2 --run-group %RG%_suite
if errorlevel 1 exit /b 1
python experiments\13_held_out_baseline.py --models "%MODEL%" --repeats 5 --run-group %RG%_heldout
if errorlevel 1 exit /b 1
echo Done. Merge: python experiments\04_build_paper_artifacts.py --merge-all-manifests --figure-models qwen3.5:4b gemma4:e4b openbmb/minicpm-v4.5:8b ministral-3:8b
endlocal
