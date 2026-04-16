@echo off
REM Resume MiniCPM v4.5 8b incremental suite after a partial run: skips repeats that already
REM have crossmodel_*_<stem>_*.json under data_new/results/. Edit RUN_GROUP_BASE if your batch used another TS.
setlocal
cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
REM Same prefix as scripts\run_paper_minicpm_v45_incremental_full.bat for the failed job on 2026-04-15:
set RUN_GROUP_BASE=minicpm_v45_incr_full_eswa_20260415_051850
set MODEL=openbmb/minicpm-v4.5:8b
set SUITE_RG=%RUN_GROUP_BASE%_suite
set HELDOUT_RG=%RUN_GROUP_BASE%_heldout
echo Resuming SUITE run-group=%SUITE_RG%
echo Then HELDOUT run-group=%HELDOUT_RG%
python experiments\06_cross_model_suite.py --resume --models "%MODEL%" --families location emotion color --baseline-repeats 20 --compare-repeats 10 --quality-repeats 5 --prompt-difficulty d2 --run-group %SUITE_RG%
if errorlevel 1 exit /b 1
python experiments\13_held_out_baseline.py --resume --models "%MODEL%" --repeats 5 --run-group %HELDOUT_RG%
if errorlevel 1 exit /b 1
echo Done. Merge: python experiments\04_build_paper_artifacts.py --merge-all-manifests --figure-models qwen3.5:4b gemma4:e4b openbmb/minicpm-v4.5:8b ministral-3:8b
endlocal
