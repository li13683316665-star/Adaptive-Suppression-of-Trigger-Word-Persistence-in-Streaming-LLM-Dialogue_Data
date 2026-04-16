@echo off
REM Full paper data: cross-model suite (incl. ablations) + held-out baselines for all paper.figure_models.
REM Matches REPRODUCIBILITY main benchmark: d2, 20/10/5. Requires Ollama running on 127.0.0.1:11434.
setlocal
cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%t
set RG=paper_quartet_full_eswa_%TS%
echo RUN_GROUP_BASE=%RG%
echo Suite manifest: crossmodel_manifest_%RG%_suite.json
echo Held-out manifest: held_out_manifest_%RG%_heldout.json
python experiments\06_cross_model_suite.py --baseline-repeats 20 --compare-repeats 10 --quality-repeats 5 --prompt-difficulty d2 --run-group %RG%_suite
if errorlevel 1 exit /b 1
python experiments\13_held_out_baseline.py --repeats 5 --run-group %RG%_heldout
if errorlevel 1 exit /b 1
echo Done. Next: python experiments\04_build_paper_artifacts.py --merge-all-manifests
endlocal
