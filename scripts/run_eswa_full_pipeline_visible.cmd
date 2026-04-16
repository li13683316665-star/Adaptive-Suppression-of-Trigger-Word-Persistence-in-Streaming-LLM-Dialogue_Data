@echo off
REM Opens a normal (not minimized) PowerShell window running the full ESWA pipeline.
REM Double-click this file or run from cmd from the project root.
cd /d "%~dp0.."
start "ESWA full pipeline" powershell.exe -NoExit -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_eswa_full_pipeline.ps1"
