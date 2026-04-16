@echo off
REM Opens a new console window with live tail of the latest data_new\results\*_stderr.log
cd /d "%~dp0.."
start "RSE benchmark progress" powershell -NoExit -NoProfile -ExecutionPolicy Bypass -File "%~dp0watch_benchmark_progress.ps1"
