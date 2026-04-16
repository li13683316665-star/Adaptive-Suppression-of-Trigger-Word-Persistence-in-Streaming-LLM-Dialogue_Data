# Full ESWA data_new pipeline: calibration -> aggregate -> cross-model suite -> paper artifacts.
# Each run uses a fresh timestamped run-group so results do not mix with older partial runs.
# Uses cmd.exe for logging so Python stderr (logging module) does not stop the pipeline in Windows PowerShell.
$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path } else { (Get-Location).Path }
Set-Location $Root
$env:RSE_RESULTS_DIR = "data_new/results"
$null = New-Item -ItemType Directory -Force -Path "data_new/logs" | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $Root "data_new/logs/eswa_full_$Stamp.txt"
$RunGroupCalib = "calib_full_$Stamp"
$RunGroupBench = "eswa_benchmark_$Stamp"
$ManifestRel = "data_new/results/crossmodel_manifest_$RunGroupBench.json"

function Run-Cmd([string]$ArgsLine) {
    $full = "python $ArgsLine"
    "`n>> $full`n" | Out-File -FilePath $Log -Append -Encoding utf8
    cmd.exe /c "$full >> `"$Log`" 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "Failed (exit $LASTEXITCODE): $full" }
}

try {
    @"
ESWA full pipeline started $(Get-Date -Format o)
Run groups: calibration=$RunGroupCalib  benchmark=$RunGroupBench
Manifest (after 06): $ManifestRel
"@ | Out-File -FilePath $Log -Encoding utf8
    Write-Host "Log: $Log"
    Write-Host "Calibration run-group: $RunGroupCalib"
    Write-Host "Benchmark run-group:  $RunGroupBench"

    "`n========== 09 calibration_sweep ==========`n" | Out-File -FilePath $Log -Append -Encoding utf8
    Run-Cmd "experiments/09_calibration_sweep.py --repeats 5 --threshold 0.05 --run-group $RunGroupCalib"

    "`n========== aggregate_calibration_sweep ==========`n" | Out-File -FilePath $Log -Append -Encoding utf8
    Run-Cmd "experiments/aggregate_calibration_sweep.py"

    "`n========== 06 cross_model_suite ==========`n" | Out-File -FilePath $Log -Append -Encoding utf8
    Run-Cmd "experiments/06_cross_model_suite.py --families location emotion color --baseline-repeats 20 --compare-repeats 10 --quality-repeats 5 --run-group $RunGroupBench --prompt-difficulty d2"

    "`n========== 04 build_paper_artifacts ==========`n" | Out-File -FilePath $Log -Append -Encoding utf8
    Run-Cmd "experiments/04_build_paper_artifacts.py --manifest $ManifestRel"

    "`n========== pytest ==========`n" | Out-File -FilePath $Log -Append -Encoding utf8
    Run-Cmd "-m pytest tests -q"

    "`nDONE $(Get-Date -Format o)" | Out-File -FilePath $Log -Append -Encoding utf8
    Write-Host "Finished OK. Log: $Log"
}
catch {
    "`nFAILED: $_`n" | Out-File -FilePath $Log -Append -Encoding utf8
    Write-Error $_
    exit 1
}
