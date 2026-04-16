# Fast path: stop python children of this repo's benchmark runs (no WMI scan of all processes).
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        $p = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction Stop).CommandLine
    } catch { return }
    if (-not $p) { return }
    if ($p -notmatch [regex]::Escape($root)) { return }
    if ($p -notmatch 'experiments\\(06_cross_model_suite|01_baseline_bias|02_algorithm_compare|05_quality_retention|07_detector_eval|08_adaptive_eval|13_held_out)') { return }
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped PID $($_.Id)"
}
