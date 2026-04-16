# Live tail of benchmark stderr (or any log). Like: tail -f
# Usage:
#   .\scripts\watch_benchmark_progress.ps1
#   .\scripts\watch_benchmark_progress.ps1 -LogPath "data_new\results\ministral_incr_full_20260414_055054_stderr.log"
#   .\scripts\watch_benchmark_progress.ps1 -Summary   # one-line stats then exit
param(
    [string]$LogPath = "",
    [switch]$Summary
)
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

if (-not $LogPath) {
    $cand = Get-ChildItem -Path (Join-Path $root "data_new\results") -Filter "*_stderr.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $cand) {
        Write-Host "No *_stderr.log under data_new\results. Pass -LogPath explicitly."
        exit 1
    }
    $LogPath = $cand.FullName
} elseif (-not [System.IO.Path]::IsPathRooted($LogPath)) {
    $LogPath = Join-Path $root $LogPath
}

if (-not (Test-Path $LogPath)) {
    Write-Host "Log not found: $LogPath"
    exit 1
}

Write-Host "=== RSE benchmark log ===" -ForegroundColor Cyan
Write-Host $LogPath
Write-Host "Lines matching 'Running command' = step starts; 'Saved JSON' = sub-run finished."
Write-Host "Full 06 (1 model, d2, 20/10/5): ~125 subprocess calls (60 baseline + 30 compare + 5 quality + 10 det + 10 adapt + 10 ablation)."
Write-Host "Press Ctrl+C to stop watching (does not stop the benchmark)."
Write-Host ""

$lines = Get-Content -Path $LogPath -ErrorAction SilentlyContinue
$run = ($lines | Select-String -Pattern "Running command:").Count
$done = ($lines | Select-String -Pattern "Saved JSON results to").Count
$last = $lines | Select-Object -Last 3

if ($Summary) {
    Write-Host "Completed JSON writes (sub-runs): $done"
    Write-Host "'Running command' lines seen: $run"
    exit 0
}

Write-Host ("[snapshot] JSON completed: {0} | 'Running command' lines: {1}" -f $done, $run) -ForegroundColor Yellow
Write-Host "--- last 3 lines ---"
$last | ForEach-Object { Write-Host $_ }
Write-Host "--- following (live, last 40 lines refresh) ---`n"

Get-Content -Path $LogPath -Wait -Tail 40 -Encoding UTF8
