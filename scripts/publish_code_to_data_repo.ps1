# Publish reproducibility code into the SAME GitHub repo as research data (under code/).
# Prerequisite: eswa_data_deposit is a git clone of your data repo with remote origin set.
#
# Usage (PowerShell, from anywhere):
#   cd "C:\Users\13511\桌面\Project RSE"
#   .\scripts\publish_code_to_data_repo.ps1
#
# Optional:
#   .\scripts\publish_code_to_data_repo.ps1 -DataRepoPath "D:\path\to\eswa_data_deposit"
#   .\scripts\publish_code_to_data_repo.ps1 -SkipPush    # commit only

param(
    [string]$DataRepoPath = "",
    [switch]$SkipPush,
    [switch]$SkipReadmeAppend
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not $DataRepoPath) {
    $DataRepoPath = Join-Path $ProjectRoot "eswa_data_deposit"
}

Write-Host "==> Building eswa_code_release..." -ForegroundColor Cyan
$build = Join-Path $ProjectRoot "scripts\build_eswa_code_release.py"
python $build
if ($LASTEXITCODE -ne 0) {
    Write-Error "build_eswa_code_release.py failed."
    exit $LASTEXITCODE
}

$codeSrc = Join-Path $ProjectRoot "eswa_code_release"
$codeDest = Join-Path $DataRepoPath "code"

if (-not (Test-Path (Join-Path $DataRepoPath ".git"))) {
    Write-Error @"
No git repository at: $DataRepoPath

Clone your GitHub data repository first, e.g.:
  cd `"$ProjectRoot`"
  git clone git@github.com:li13683316665-star/Adaptive-Suppression-of-Trigger-Word-Persistence-in-Streaming-LLM-Dialogue_Data.git eswa_data_deposit
Then run this script again (or pass -DataRepoPath to your clone).
"@
    exit 1
}

Write-Host "==> Syncing code -> $codeDest" -ForegroundColor Cyan
if (Test-Path $codeDest) {
    Remove-Item $codeDest -Recurse -Force
}
New-Item -ItemType Directory -Path $codeDest -Force | Out-Null
Copy-Item -Path (Join-Path $codeSrc "*") -Destination $codeDest -Recurse -Force

$marker = "## Reproducibility code (same repository)"
$rootReadme = Join-Path $DataRepoPath "README.md"
if (-not $SkipReadmeAppend -and (Test-Path $rootReadme)) {
    $raw = Get-Content $rootReadme -Raw -Encoding UTF8
    if ($raw -notlike "*Reproducibility code (same repository)*") {
        $add = @"

## Reproducibility code (same repository)

The ``code/`` directory contains experiment scripts, benchmark prompts under ``code/data/prompts/``, library code under ``code/src/``, ``code/configs/``, ``code/tests/``, and ``code/requirements.txt``. See ``code/README.md`` for environment setup.

To regenerate paper figures after cloning this repository: copy the JSON files from ``results/`` (at repo root) into ``code/data_new/results/``, then from the ``code/`` folder set ``PYTHONPATH`` to that folder and run ``python experiments/04_build_paper_artifacts.py`` (PNG output under ``code/Docs/Paper/figures/``).
"@
        Add-Content -Path $rootReadme -Value $add -Encoding utf8
        Write-Host "==> Appended reproducibility section to data repo README.md" -ForegroundColor DarkGray
    }
}

Set-Location $DataRepoPath
git add code
if (-not $SkipReadmeAppend -and (Test-Path $rootReadme)) {
    git add README.md
}

Write-Host "==> git status" -ForegroundColor Cyan
git status

$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "Nothing to commit (no staged changes)." -ForegroundColor Yellow
    exit 0
}

$msg = "Add reproducibility code under code/ (experiments, prompts, src, configs, tests)"
git commit -m $msg
if ($LASTEXITCODE -ne 0) {
    Write-Error "git commit failed."
    exit $LASTEXITCODE
}

if ($SkipPush) {
    Write-Host "==> SkipPush: not running git push." -ForegroundColor Yellow
    exit 0
}

Write-Host "==> git push origin main" -ForegroundColor Cyan
git push origin main
if ($LASTEXITCODE -ne 0) {
    Write-Error "git push failed. Try: git pull origin main --rebase, then push again."
    exit $LASTEXITCODE
}

Write-Host "Done." -ForegroundColor Green
