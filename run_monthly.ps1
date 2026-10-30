# ==============================================================================
#  Sharad's Monthly Option Selling Desk - Runner
# ==============================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Starting Sharad's 30-Day Monthly Option Selling Desk..." -ForegroundColor Cyan

# Check for venv
$venvPython = Join-Path $ScriptDir "venv\Scripts\python.exe"
$pyCmd = if (Test-Path $venvPython) { $venvPython } else { "python" }

& $pyCmd monthly_engine.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERR] monthly_engine.py exited with error code $LASTEXITCODE" -ForegroundColor Red
} else {
    Write-Host "[OK] Monthly run completed successfully." -ForegroundColor Green
}
