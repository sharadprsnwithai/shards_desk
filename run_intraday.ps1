# ==============================================================================
#  Sharad's 10:00 AM Intraday Momentum Desk - Runner
# ==============================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Starting Sharad's 10:00 AM Intraday Momentum Desk..." -ForegroundColor Yellow

$venvPython = Join-Path $ScriptDir "venv\Scripts\python.exe"
$pyCmd = if (Test-Path $venvPython) { $venvPython } else { "python" }

& $pyCmd daily_engine.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERR] daily_engine.py exited with error code $LASTEXITCODE" -ForegroundColor Red
} else {
    Write-Host "[OK] 10:00 AM Intraday run completed successfully." -ForegroundColor Green
}
