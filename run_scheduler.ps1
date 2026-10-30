# ==============================================================================
#  Sharad's Trading Desk - Background Scheduler Runner
# ==============================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Starting Sharad's Option Desk Background Scheduler..." -ForegroundColor Cyan
Write-Host "  * Mon-Fri @ 10:00 AM IST: Intraday Momentum Desk -> Telegram" -ForegroundColor Gray
Write-Host "  * Last Day of Month @ 5:00 PM IST: Monthly Option Desk -> Telegram" -ForegroundColor Gray

$venvPython = Join-Path $ScriptDir "venv\Scripts\python.exe"
$pyCmd = if (Test-Path $venvPython) { $venvPython } else { "python" }

& $pyCmd scheduler.py
