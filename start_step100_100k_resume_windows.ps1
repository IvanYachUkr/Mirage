$ErrorActionPreference = "Stop"

$BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $BaseDir "_runner_logs\candidate100k_2000_2050"
$RunnerScript = Join-Path $BaseDir "run_step100_100k_resume_windows.ps1"
$PidFile = Join-Path $LogDir "windows_step100_100k_resume_launcher.pid"
$PowerShellExe = if ($env:POWERSHELL_EXE) { $env:POWERSHELL_EXE } else { "powershell.exe" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$runner = Start-Process `
    -FilePath $PowerShellExe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunnerScript) `
    -WindowStyle Hidden `
    -PassThru

$runner.Id | Set-Content -Path $PidFile
Write-Host "started $($runner.Id)"
Write-Host "log $(Join-Path $LogDir 'windows_step100_100k_resume.log')"
