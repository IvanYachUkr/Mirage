$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

param(
  [int]$Movies = 100,
  [int]$UntilStep = 130,
  [string]$Model = "gemini-3.1-flash-lite-preview",
  [switch]$SkipCompare,
  [switch]$RunCalibration
)

$python = 'C:\Users\vanya\AppData\Local\Programs\Python\Python312\python.exe'
$script = Join-Path $PSScriptRoot 'run_full_api_pipeline.py'

$cmd = @(
  $script,
  '--n-movies', $Movies,
  '--until-step', $UntilStep,
  '--model', $Model
)

if ($SkipCompare) {
  $cmd += '--skip-compare'
}
if ($RunCalibration) {
  $cmd += '--run-calibration'
}

& $python -u @cmd
