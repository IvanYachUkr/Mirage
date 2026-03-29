param(
  [int]$Movies = 200000,
  [int]$UntilStep = 130,
  [string]$Model = "gemini-3.1-flash-lite-preview",
  [switch]$SkipCompare,
  [switch]$RunCalibration
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-Python {
  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) {
    return @($py.Source, '-3')
  }
  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) {
    return @($python.Source)
  }
  throw "Could not find Python 3. Install Python and make sure 'py' or 'python' is on PATH."
}

$python = Resolve-Python
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

if ($python.Length -gt 1) {
  & $python[0] $python[1..($python.Length - 1)] -u @cmd
} else {
  & $python[0] -u @cmd
}
