param(
  [string]$Python = "C:\Users\vanya\AppData\Local\Programs\Python\Python312\python.exe",
  [string]$CardinalitySeeds = "5-10",
  [string]$CostSeeds = "4-13",
  [int]$Epochs = 100,
  [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

$Root = "C:\Users\vanya\Documents\DATA_SYS_LAB"
$Script = Join-Path $Root "test44\final_benchmarks\scripts\extend_mscn_seeds.py"
$LogDir = Join-Path $Root "test44\final_benchmarks\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$MainLog = Join-Path $LogDir "overnight_mscn_extra_seeds_$Stamp.log"

function Write-Log {
  param([string]$Message)
  $line = "$(Get-Date -Format o) $Message"
  Write-Host $line
  Add-Content -Path $MainLog -Value $line
}

function Invoke-Checked {
  param(
    [string]$Name,
    [string[]]$PyArgs
  )
  $stdout = Join-Path $LogDir "$Name.$Stamp.out.log"
  $stderr = Join-Path $LogDir "$Name.$Stamp.err.log"
  Write-Log "START $Name stdout=$stdout stderr=$stderr args=$($PyArgs -join ' ')"
  & $Python @PyArgs 1> $stdout 2> $stderr
  $exitCode = $LASTEXITCODE
  Write-Log "END $Name exit_code=$exitCode"
  if ($exitCode -ne 0) {
    Write-Log "FAILED $Name stderr_tail:"
    if (Test-Path $stderr) {
      Get-Content $stderr -Tail 80 | ForEach-Object { Write-Log "  $_" }
    }
    throw "$Name failed with exit code $exitCode"
  }
}

Write-Log "overnight_mscn_extra_seeds begin"

Invoke-Checked "cardinality_seeds_$($CardinalitySeeds -replace '[^0-9A-Za-z]+','_')" @(
  $Script,
  "--kind", "cardinality",
  "--seeds", $CardinalitySeeds,
  "--epochs", "$Epochs",
  "--device", $Device,
  "--skip-existing"
)

Invoke-Checked "cost_seeds_$($CostSeeds -replace '[^0-9A-Za-z]+','_')" @(
  $Script,
  "--kind", "cost",
  "--seeds", $CostSeeds,
  "--epochs", "$Epochs",
  "--device", $Device,
  "--skip-existing"
)

Write-Log "overnight_mscn_extra_seeds complete"
