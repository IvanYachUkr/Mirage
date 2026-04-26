param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$candidatePaths = @()
if (-not [string]::IsNullOrWhiteSpace($env:PYTHON_EXE)) {
    $candidatePaths += $env:PYTHON_EXE
}
$candidatePaths += Join-Path $scriptDir ".venv\Scripts\python.exe"
$candidatePaths += Join-Path (Split-Path -Parent $scriptDir) ".venv\Scripts\python.exe"

$python = $null
foreach ($candidate in $candidatePaths) {
    if ($candidate -and (Test-Path $candidate)) {
        $python = $candidate
        break
    }
}

if (-not $python) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        $python = $cmd.Source
    }
}

if (-not $python) {
    Write-Error "Python executable not found. Set PYTHON_EXE or add python to PATH."
    exit 1
}

& $python @PythonArgs
$exitCode = $LASTEXITCODE
if ($null -eq $exitCode) {
    $exitCode = 0
}
exit $exitCode
