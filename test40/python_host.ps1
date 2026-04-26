param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$ErrorActionPreference = "Stop"

$python = "C:\Users\vanya\Documents\DATA_SYS_LAB\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python executable not found: $python"
    exit 1
}

& $python @PythonArgs
$exitCode = $LASTEXITCODE
if ($null -eq $exitCode) {
    $exitCode = 0
}
exit $exitCode
