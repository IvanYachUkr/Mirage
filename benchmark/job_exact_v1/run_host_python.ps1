param(
    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,
    [string]$StdoutLogPath = "",
    [string]$WorkingDir = "",
    [string]$ScriptArgString = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent (Split-Path -Parent $scriptDir)
$python = Join-Path $repo ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python executable not found: $python"
    exit 1
}

if (-not (Test-Path $ScriptPath)) {
    Write-Error "Target script not found: $ScriptPath"
    exit 1
}

if ([string]::IsNullOrWhiteSpace($StdoutLogPath)) {
    $runLogDir = Join-Path $scriptDir "_runner_logs"
    New-Item -ItemType Directory -Path $runLogDir -Force | Out-Null
    $StdoutLogPath = Join-Path $runLogDir (([System.IO.Path]::GetFileNameWithoutExtension($ScriptPath)) + ".log")
}

$stdoutDir = Split-Path -Parent $StdoutLogPath
if ($stdoutDir) {
    New-Item -ItemType Directory -Path $stdoutDir -Force | Out-Null
}

$resolvedWorkingDir = ""
if (-not [string]::IsNullOrWhiteSpace($WorkingDir)) {
    $resolvedWorkingDir = (Resolve-Path $WorkingDir).Path
}

$stdoutTmp = Join-Path $env:TEMP ("datasys_host_stdout_" + [guid]::NewGuid().ToString("N") + ".log")
$stderrTmp = Join-Path $env:TEMP ("datasys_host_stderr_" + [guid]::NewGuid().ToString("N") + ".log")

"[$(Get-Date -Format s)] host launcher start" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null
"python: $python" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null
"script: $ScriptPath" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null
if ($resolvedWorkingDir) {
    "cwd: $resolvedWorkingDir" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null
}
$argSummary = if (-not [string]::IsNullOrWhiteSpace($ScriptArgString)) { $ScriptArgString } else { $ScriptArgs -join ' ' }
"args: $argSummary" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $python
$renderedArgs = if (-not [string]::IsNullOrWhiteSpace($ScriptArgString)) {
    $ScriptArgString
} else {
    ($ScriptArgs | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }) -join ' '
}
$startInfo.Arguments = '"' + $ScriptPath + '" ' + $renderedArgs
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
if ($resolvedWorkingDir) {
    $startInfo.WorkingDirectory = $resolvedWorkingDir
}

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $startInfo

try {
    $null = $proc.Start()
    $stdoutText = $proc.StandardOutput.ReadToEnd()
    $stderrText = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()

    if ($stdoutText) {
        $stdoutText | Out-File -FilePath $stdoutTmp -Encoding utf8
        Get-Content $stdoutTmp | Tee-Object -FilePath $StdoutLogPath -Append
    }
    if ($stderrText) {
        $stderrText | Out-File -FilePath $stderrTmp -Encoding utf8
        Get-Content $stderrTmp | Tee-Object -FilePath $StdoutLogPath -Append
    }

    $exitCode = $proc.ExitCode
    if ($null -eq $exitCode) {
        $exitCode = 0
    }
}
finally {
    Remove-Item $stdoutTmp -ErrorAction SilentlyContinue
    Remove-Item $stderrTmp -ErrorAction SilentlyContinue
    if ($proc) {
        $proc.Dispose()
    }
}

"[$(Get-Date -Format s)] host launcher exit=$exitCode" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null
exit $exitCode
