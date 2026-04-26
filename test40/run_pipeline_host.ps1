param(
    [string]$ConfigPath = "",
    [string]$StdoutLogPath = "",
    [string]$UsageLogPath = "",
    [switch]$UseSmokeDefaults,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArgs
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $scriptDir
$runLogDir = Join-Path $scriptDir "_runner_logs\full_api_pipeline"
$python = Join-Path $repo ".venv\Scripts\python.exe"
$script = Join-Path $scriptDir "run_pipeline.py"

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $scriptDir "benchmark_candidate_profile.json"
}
if ([string]::IsNullOrWhiteSpace($StdoutLogPath)) {
    $StdoutLogPath = Join-Path $runLogDir "host_run.log"
}
if ([string]::IsNullOrWhiteSpace($UsageLogPath)) {
    $UsageLogPath = Join-Path $runLogDir "host_run_llm_usage.jsonl"
}

if (-not (Test-Path $python)) {
    Write-Error "Python executable not found: $python"
    exit 1
}
if (-not (Test-Path $script)) {
    Write-Error "Pipeline script not found: $script"
    exit 1
}

$stdoutDir = Split-Path -Parent $StdoutLogPath
if ($stdoutDir) {
    New-Item -ItemType Directory -Path $stdoutDir -Force | Out-Null
}

$env:DATA_SYS_PIPELINE_CONFIG = $ConfigPath
$env:DATA_SYS_LLM_USAGE_LOG = $UsageLogPath

if ($UseSmokeDefaults) {
    $env:DATA_SYS_OVERLOAD_PROFILE = "smoke"
    $env:DATA_SYS_PERSON_ENRICH_BATCH_SIZE = "24"
    $env:DATA_SYS_PERSON_ENRICH_OUTER_RETRIES = "2"
    $env:DATA_SYS_LATENT_BATCH_SIZE = "24"
    $env:DATA_SYS_LATENT_MAX_RETRIES = "3"
}

"[$(Get-Date -Format s)] host launcher start" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null
"python: $python" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null
"script: $script" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null
"config: $ConfigPath" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null
"usage log: $UsageLogPath" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null
"args: $($PipelineArgs -join ' ')" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null

$allArgs = @($script) + $PipelineArgs
$stdoutTmp = Join-Path $env:TEMP ("datasys_host_stdout_" + [guid]::NewGuid().ToString("N") + ".log")
$stderrTmp = Join-Path $env:TEMP ("datasys_host_stderr_" + [guid]::NewGuid().ToString("N") + ".log")

try {
    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList $allArgs `
        -Wait `
        -PassThru `
        -NoNewWindow `
        -RedirectStandardOutput $stdoutTmp `
        -RedirectStandardError $stderrTmp

    if (Test-Path $stdoutTmp) {
        Get-Content $stdoutTmp | Tee-Object -FilePath $StdoutLogPath -Append
    }
    if (Test-Path $stderrTmp) {
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
}

"[$(Get-Date -Format s)] host launcher exit=$exitCode" | Tee-Object -FilePath $StdoutLogPath -Append | Out-Null
exit $exitCode
