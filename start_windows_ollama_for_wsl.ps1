param(
    [string]$Model = 'qwen36-35b-a3b-mxfp4-moe',
    [int]$ContextLength = 4096,
    [switch]$ForceRestart
)

$ErrorActionPreference = 'Stop'

function Write-Info {
    param([string]$Message)
    Write-Host "[ollama-wsl] $Message"
}

function Test-OllamaApi {
    try {
        Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Resolve-OllamaExe {
    $cmd = Get-Command 'ollama' -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'),
        (Join-Path $env:LOCALAPPDATA 'Ollama\ollama.exe'),
        (Join-Path $env:ProgramFiles 'Ollama\ollama.exe')
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) {
            return $path
        }
    }

    $whereExe = Join-Path $env:WINDIR 'System32\where.exe'
    if (Test-Path $whereExe) {
        $located = & $whereExe ollama 2>$null | Select-Object -First 1
        if ($located -and (Test-Path $located)) {
            return $located
        }
    }

    throw 'Could not locate ollama.exe. Install Ollama or add it to PATH.'
}

if ($ForceRestart) {
    $procs = Get-Process -Name 'ollama' -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Info 'Stopping existing Ollama processes so the WSL-visible host binding can take effect...'
        $procs | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
}

$ollama = Resolve-OllamaExe
$logs = Join-Path $PSScriptRoot 'reports'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stdout = Join-Path $logs 'ollama_windows_wsl.stdout.log'
$stderr = Join-Path $logs 'ollama_windows_wsl.stderr.log'

if ($ForceRestart -or -not (Test-OllamaApi)) {
    if ($ForceRestart) {
        $procs = Get-Process -Name 'ollama' -ErrorAction SilentlyContinue
        if ($procs) {
            Write-Info 'Stopping any auto-restarted localhost-only Ollama process...'
            $procs | Stop-Process -Force
            Start-Sleep -Milliseconds 500
        }
    }
    $launcher = @"
`$env:OLLAMA_HOST = '0.0.0.0:11434'
`$env:OLLAMA_NO_CLOUD = '1'
`$env:OLLAMA_CONTEXT_LENGTH = '$ContextLength'
`$env:OLLAMA_NUM_PARALLEL = '1'
`$env:OLLAMA_MAX_LOADED_MODELS = '1'
& '$ollama' serve
"@
    Write-Info 'Starting Ollama with OLLAMA_HOST=0.0.0.0:11434...'
    Start-Process -FilePath 'powershell.exe' `
        -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $launcher) `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr
}

$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    if (Test-OllamaApi) {
        $ready = $true
        break
    }
    Start-Sleep -Seconds 1
}

if (-not $ready) {
    throw "Ollama did not become ready within 60s. Check $stdout and $stderr"
}

Write-Info "Ollama API is up on Windows localhost. Model target: $Model"
Write-Info 'From WSL, run: ./detect_ollama_endpoint.sh'
Write-Info 'If WSL still cannot reach it, rerun this script with -ForceRestart.'
