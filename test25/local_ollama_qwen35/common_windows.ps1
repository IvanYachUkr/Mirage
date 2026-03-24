$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$LogDir = Join-Path $ScriptDir 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ModelRepo = 'unsloth/Qwen3.5-35B-A3B-GGUF'
$ModelFile = 'Qwen3.5-35B-A3B-Q4_K_M.gguf'
$ModelName = 'qwen35-35b-a3b'
$ModelDir = Join-Path $HOME "models\$ModelName"
$RuntimeEnvFile = Join-Path $ScriptDir 'runtime_env.ps1'
$VenvDir = Join-Path $ProjectDir '.venv-local-ollama'
$RequirementsFile = Join-Path $ScriptDir 'requirements.local_pipeline.txt'
$MinGpuVramMb = 24000
$CpuNumCtx = 4096
$GpuNumCtx = 8192

$script:LocalAccelerator = 'cpu'
$script:LocalGpuVendor = 'none'
$script:LocalGpuId = ''
$script:LocalGpuVramMb = 0
$script:LocalAcceleratorReason = 'not-detected'
$script:LocalNumCtx = $CpuNumCtx

function Write-Msg {
    param([string]$Message)
    Write-Host "[test25-local] $Message"
}

function Throw-Fail {
    param([string]$Message)
    throw "[test25-local][ERROR] $Message"
}

function Assert-Windows {
    $isWindows = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
        [System.Runtime.InteropServices.OSPlatform]::Windows
    )
    if (-not $isWindows) {
        Throw-Fail "These PowerShell scripts are intended for Windows."
    }
}

function Get-PythonCommand {
    foreach ($candidate in @('python', 'py')) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Name
        }
    }
    Throw-Fail "Python was not found on PATH."
}

function Get-VenvPython {
    return Join-Path $VenvDir 'Scripts\python.exe'
}

function Get-BestNvidiaGpu {
    $nvidia = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    if (-not $nvidia) {
        return $null
    }
    try {
        $lines = & $nvidia.Source --query-gpu=index,memory.total,gpu_uuid,name --format=csv,noheader,nounits 2>$null
    } catch {
        return $null
    }
    $best = $null
    foreach ($line in $lines) {
        $parts = $line -split ','
        if ($parts.Count -lt 4) {
            continue
        }
        $idx = $parts[0].Trim()
        $memRaw = $parts[1].Trim()
        $uuid = $parts[2].Trim()
        $name = $parts[3].Trim()
        $memMb = 0
        if (-not [int]::TryParse([string]([math]::Round([double]$memRaw)), [ref]$memMb)) {
            continue
        }
        $deviceId = if ($uuid) { $uuid } else { $idx }
        $entry = [pscustomobject]@{
            Vendor = 'nvidia'
            DeviceId = $deviceId
            VramMb = $memMb
            Name = $name
        }
        if (-not $best -or $entry.VramMb -gt $best.VramMb) {
            $best = $entry
        }
    }
    return $best
}

function Get-BestAmdGpu {
    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction Stop |
            Where-Object {
                $_.Name -match 'AMD|Radeon' -and
                $_.Name -notmatch 'Microsoft Basic'
            }
    } catch {
        return $null
    }

    $best = $null
    $ordinal = 0
    foreach ($gpu in $gpus) {
        $adapterRam = [int64]$gpu.AdapterRAM
        if ($adapterRam -le 0) {
            $ordinal++
            continue
        }
        $memMb = [int][math]::Floor($adapterRam / 1MB)
        $entry = [pscustomobject]@{
            Vendor = 'amd'
            DeviceId = [string]$ordinal
            VramMb = $memMb
            Name = [string]$gpu.Name
        }
        if (-not $best -or $entry.VramMb -gt $best.VramMb) {
            $best = $entry
        }
        $ordinal++
    }
    return $best
}

function Detect-Accelerator {
    $candidates = @()
    $nvidia = Get-BestNvidiaGpu
    if ($nvidia) { $candidates += $nvidia }
    $amd = Get-BestAmdGpu
    if ($amd) { $candidates += $amd }

    if ($candidates.Count -eq 0) {
        $script:LocalAccelerator = 'cpu'
        $script:LocalGpuVendor = 'none'
        $script:LocalGpuId = ''
        $script:LocalGpuVramMb = 0
        $script:LocalAcceleratorReason = 'no_eligible_gpu_detected'
        $script:LocalNumCtx = $CpuNumCtx
        return
    }

    $best = $candidates | Sort-Object VramMb -Descending | Select-Object -First 1
    $script:LocalGpuVendor = $best.Vendor
    $script:LocalGpuId = $best.DeviceId
    $script:LocalGpuVramMb = [int]$best.VramMb

    if ($best.VramMb -ge $MinGpuVramMb) {
        $script:LocalAccelerator = 'gpu'
        $script:LocalAcceleratorReason = "eligible_${($best.Vendor)}_gpu"
        $script:LocalNumCtx = $GpuNumCtx
    } else {
        $script:LocalAccelerator = 'cpu'
        $script:LocalAcceleratorReason = "${($best.Vendor)}_gpu_below_24gb"
        $script:LocalNumCtx = $CpuNumCtx
    }
}

function Set-RuntimeEnvironment {
    $env:OLLAMA_HOST = '127.0.0.1:11434'
    $env:OLLAMA_NO_CLOUD = '1'
    $env:OLLAMA_CONTEXT_LENGTH = [string]$script:LocalNumCtx
    $env:OLLAMA_NUM_PARALLEL = '1'
    $env:OLLAMA_MAX_LOADED_MODELS = '1'
    $env:OLLAMA_MAX_QUEUE = '128'

    $env:LLM_PROVIDER = 'local'
    $env:LOCAL_LLM_URL = 'http://127.0.0.1:11434/v1'
    $env:LOCAL_LLM_MODEL = $ModelName
    $env:LOCAL_LLM_API_KEY = 'not-needed'

    $env:TEST25_LOCAL_ACCELERATOR = $script:LocalAccelerator
    $env:TEST25_LOCAL_GPU_VENDOR = $script:LocalGpuVendor
    $env:TEST25_LOCAL_GPU_ID = $script:LocalGpuId
    $env:TEST25_LOCAL_GPU_VRAM_MB = [string]$script:LocalGpuVramMb
    $env:TEST25_LOCAL_ACCELERATOR_REASON = $script:LocalAcceleratorReason

    if ($script:LocalAccelerator -eq 'gpu') {
        if ($script:LocalGpuVendor -eq 'nvidia') {
            $env:CUDA_VISIBLE_DEVICES = $script:LocalGpuId
            Remove-Item Env:ROCR_VISIBLE_DEVICES, Env:HIP_VISIBLE_DEVICES, Env:GPU_DEVICE_ORDINAL -ErrorAction SilentlyContinue
        } elseif ($script:LocalGpuVendor -eq 'amd') {
            $env:ROCR_VISIBLE_DEVICES = $script:LocalGpuId
            Remove-Item Env:CUDA_VISIBLE_DEVICES, Env:HIP_VISIBLE_DEVICES, Env:GPU_DEVICE_ORDINAL -ErrorAction SilentlyContinue
        }
    } else {
        $env:CUDA_VISIBLE_DEVICES = '-1'
        $env:ROCR_VISIBLE_DEVICES = '-1'
        $env:HIP_VISIBLE_DEVICES = '-1'
        $env:GPU_DEVICE_ORDINAL = '-1'
    }
}

function Write-RuntimeEnvironmentFile {
    $lines = @(
        '$env:OLLAMA_HOST = ''127.0.0.1:11434'''
        '$env:OLLAMA_NO_CLOUD = ''1'''
        ('$env:OLLAMA_CONTEXT_LENGTH = ''' + $script:LocalNumCtx + '''')
        '$env:OLLAMA_NUM_PARALLEL = ''1'''
        '$env:OLLAMA_MAX_LOADED_MODELS = ''1'''
        '$env:OLLAMA_MAX_QUEUE = ''128'''
        '$env:LLM_PROVIDER = ''local'''
        '$env:LOCAL_LLM_URL = ''http://127.0.0.1:11434/v1'''
        ('$env:LOCAL_LLM_MODEL = ''' + $ModelName + '''')
        '$env:LOCAL_LLM_API_KEY = ''not-needed'''
        ('$env:TEST25_LOCAL_ACCELERATOR = ''' + $script:LocalAccelerator + '''')
        ('$env:TEST25_LOCAL_GPU_VENDOR = ''' + $script:LocalGpuVendor + '''')
        ('$env:TEST25_LOCAL_GPU_ID = ''' + $script:LocalGpuId + '''')
        ('$env:TEST25_LOCAL_GPU_VRAM_MB = ''' + $script:LocalGpuVramMb + '''')
        ('$env:TEST25_LOCAL_ACCELERATOR_REASON = ''' + $script:LocalAcceleratorReason + '''')
    )
    if ($script:LocalAccelerator -eq 'gpu' -and $script:LocalGpuVendor -eq 'nvidia') {
        $lines += ('$env:CUDA_VISIBLE_DEVICES = ''' + $script:LocalGpuId + '''')
    } elseif ($script:LocalAccelerator -eq 'gpu' -and $script:LocalGpuVendor -eq 'amd') {
        $lines += ('$env:ROCR_VISIBLE_DEVICES = ''' + $script:LocalGpuId + '''')
    } else {
        $lines += @(
            '$env:CUDA_VISIBLE_DEVICES = ''-1'''
            '$env:ROCR_VISIBLE_DEVICES = ''-1'''
            '$env:HIP_VISIBLE_DEVICES = ''-1'''
            '$env:GPU_DEVICE_ORDINAL = ''-1'''
        )
    }
    Set-Content -Path $RuntimeEnvFile -Encoding ASCII -Value $lines
}

function Print-AcceleratorSummary {
    Write-Msg "Accelerator mode: $($script:LocalAccelerator)"
    Write-Msg "GPU vendor:       $($script:LocalGpuVendor)"
    Write-Msg "GPU id:           $(if ($script:LocalGpuId) { $script:LocalGpuId } else { '<none>' })"
    Write-Msg "GPU VRAM MB:      $($script:LocalGpuVramMb)"
    Write-Msg "Reason:           $($script:LocalAcceleratorReason)"
    Write-Msg "num_ctx:          $($script:LocalNumCtx)"
}

function Ensure-PythonVenv {
    $python = Get-PythonCommand
    if (-not (Test-Path $VenvDir)) {
        & $python -m venv $VenvDir
    }
    $venvPython = Get-VenvPython
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r $RequirementsFile
}

function Get-HfCliPath {
    $hfCli = Join-Path $VenvDir 'Scripts\hf.exe'
    if (-not (Test-Path $hfCli)) {
        Throw-Fail "hf.exe was not found in $VenvDir. Re-run install_all_windows.ps1."
    }
    return $hfCli
}

function Ensure-OllamaInstalled {
    $cmd = Get-Command 'ollama' -ErrorAction SilentlyContinue
    if ($cmd) {
        Write-Msg 'Ollama already installed.'
        return
    }
    Write-Msg 'Installing Ollama...'
    Invoke-RestMethod 'https://ollama.com/install.ps1' | Invoke-Expression
}

function Test-OllamaApiReady {
    try {
        Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -Method Get -TimeoutSec 5 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Stop-OllamaProcesses {
    $procs = Get-Process -Name 'ollama' -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Msg 'Stopping existing Ollama processes to enforce the selected runtime mode...'
        $procs | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
}

function Ensure-OllamaServer {
    param(
        [switch]$ForceRestart
    )

    if ($ForceRestart) {
        Stop-OllamaProcesses
    } elseif (Test-OllamaApiReady) {
        Write-Msg 'Ollama API already available on 127.0.0.1:11434.'
        return
    }

    Write-RuntimeEnvironmentFile
    $ollamaCmd = (Get-Command 'ollama').Source
    $launcher = @"
. '$RuntimeEnvFile'
& '$ollamaCmd' serve
"@
    $stdoutLogPath = Join-Path $LogDir 'ollama_serve_windows.stdout.log'
    $stderrLogPath = Join-Path $LogDir 'ollama_serve_windows.stderr.log'
    Start-Process -FilePath 'powershell.exe' `
        -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $launcher) `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLogPath `
        -RedirectStandardError $stderrLogPath

    $waited = 0
    while (-not (Test-OllamaApiReady)) {
        Start-Sleep -Seconds 1
        $waited++
        if ($waited -ge 60) {
            Throw-Fail "Ollama API did not come up within 60s. Check $stdoutLogPath and $stderrLogPath"
        }
    }
    Write-Msg 'Ollama server is up.'
}

function Ensure-ModelFile {
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    $target = Join-Path $ModelDir $ModelFile
    if (Test-Path $target) {
        Write-Msg "GGUF already present: $target"
        return
    }
    Write-Msg 'Downloading GGUF from Hugging Face...'
    $hfCli = Get-HfCliPath
    & $hfCli download $ModelRepo $ModelFile --local-dir $ModelDir
}

function Write-Modelfile {
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    @"
FROM $ModelDir\$ModelFile
PARAMETER num_ctx $($script:LocalNumCtx)
"@ | Set-Content -Path (Join-Path $ModelDir 'Modelfile') -Encoding ASCII
}

function Test-OllamaModelExists {
    try {
        $lines = & ollama list 2>$null
    } catch {
        return $false
    }
    foreach ($line in $lines) {
        if ($line -match "^\s*$([regex]::Escape($ModelName))(\s|$)") {
            return $true
        }
    }
    return $false
}

function Ensure-OllamaModel {
    Write-Modelfile
    if (Test-OllamaModelExists) {
        Write-Msg "Ollama model already imported: $ModelName"
        return
    }
    Write-Msg "Importing GGUF into Ollama as $ModelName..."
    & ollama create $ModelName -f (Join-Path $ModelDir 'Modelfile')
}

function Warm-Model {
    Write-Msg "Warming model $ModelName..."
    $body = @{
        model = $ModelName
        prompt = 'ping'
        stream = $false
        options = @{
            num_ctx = 128
        }
    } | ConvertTo-Json -Depth 5
    Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/generate' -Method Post -ContentType 'application/json' -Body $body | Out-Null
}

function Show-RuntimeStatus {
    Write-Msg 'Ollama runtime status:'
    & ollama ps
}

function Warn-IfCpuOnlyMemoryIsTight {
    if ($script:LocalAccelerator -ne 'cpu') {
        return
    }
    try {
        $totalBytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
        $memGb = [math]::Floor([double]$totalBytes / 1GB)
        if ($memGb -lt 48) {
            Write-Msg "WARNING: CPU-only mode with ${memGb} GB RAM may be rough for this 22 GB GGUF."
        }
    } catch {
    }
}

function Run-Pipeline200k {
    param(
        [string[]]$ExtraArgs = @()
    )

    $venvPython = Get-VenvPython
    if (-not (Test-Path $venvPython)) {
        Throw-Fail "Python venv not found at $venvPython. Run install_all_windows.ps1 first."
    }

    & $venvPython (Join-Path $ProjectDir 'run_pipeline.py') `
        --fresh `
        --n-movies 200000 `
        --start-year 1950 `
        --end-year 2025 `
        --n-titles 200000 `
        --n-persons 450000 `
        --n-companies 30000 `
        --n-keywords 38000 `
        --n-characters 786000 `
        --enable-llm-evolution `
        --until-step 100 `
        @ExtraArgs
}
