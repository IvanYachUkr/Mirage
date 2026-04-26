param(
    [string]$ProfileId,
    [switch]$DisableAutoProfile,
    [string]$ModelRepo,
    [string]$ModelFile,
    [string]$ModelName,
    [int]$MinGpuVramMb,
    [int]$CpuNumCtx,
    [int]$GpuNumCtx
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($PSBoundParameters.ContainsKey('ProfileId')) { $env:LOCAL_OLLAMA_PROFILE_ID = $ProfileId }
if ($DisableAutoProfile.IsPresent) { $env:LOCAL_OLLAMA_DISABLE_AUTO_PROFILE = '1' }
if ($PSBoundParameters.ContainsKey('ModelRepo')) { $env:LOCAL_OLLAMA_MODEL_REPO = $ModelRepo }
if ($PSBoundParameters.ContainsKey('ModelFile')) { $env:LOCAL_OLLAMA_MODEL_FILE = $ModelFile }
if ($PSBoundParameters.ContainsKey('ModelName')) { $env:LOCAL_OLLAMA_MODEL_NAME = $ModelName }
if ($PSBoundParameters.ContainsKey('MinGpuVramMb')) { $env:LOCAL_OLLAMA_MIN_GPU_VRAM_MB = [string]$MinGpuVramMb }
if ($PSBoundParameters.ContainsKey('CpuNumCtx')) { $env:LOCAL_OLLAMA_CPU_NUM_CTX = [string]$CpuNumCtx }
if ($PSBoundParameters.ContainsKey('GpuNumCtx')) { $env:LOCAL_OLLAMA_GPU_NUM_CTX = [string]$GpuNumCtx }

. "$PSScriptRoot\common_windows.ps1"

Assert-Windows
Write-Msg 'Installing local Ollama bundle for Windows...'

Ensure-PythonVenv
Ensure-OllamaInstalled

Detect-Accelerator
Set-RuntimeEnvironment
Write-RuntimeEnvironmentFile
Print-AcceleratorSummary
Warn-IfCpuOnlyMemoryIsTight

Ensure-OllamaServer -ForceRestart
Ensure-ModelFile
Ensure-OllamaModel
Warm-Model
Show-RuntimeStatus

Write-Msg 'Install complete.'
Write-Msg "Next command: $PSScriptRoot\run_pipeline_200k_local_windows.ps1"
