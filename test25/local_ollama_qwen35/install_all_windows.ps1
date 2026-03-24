$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. "$PSScriptRoot\common_windows.ps1"

Assert-Windows
Write-Msg 'Installing test25 local Ollama bundle for Windows...'

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
