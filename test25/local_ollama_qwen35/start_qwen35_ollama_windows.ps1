$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. "$PSScriptRoot\common_windows.ps1"

Assert-Windows
Detect-Accelerator
Set-RuntimeEnvironment
Write-RuntimeEnvironmentFile
Print-AcceleratorSummary

Ensure-OllamaInstalled
Ensure-OllamaServer -ForceRestart
Ensure-ModelFile
Ensure-OllamaModel
Warm-Model
Show-RuntimeStatus

Write-Msg 'Ollama is ready for test25.'
