$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. "$PSScriptRoot\common_windows.ps1"

Assert-Windows
Detect-Accelerator
Set-RuntimeEnvironment
Write-RuntimeEnvironmentFile
Print-AcceleratorSummary

Ensure-PythonVenv
Ensure-OllamaInstalled
Ensure-OllamaServer -ForceRestart
Ensure-ModelFile
Ensure-OllamaModel
Warm-Model
Show-RuntimeStatus

Write-Msg 'Starting test25 200k pipeline with local Ollama model on Windows...'
Run-Pipeline200k -ExtraArgs $args
