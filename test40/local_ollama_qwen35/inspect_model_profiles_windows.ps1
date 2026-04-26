param(
    [string]$ProfileId,
    [switch]$DisableAutoProfile
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($PSBoundParameters.ContainsKey('ProfileId')) { $env:LOCAL_OLLAMA_PROFILE_ID = $ProfileId }
if ($DisableAutoProfile.IsPresent) { $env:LOCAL_OLLAMA_DISABLE_AUTO_PROFILE = '1' }

. "$PSScriptRoot\common_windows.ps1"

Assert-Windows
Detect-Accelerator
Print-AcceleratorSummary

$profilesDoc = Get-ModelProfiles
$profiles = @($profilesDoc.profiles | Sort-Object priority -Descending)

Write-Host ''
Write-Host 'Published model profiles (pre-download estimates):'
$profiles |
    Select-Object `
        @{Name = 'id'; Expression = { $_.id } },
        @{Name = 'size_gb'; Expression = { $_.published_size_gb } },
        @{Name = 'min_single_mb'; Expression = { Get-ObjectPropertyValue -Object $_ -Name 'min_single_gpu_vram_mb' -Default '' } },
        @{Name = 'min_total_mb'; Expression = { Get-ObjectPropertyValue -Object $_ -Name 'min_total_gpu_vram_mb' -Default '' } },
        @{Name = 'min_gpu_count'; Expression = { Get-ObjectPropertyValue -Object $_ -Name 'min_gpu_count' -Default 1 } },
        @{Name = 'visibility'; Expression = { Get-ObjectPropertyValue -Object $_ -Name 'visibility' -Default 'single' } },
        @{Name = 'model_source'; Expression = {
            $files = Get-StringArrayPropertyValue -Object $_ -Name 'model_files'
            if ($files.Count -gt 0) {
                ($files -join '; ')
            } else {
                Get-ObjectPropertyValue -Object $_ -Name 'model_file' -Default ''
            }
        } } |
    Format-Table -AutoSize
