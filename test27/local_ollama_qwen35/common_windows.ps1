$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$LogDir = Join-Path $ScriptDir 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ProfilesFile = Join-Path $ScriptDir 'model_profiles.json'
$RuntimeEnvFile = Join-Path $ScriptDir 'runtime_env.ps1'
$VenvDir = Join-Path $ProjectDir '.venv-local-ollama'
$RequirementsFile = Join-Path $ScriptDir 'requirements.local_pipeline.txt'
$DefaultMinGpuVramMb = 24000
$DefaultCpuNumCtx = 4096
$DefaultGpuNumCtx = 8192

function Get-EnvOrDefault {
    param(
        [string]$Name,
        [string]$Default
    )
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    return $value.Trim()
}

function Get-EnvIntOrDefault {
    param(
        [string]$Name,
        [int]$Default
    )
    $raw = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $Default
    }
    $parsed = 0
    if ([int]::TryParse($raw.Trim(), [ref]$parsed)) {
        return $parsed
    }
    return $Default
}

function Get-EnvBoolOrDefault {
    param(
        [string]$Name,
        [bool]$Default
    )
    $raw = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $Default
    }
    switch ($raw.Trim().ToLowerInvariant()) {
        '1' { return $true }
        'true' { return $true }
        'yes' { return $true }
        'on' { return $true }
        '0' { return $false }
        'false' { return $false }
        'no' { return $false }
        'off' { return $false }
        default { return $Default }
    }
}

function Get-ObjectPropertyValue {
    param(
        [object]$Object,
        [string]$Name,
        $Default = $null
    )
    if (-not $Object) {
        return $Default
    }
    $prop = $Object.PSObject.Properties[$Name]
    if ($null -eq $prop) {
        return $Default
    }
    return $prop.Value
}

function Get-StringArrayPropertyValue {
    param(
        [object]$Object,
        [string]$Name
    )
    $value = Get-ObjectPropertyValue -Object $Object -Name $Name -Default $null
    if ($null -eq $value) {
        return ,@()
    }
    if ($value -is [System.Array]) {
        return ,@($value | ForEach-Object { [string]$_ })
    }
    return ,@([string]$value)
}

$ExplicitModelRepo = [Environment]::GetEnvironmentVariable('LOCAL_OLLAMA_MODEL_REPO')
$ExplicitModelFile = [Environment]::GetEnvironmentVariable('LOCAL_OLLAMA_MODEL_FILE')
$ExplicitModelName = [Environment]::GetEnvironmentVariable('LOCAL_OLLAMA_MODEL_NAME')
$ExplicitProfileId = Get-EnvOrDefault 'LOCAL_OLLAMA_PROFILE_ID' ''
$DisableAutoProfile = Get-EnvBoolOrDefault 'LOCAL_OLLAMA_DISABLE_AUTO_PROFILE' $false
$HasExplicitModelOverride = @($ExplicitModelRepo, $ExplicitModelFile, $ExplicitModelName) |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    Measure-Object |
    Select-Object -ExpandProperty Count
$HasExplicitModelOverride = ($HasExplicitModelOverride -gt 0)

$ModelRepo = ''
$ModelFile = ''
$ModelFiles = @()
$ModelMergeOutputFile = ''
$ModelImportPath = ''
$ModelName = ''
$ModelDir = ''
$ModelProfileId = ''
$ModelProfilePublishedSizeGb = 0.0
$ModelSelectionReason = ''
$MinGpuVramMb = Get-EnvIntOrDefault 'LOCAL_OLLAMA_MIN_GPU_VRAM_MB' $DefaultMinGpuVramMb
$CpuNumCtx = Get-EnvIntOrDefault 'LOCAL_OLLAMA_CPU_NUM_CTX' $DefaultCpuNumCtx
$GpuNumCtx = Get-EnvIntOrDefault 'LOCAL_OLLAMA_GPU_NUM_CTX' $DefaultGpuNumCtx

$script:LocalAccelerator = 'cpu'
$script:LocalGpuVendor = 'none'
$script:LocalGpuId = ''
$script:LocalGpuVramMb = 0
$script:LocalGpuCount = 0
$script:LocalTotalGpuVramMb = 0
$script:LocalVisibleGpuIds = ''
$script:LocalAcceleratorReason = 'not-detected'
$script:LocalNumCtx = $CpuNumCtx
$script:GpuInventory = @()
$script:GpuVendorSummaries = @{}
$script:ModelProfiles = $null
$script:SelectedProfile = $null
$script:SelectedVendor = $null

function Write-Msg {
    param([string]$Message)
    Write-Host "[local-ollama] $Message"
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

function Get-NvidiaGpus {
    $nvidia = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    if (-not $nvidia) {
        return @()
    }
    try {
        $lines = & $nvidia.Source --query-gpu=index,memory.total,name --format=csv,noheader,nounits 2>$null
    } catch {
        return @()
    }

    $items = @()
    foreach ($line in $lines) {
        $parts = $line -split ','
        if ($parts.Count -lt 3) {
            continue
        }
        $idx = $parts[0].Trim()
        $memRaw = $parts[1].Trim()
        $name = ($parts[2..($parts.Count - 1)] -join ',').Trim()
        $memMb = 0
        if (-not [int]::TryParse([string]([math]::Round([double]$memRaw)), [ref]$memMb)) {
            continue
        }
        $items += [pscustomobject]@{
            Vendor = 'nvidia'
            DeviceId = $idx
            VramMb = $memMb
            Name = $name
        }
    }
    return $items
}

function Get-AmdGpus {
    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction Stop |
            Where-Object {
                $_.Name -match 'AMD|Radeon' -and
                $_.Name -notmatch 'Microsoft Basic'
            }
    } catch {
        return @()
    }

    $items = @()
    $ordinal = 0
    foreach ($gpu in $gpus) {
        $adapterRam = [int64]$gpu.AdapterRAM
        if ($adapterRam -le 0) {
            $ordinal++
            continue
        }
        $items += [pscustomobject]@{
            Vendor = 'amd'
            DeviceId = [string]$ordinal
            VramMb = [int][math]::Floor($adapterRam / 1MB)
            Name = [string]$gpu.Name
        }
        $ordinal++
    }
    return $items
}

function Get-ModelProfiles {
    if ($script:ModelProfiles) {
        return $script:ModelProfiles
    }
    if (-not (Test-Path $ProfilesFile)) {
        Throw-Fail "Model profiles file not found: $ProfilesFile"
    }
    $raw = Get-Content -Path $ProfilesFile -Raw | ConvertFrom-Json
    $script:ModelProfiles = $raw
    return $script:ModelProfiles
}

function Get-ProfileById {
    param([string]$Id)
    $profilesDoc = Get-ModelProfiles
    $profiles = @($profilesDoc.profiles)
    return $profiles | Where-Object { $_.id -eq $Id } | Select-Object -First 1
}

function Build-GpuVendorSummaries {
    param([object[]]$Gpus)

    $map = @{}
    foreach ($group in ($Gpus | Group-Object Vendor)) {
        $items = @(
            $group.Group | Sort-Object `
                @{Expression = { [int]$_.VramMb }; Descending = $true },
                @{Expression = { [int]$_.DeviceId }; Descending = $false }
        )
        $best = $items | Select-Object -First 1
        $visibleIds = ($items | Sort-Object {[int]$_.DeviceId} | ForEach-Object { $_.DeviceId }) -join ','
        $map[$group.Name] = [pscustomobject]@{
            Vendor = $group.Name
            Count = $items.Count
            TotalVramMb = [int](($items | Measure-Object -Property VramMb -Sum).Sum)
            BestVramMb = [int]$best.VramMb
            BestDeviceId = [string]$best.DeviceId
            VisibleIds = $visibleIds
            BestName = [string]$best.Name
            Names = ($items | Select-Object -ExpandProperty Name -Unique)
        }
    }
    return $map
}

function Test-ProfileFitsVendorSummary {
    param(
        [object]$Profile,
        [object]$Summary
    )

    if (-not $Summary) {
        return $false
    }

    $minSingle = [int](Get-ObjectPropertyValue -Object $Profile -Name 'min_single_gpu_vram_mb' -Default 0)
    $minTotal = [int](Get-ObjectPropertyValue -Object $Profile -Name 'min_total_gpu_vram_mb' -Default 0)
    $minCount = [int](Get-ObjectPropertyValue -Object $Profile -Name 'min_gpu_count' -Default 1)
    $visibility = [string](Get-ObjectPropertyValue -Object $Profile -Name 'visibility' -Default 'single')

    if ($Summary.Count -lt $minCount) {
        return $false
    }
    if ($Summary.BestVramMb -lt $minSingle) {
        return $false
    }
    if ($visibility -eq 'all' -and $Summary.TotalVramMb -lt $minTotal) {
        return $false
    }
    return $true
}

function Resolve-ModelSelection {
    $profilesDoc = Get-ModelProfiles
    $profiles = @($profilesDoc.profiles | Sort-Object priority -Descending)

    if ($HasExplicitModelOverride) {
        $script:SelectedProfile = $null
        $script:ModelRepo = Get-EnvOrDefault 'LOCAL_OLLAMA_MODEL_REPO' ''
        $script:ModelFile = Get-EnvOrDefault 'LOCAL_OLLAMA_MODEL_FILE' ''
        $script:ModelFiles = @()
        if (-not [string]::IsNullOrWhiteSpace($script:ModelFile)) {
            $script:ModelFiles = @($script:ModelFile)
        }
        $script:ModelMergeOutputFile = ''
        $script:ModelName = Get-EnvOrDefault 'LOCAL_OLLAMA_MODEL_NAME' ''
        $script:ModelProfileId = if ($ExplicitProfileId) { $ExplicitProfileId } else { 'custom' }
        $script:ModelProfilePublishedSizeGb = 0.0
        $script:ModelSelectionReason = 'manual_model_override'
        if ([string]::IsNullOrWhiteSpace($script:ModelRepo) -or [string]::IsNullOrWhiteSpace($script:ModelFile) -or [string]::IsNullOrWhiteSpace($script:ModelName)) {
            Throw-Fail 'When overriding model fields manually, LOCAL_OLLAMA_MODEL_REPO, LOCAL_OLLAMA_MODEL_FILE, and LOCAL_OLLAMA_MODEL_NAME must all be set.'
        }
        $script:ModelDir = Join-Path $HOME "models\$script:ModelName"
        $script:ModelImportPath = Join-Path $script:ModelDir $script:ModelFile
        return
    }

    $selected = $null
    $selectedVendor = $null
    if ($ExplicitProfileId) {
        $selected = Get-ProfileById -Id $ExplicitProfileId
        if (-not $selected) {
            Throw-Fail "Unknown LOCAL_OLLAMA_PROFILE_ID: $ExplicitProfileId"
        }
        $script:ModelSelectionReason = "explicit_profile:$ExplicitProfileId"
    } elseif ($DisableAutoProfile) {
        $selected = Get-ProfileById -Id $profilesDoc.default_profile_id
        $script:ModelSelectionReason = "auto_profile_disabled:$($profilesDoc.default_profile_id)"
    } else {
        $candidates = @()
        foreach ($profile in $profiles) {
            foreach ($vendor in $script:GpuVendorSummaries.Keys) {
                $summary = $script:GpuVendorSummaries[$vendor]
                if (Test-ProfileFitsVendorSummary -Profile $profile -Summary $summary) {
                    $candidates += [pscustomobject]@{
                        Profile = $profile
                        Vendor = $vendor
                        Summary = $summary
                    }
                }
            }
        }
        if ($candidates.Count -gt 0) {
            $best = $candidates |
                Sort-Object @{Expression = { [int]$_.Profile.priority }; Descending = $true },
                            @{Expression = { [int]$_.Summary.TotalVramMb }; Descending = $true },
                            @{Expression = { [int]$_.Summary.BestVramMb }; Descending = $true } |
                Select-Object -First 1
            $selected = $best.Profile
            $selectedVendor = $best.Vendor
            $script:ModelSelectionReason = "auto_profile_fit:$($selected.id):$selectedVendor"
        } else {
            $selected = Get-ProfileById -Id $profilesDoc.fallback_profile_id
            $script:ModelSelectionReason = "auto_profile_fallback:$($profilesDoc.fallback_profile_id)"
        }
    }

    if (-not $selected) {
        Throw-Fail 'No model profile could be selected.'
    }

    $script:SelectedProfile = $selected
    $script:SelectedVendor = $selectedVendor
    $script:ModelRepo = [string](Get-ObjectPropertyValue -Object $selected -Name 'model_repo' -Default '')
    $script:ModelFile = [string](Get-ObjectPropertyValue -Object $selected -Name 'model_file' -Default '')
    $script:ModelFiles = @(Get-StringArrayPropertyValue -Object $selected -Name 'model_files')
    if ($script:ModelFiles.Count -eq 0 -and -not [string]::IsNullOrWhiteSpace($script:ModelFile)) {
        $script:ModelFiles = @($script:ModelFile)
    }
    $script:ModelMergeOutputFile = [string](Get-ObjectPropertyValue -Object $selected -Name 'merge_output_file' -Default '')
    $script:ModelName = [string](Get-ObjectPropertyValue -Object $selected -Name 'model_name' -Default '')
    $script:ModelProfileId = [string](Get-ObjectPropertyValue -Object $selected -Name 'id' -Default '')
    $script:ModelProfilePublishedSizeGb = [double](Get-ObjectPropertyValue -Object $selected -Name 'published_size_gb' -Default 0.0)
    if (-not [Environment]::GetEnvironmentVariable('LOCAL_OLLAMA_MIN_GPU_VRAM_MB')) {
        $profileMin = [int](Get-ObjectPropertyValue -Object $selected -Name 'min_single_gpu_vram_mb' -Default $DefaultMinGpuVramMb)
        $script:MinGpuVramMb = $profileMin
    }
    $script:ModelDir = Join-Path $HOME "models\$script:ModelName"
    if (-not [string]::IsNullOrWhiteSpace($script:ModelMergeOutputFile)) {
        $script:ModelImportPath = Join-Path $script:ModelDir $script:ModelMergeOutputFile
    } elseif (-not [string]::IsNullOrWhiteSpace($script:ModelFile)) {
        $script:ModelImportPath = Join-Path $script:ModelDir $script:ModelFile
    } elseif ($script:ModelFiles.Count -gt 0) {
        $script:ModelImportPath = Join-Path $script:ModelDir $script:ModelFiles[0]
    } else {
        Throw-Fail "Selected model profile $script:ModelProfileId does not define a GGUF source."
    }
}

function Resolve-AcceleratorMode {
    $script:LocalAccelerator = 'cpu'
    $script:LocalGpuVendor = 'none'
    $script:LocalGpuId = ''
    $script:LocalGpuVramMb = 0
    $script:LocalGpuCount = 0
    $script:LocalTotalGpuVramMb = 0
    $script:LocalVisibleGpuIds = ''
    $script:LocalAcceleratorReason = 'no_eligible_gpu_detected'
    $script:LocalNumCtx = $CpuNumCtx

    if ($script:GpuInventory.Count -eq 0) {
        return
    }

    $summary = $null
    if ($script:SelectedProfile -and $script:SelectedVendor -and $script:GpuVendorSummaries.ContainsKey($script:SelectedVendor)) {
        $summary = $script:GpuVendorSummaries[$script:SelectedVendor]
    } elseif ($script:SelectedProfile) {
        $eligible = @()
        foreach ($vendor in $script:GpuVendorSummaries.Keys) {
            $candidate = $script:GpuVendorSummaries[$vendor]
            if (Test-ProfileFitsVendorSummary -Profile $script:SelectedProfile -Summary $candidate) {
                $eligible += $candidate
            }
        }
        if ($eligible.Count -gt 0) {
            $summary = $eligible |
                Sort-Object `
                    @{Expression = { [int]$_.TotalVramMb }; Descending = $true },
                    @{Expression = { [int]$_.BestVramMb }; Descending = $true } |
                Select-Object -First 1
        }
    }

    if (-not $summary) {
        $summary = $script:GpuInventory | Sort-Object VramMb -Descending | Select-Object -First 1
        $summary = [pscustomobject]@{
            Vendor = $summary.Vendor
            Count = 1
            TotalVramMb = [int]$summary.VramMb
            BestVramMb = [int]$summary.VramMb
            BestDeviceId = [string]$summary.DeviceId
            VisibleIds = [string]$summary.DeviceId
            BestName = [string]$summary.Name
            Names = @([string]$summary.Name)
        }
    }

    $visibility = if ($script:SelectedProfile) {
        [string](Get-ObjectPropertyValue -Object $script:SelectedProfile -Name 'visibility' -Default 'single')
    } else {
        'single'
    }
    $profileFits = if ($script:SelectedProfile) { Test-ProfileFitsVendorSummary -Profile $script:SelectedProfile -Summary $summary } else { $summary.BestVramMb -ge $MinGpuVramMb }
    if (-not $profileFits) {
        $script:LocalGpuVendor = [string]$summary.Vendor
        $script:LocalGpuId = [string]$summary.BestDeviceId
        $script:LocalGpuVramMb = [int]$summary.BestVramMb
        $script:LocalGpuCount = [int]$summary.Count
        $script:LocalTotalGpuVramMb = [int]$summary.TotalVramMb
        $script:LocalVisibleGpuIds = [string]$summary.VisibleIds
        $script:LocalAcceleratorReason = "selected_profile_not_fit:$ModelProfileId"
        return
    }

    $script:LocalAccelerator = 'gpu'
    $script:LocalGpuVendor = [string]$summary.Vendor
    $script:LocalGpuVramMb = [int]$summary.BestVramMb
    $script:LocalGpuCount = [int]$summary.Count
    $script:LocalTotalGpuVramMb = [int]$summary.TotalVramMb
    $script:LocalNumCtx = $GpuNumCtx
    if ($visibility -eq 'all' -and $summary.Count -gt 1) {
        $script:LocalGpuId = [string]$summary.VisibleIds
        $script:LocalVisibleGpuIds = [string]$summary.VisibleIds
        $script:LocalAcceleratorReason = "selected_profile_all_gpus:$ModelProfileId"
    } else {
        $script:LocalGpuId = [string]$summary.BestDeviceId
        $script:LocalVisibleGpuIds = [string]$summary.BestDeviceId
        $script:LocalAcceleratorReason = "selected_profile_single_gpu:$ModelProfileId"
    }
}

function Detect-Accelerator {
    $script:GpuInventory = @()
    $script:GpuInventory += @(Get-NvidiaGpus)
    $script:GpuInventory += @(Get-AmdGpus)
    $script:GpuVendorSummaries = Build-GpuVendorSummaries -Gpus $script:GpuInventory
    Resolve-ModelSelection
    Resolve-AcceleratorMode
}

function Set-RuntimeEnvironment {
    $modelFileValue = if (-not [string]::IsNullOrWhiteSpace($ModelMergeOutputFile)) { $ModelMergeOutputFile } else { $ModelFile }
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
    $env:LOCAL_OLLAMA_MODEL_REPO = $ModelRepo
    $env:LOCAL_OLLAMA_MODEL_FILE = $modelFileValue
    $env:LOCAL_OLLAMA_MODEL_FILES = ($ModelFiles -join ';')
    $env:LOCAL_OLLAMA_MODEL_NAME = $ModelName
    $env:LOCAL_OLLAMA_PROFILE_ID = $ModelProfileId
    $env:LOCAL_OLLAMA_MIN_GPU_VRAM_MB = [string]$MinGpuVramMb
    $env:LOCAL_OLLAMA_CPU_NUM_CTX = [string]$CpuNumCtx
    $env:LOCAL_OLLAMA_GPU_NUM_CTX = [string]$GpuNumCtx

    $env:TEST25_LOCAL_ACCELERATOR = $script:LocalAccelerator
    $env:TEST25_LOCAL_GPU_VENDOR = $script:LocalGpuVendor
    $env:TEST25_LOCAL_GPU_ID = $script:LocalGpuId
    $env:TEST25_LOCAL_GPU_VRAM_MB = [string]$script:LocalGpuVramMb
    $env:TEST25_LOCAL_GPU_COUNT = [string]$script:LocalGpuCount
    $env:TEST25_LOCAL_TOTAL_GPU_VRAM_MB = [string]$script:LocalTotalGpuVramMb
    $env:TEST25_LOCAL_VISIBLE_GPU_IDS = $script:LocalVisibleGpuIds
    $env:TEST25_LOCAL_ACCELERATOR_REASON = $script:LocalAcceleratorReason

    if ($script:LocalAccelerator -eq 'gpu') {
        if ($script:LocalGpuVendor -eq 'nvidia') {
            $env:CUDA_VISIBLE_DEVICES = $script:LocalVisibleGpuIds
            Remove-Item Env:ROCR_VISIBLE_DEVICES, Env:HIP_VISIBLE_DEVICES, Env:GPU_DEVICE_ORDINAL -ErrorAction SilentlyContinue
        } elseif ($script:LocalGpuVendor -eq 'amd') {
            $env:ROCR_VISIBLE_DEVICES = $script:LocalVisibleGpuIds
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
    $modelFileValue = if (-not [string]::IsNullOrWhiteSpace($ModelMergeOutputFile)) { $ModelMergeOutputFile } else { $ModelFile }
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
        ('$env:LOCAL_OLLAMA_MODEL_REPO = ''' + $ModelRepo + '''')
        ('$env:LOCAL_OLLAMA_MODEL_FILE = ''' + $modelFileValue + '''')
        ('$env:LOCAL_OLLAMA_MODEL_FILES = ''' + ($ModelFiles -join ';') + '''')
        ('$env:LOCAL_OLLAMA_MODEL_NAME = ''' + $ModelName + '''')
        ('$env:LOCAL_OLLAMA_PROFILE_ID = ''' + $ModelProfileId + '''')
        ('$env:LOCAL_OLLAMA_MIN_GPU_VRAM_MB = ''' + $MinGpuVramMb + '''')
        ('$env:LOCAL_OLLAMA_CPU_NUM_CTX = ''' + $CpuNumCtx + '''')
        ('$env:LOCAL_OLLAMA_GPU_NUM_CTX = ''' + $GpuNumCtx + '''')
        ('$env:TEST25_LOCAL_ACCELERATOR = ''' + $script:LocalAccelerator + '''')
        ('$env:TEST25_LOCAL_GPU_VENDOR = ''' + $script:LocalGpuVendor + '''')
        ('$env:TEST25_LOCAL_GPU_ID = ''' + $script:LocalGpuId + '''')
        ('$env:TEST25_LOCAL_GPU_VRAM_MB = ''' + $script:LocalGpuVramMb + '''')
        ('$env:TEST25_LOCAL_GPU_COUNT = ''' + $script:LocalGpuCount + '''')
        ('$env:TEST25_LOCAL_TOTAL_GPU_VRAM_MB = ''' + $script:LocalTotalGpuVramMb + '''')
        ('$env:TEST25_LOCAL_VISIBLE_GPU_IDS = ''' + $script:LocalVisibleGpuIds + '''')
        ('$env:TEST25_LOCAL_ACCELERATOR_REASON = ''' + $script:LocalAcceleratorReason + '''')
    )
    if ($script:LocalAccelerator -eq 'gpu' -and $script:LocalGpuVendor -eq 'nvidia') {
        $lines += ('$env:CUDA_VISIBLE_DEVICES = ''' + $script:LocalVisibleGpuIds + '''')
    } elseif ($script:LocalAccelerator -eq 'gpu' -and $script:LocalGpuVendor -eq 'amd') {
        $lines += ('$env:ROCR_VISIBLE_DEVICES = ''' + $script:LocalVisibleGpuIds + '''')
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
    Write-Msg "Model profile:     $ModelProfileId"
    Write-Msg "Profile size (GB): $ModelProfilePublishedSizeGb"
    Write-Msg "Selection reason:  $ModelSelectionReason"
    Write-Msg "Model repo:        $ModelRepo"
    Write-Msg "Model file:        $(if ($ModelFile) { $ModelFile } else { '<multi-file>' })"
    Write-Msg "Model files:       $(if ($ModelFiles.Count -gt 1) { $ModelFiles -join '; ' } else { '<single-file>' })"
    Write-Msg "Model import path: $ModelImportPath"
    Write-Msg "Model alias:       $ModelName"
    Write-Msg "Accelerator mode: $($script:LocalAccelerator)"
    Write-Msg "GPU vendor:       $($script:LocalGpuVendor)"
    Write-Msg "Visible GPU ids:  $(if ($script:LocalVisibleGpuIds) { $script:LocalVisibleGpuIds } else { '<none>' })"
    Write-Msg "GPU id:           $(if ($script:LocalGpuId) { $script:LocalGpuId } else { '<none>' })"
    Write-Msg "GPU count:        $($script:LocalGpuCount)"
    Write-Msg "Best GPU VRAM MB: $($script:LocalGpuVramMb)"
    Write-Msg "Total GPU VRAM MB:$($script:LocalTotalGpuVramMb)"
    Write-Msg "Min GPU VRAM MB:  $MinGpuVramMb"
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

function Get-GgufSplitCommand {
    $override = [Environment]::GetEnvironmentVariable('LOCAL_OLLAMA_GGUF_SPLIT_PATH')
    if (-not [string]::IsNullOrWhiteSpace($override) -and (Test-Path $override)) {
        return $override
    }
    foreach ($candidate in @('gguf-split.exe', 'gguf-split')) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
    }
    return $null
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
    $hfCli = Get-HfCliPath
    $downloadList = @($ModelFiles)
    if ($downloadList.Count -eq 0 -and -not [string]::IsNullOrWhiteSpace($ModelFile)) {
        $downloadList = @($ModelFile)
    }
    if ($downloadList.Count -eq 0) {
        Throw-Fail 'No GGUF files were configured for the selected model profile.'
    }

    $missing = @()
    foreach ($entry in $downloadList) {
        $target = Join-Path $ModelDir $entry
        if (-not (Test-Path $target)) {
            $missing += $entry
        }
    }
    if ($missing.Count -eq 0) {
        Write-Msg "GGUF source files already present for $ModelName"
        return
    }

    Write-Msg 'Downloading GGUF from Hugging Face...'
    foreach ($entry in $missing) {
        & $hfCli download $ModelRepo $entry --local-dir $ModelDir
    }
}

function Ensure-MergedModelFile {
    if ([string]::IsNullOrWhiteSpace($ModelMergeOutputFile)) {
        return
    }
    $mergedTarget = Join-Path $ModelDir $ModelMergeOutputFile
    if (Test-Path $mergedTarget) {
        Write-Msg "Merged GGUF already present: $mergedTarget"
        return
    }
    if ($ModelFiles.Count -eq 0) {
        Throw-Fail "Profile $ModelProfileId requires sharded GGUF inputs but no model_files were configured."
    }

    $firstShard = Join-Path $ModelDir $ModelFiles[0]
    if (-not (Test-Path $firstShard)) {
        Throw-Fail "First GGUF shard missing: $firstShard"
    }
    $ggufSplit = Get-GgufSplitCommand
    if (-not $ggufSplit) {
        Throw-Fail "Profile $ModelProfileId requires GGUF shard merging. Install gguf-split or set LOCAL_OLLAMA_GGUF_SPLIT_PATH."
    }
    Write-Msg "Merging GGUF shards into $mergedTarget ..."
    & $ggufSplit --merge $firstShard $mergedTarget
}

function Write-Modelfile {
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    @"
FROM $ModelImportPath
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
    Ensure-MergedModelFile
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
            Write-Msg "WARNING: CPU-only mode with ${memGb} GB RAM may be rough for $ModelFile."
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
        --model $ModelName `
        --enable-llm-evolution `
        --until-step 130 `
        @ExtraArgs
}
