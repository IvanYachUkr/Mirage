$ErrorActionPreference = "Stop"

$BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProfilePath = Join-Path $BaseDir "local_run_profiles\candidate100k_2000_2050.env"
$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$LogDir = Join-Path $BaseDir "_runner_logs\candidate100k_2000_2050"
$LogFile = Join-Path $LogDir "windows_step100_100k_resume.log"
$PidFile = Join-Path $LogDir "windows_step100_100k_resume.pid"

function Import-RunProfile {
    param([string]$Path)
    foreach ($line in Get-Content $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        if ($parts.Count -ne 2) {
            continue
        }
        [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
    }
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $BaseDir
Import-RunProfile -Path $ProfilePath

$env:RUN_PROFILE = "candidate100k_2000_2050"
$env:LLM_PROVIDER = "gemini"
$env:LOCAL_LLM_MODEL = if ($env:LOCAL_LLM_MODEL) { $env:LOCAL_LLM_MODEL } else { "gemini-3.1-flash-lite-preview" }
$env:PYTHONUNBUFFERED = "1"
$env:DATA_SYS_PIPELINE_CONFIG = Join-Path $BaseDir "v18_config.json"
$env:DATA_SYS_START_YEAR = $env:START_YEAR
$env:DATA_SYS_END_YEAR = $env:END_YEAR
$env:DATA_SYS_LLM_USAGE_LOG = Join-Path $BaseDir "reports\gemini_100k_2000_2050_usage.jsonl"
$env:DATA_SYS_LLM_USAGE_LOG_LATEST = Join-Path $BaseDir "reports\gemini_100k_2000_2050_usage_latest.jsonl"
$env:DATA_SYS_LLM_MAX_ATTEMPTS = "5"
$env:DATA_SYS_LLM_TIMEOUT_SEC = "180"
$env:DATA_SYS_LLM_BASE_DELAY_SEC = "2"
$env:DATA_SYS_LLM_MAX_DELAY_SEC = "60"
$env:DATA_SYS_SKIP_DIAGNOSTIC_COLD_EDGES = "1"
$env:DATA_SYS_FAST_TITLE_TAGLINES = if ($env:FAST_TITLE_TAGLINES) { $env:FAST_TITLE_TAGLINES } else { "1" }
$env:DATA_SYS_NEAR_DUPLICATE_LIMIT = "50000"

$PID | Set-Content -Path $PidFile

@(
    "started windows foreground $PID at $(Get-Date -Format o)"
    "log $LogFile"
    "base $BaseDir"
    "python $PythonExe"
) | Out-File -FilePath $LogFile -Append -Encoding utf8

$Args = @(
    (Join-Path $BaseDir "run_pipeline.py"),
    "--n-movies", $env:N_MOVIES,
    "--start-year", $env:START_YEAR,
    "--end-year", $env:END_YEAR,
    "--n-persons", $env:N_PERSONS,
    "--n-companies", $env:N_COMPANIES,
    "--n-keywords", $env:N_KEYWORDS,
    "--n-characters", $env:N_CHARACTERS,
    "--n-titles", $env:N_TITLES,
    "--seed", "42",
    "--mode", "research",
    "--model", $env:LOCAL_LLM_MODEL,
    "--bootstrap-model", $env:LOCAL_LLM_MODEL,
    "--planning-model", $env:LOCAL_LLM_MODEL,
    "--bulk-artifact-model", $env:LOCAL_LLM_MODEL,
    "--force-scalable-graph",
    "--benchmark-mode",
    "--resume-step100",
    "--rerank-budget-movies", $env:RERANK_BUDGET_MOVIES,
    "--keyword-rerank-budget-movies", $env:KEYWORD_RERANK_BUDGET_MOVIES,
    "--from-step", "100",
    "--until-step", "100",
    "--force"
)

& $PythonExe @Args *>> $LogFile
exit $LASTEXITCODE
