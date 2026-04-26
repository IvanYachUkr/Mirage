#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUESTED_TARGET="${1:-help}"

is_continue_like_target() {
  case "$REQUESTED_TARGET" in
    profile-continue|continue-profile|candidate20k-continue|\
    profile-continue-export|continue-profile-export|candidate20k-continue-export|\
    resume-profile-step100|resume20k-step100)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

ACTIVE_LOCAL_PROVIDER_ENV="$BASE_DIR/reports/active_local_provider.env"
if is_continue_like_target && [[ "${DATA_SYS_IGNORE_SAVED_LOCAL_ENV:-0}" != "1" && -f "$ACTIVE_LOCAL_PROVIDER_ENV" ]]; then
  # Load this before resolving RUN_PROFILE so continue/resume uses the same
  # profile, endpoint, and model that created the interrupted run.
  # shellcheck disable=SC1090
  source "$ACTIVE_LOCAL_PROVIDER_ENV"
fi

if [[ -z "${PYTHON_BIN:-}" && -x "$BASE_DIR/.venv-linux/bin/python" ]]; then
  PYTHON_BIN="$BASE_DIR/.venv-linux/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

RUN_PROFILE="${RUN_PROFILE:-${PROFILE:-candidate20k}}"
PROFILE_FILE="${PROFILE_FILE:-}"
if [[ -z "$PROFILE_FILE" && -f "$BASE_DIR/local_run_profiles/$RUN_PROFILE.env" ]]; then
  PROFILE_FILE="$BASE_DIR/local_run_profiles/$RUN_PROFILE.env"
fi
if [[ -n "$PROFILE_FILE" ]]; then
  if [[ ! -f "$PROFILE_FILE" ]]; then
    echo "Profile file does not exist: $PROFILE_FILE" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "$PROFILE_FILE"
  set +a
fi

source_saved_local_env_if_needed() {
  local active_env="$ACTIVE_LOCAL_PROVIDER_ENV"
  if is_continue_like_target && [[ "${DATA_SYS_IGNORE_SAVED_LOCAL_ENV:-0}" != "1" && -f "$active_env" ]]; then
    # Continue/resume should default to the provider that created this run
    # instead of inheriting a stale shell-level Gemini/API setting.
    # shellcheck disable=SC1090
    source "$active_env"
  fi

  local provider="${LLM_PROVIDER:-local}"
  case "$provider" in
    local|ollama|vllm|openai|tgi|litellm)
      ;;
    *)
      return 0
      ;;
  esac

  if [[ -n "${LOCAL_LLM_MODEL:-}" && -n "${LOCAL_LLM_URL:-}" ]]; then
    return 0
  fi

  local env_file
  for env_file in \
    "$active_env" \
    "$BASE_DIR/local_ollama_qwen35/runtime_env.sh" \
    "$BASE_DIR/reports/ollama_detected.env"; do
    if [[ -f "$env_file" ]]; then
      # shellcheck disable=SC1090
      source "$env_file"
    fi
    if [[ -n "${LOCAL_LLM_MODEL:-}" && -n "${LOCAL_LLM_URL:-}" ]]; then
      break
    fi
  done
}

source_saved_local_env_if_needed

export LLM_PROVIDER="${LLM_PROVIDER:-local}"
export LOCAL_LLM_URL="${LOCAL_LLM_URL:-http://127.0.0.1:8000/v1}"
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-}"
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-not-needed}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-$BASE_DIR/benchmark_candidate_profile.json}"
export DATA_SYS_PIPELINE_CONFIG="${DATA_SYS_PIPELINE_CONFIG:-$PIPELINE_CONFIG}"
export DATA_SYS_LLM_TIMEOUT_SEC="${DATA_SYS_LLM_TIMEOUT_SEC:-180}"
export DATA_SYS_LLM_MAX_ATTEMPTS="${DATA_SYS_LLM_MAX_ATTEMPTS:-3}"
export DATA_SYS_LLM_BASE_DELAY_SEC="${DATA_SYS_LLM_BASE_DELAY_SEC:-2}"
export DATA_SYS_LLM_MAX_DELAY_SEC="${DATA_SYS_LLM_MAX_DELAY_SEC:-20}"
export DATA_SYS_LLM_USAGE_LOG="${DATA_SYS_LLM_USAGE_LOG:-$BASE_DIR/reports/local_llm_usage.jsonl}"
export DATA_SYS_LOCAL_CHECK_MAX_TOKENS="${DATA_SYS_LOCAL_CHECK_MAX_TOKENS:-768}"
export DATA_SYS_OLLAMA_NATIVE="${DATA_SYS_OLLAMA_NATIVE:-1}"
export DATA_SYS_OLLAMA_THINK="${DATA_SYS_OLLAMA_THINK:-0}"
LAB_ENABLE_SPEED_AUDIT="${LAB_ENABLE_SPEED_AUDIT:-1}"
LAB_ENABLE_MEMORY_AUDIT="${LAB_ENABLE_MEMORY_AUDIT:-0}"
export DATA_SYS_SPEED_AUDIT="${DATA_SYS_SPEED_AUDIT:-$LAB_ENABLE_SPEED_AUDIT}"
export DATA_SYS_MEMORY_AUDIT="${DATA_SYS_MEMORY_AUDIT:-$LAB_ENABLE_MEMORY_AUDIT}"
export DATA_SYS_SPEED_AUDIT_EXPERIMENT="${DATA_SYS_SPEED_AUDIT_EXPERIMENT:-${RUN_PROFILE}_step100}"
export DATA_SYS_MEMORY_AUDIT_EXPERIMENT="${DATA_SYS_MEMORY_AUDIT_EXPERIMENT:-${RUN_PROFILE}_step100}"

N_MOVIES="${N_MOVIES:-20000}"
START_YEAR="${START_YEAR:-1950}"
END_YEAR="${END_YEAR:-2025}"
N_PERSONS="${N_PERSONS:-64000}"
N_COMPANIES="${N_COMPANIES:-1400}"
N_KEYWORDS="${N_KEYWORDS:-3200}"
N_CHARACTERS="${N_CHARACTERS:-340000}"
N_TITLES="${N_TITLES:-26000}"
SEED="${SEED:-42}"
SMOKE_N_MOVIES="${SMOKE_N_MOVIES:-250}"
SMOKE_N_PERSONS="${SMOKE_N_PERSONS:-1200}"
SMOKE_N_COMPANIES="${SMOKE_N_COMPANIES:-90}"
SMOKE_N_KEYWORDS="${SMOKE_N_KEYWORDS:-650}"
SMOKE_N_CHARACTERS="${SMOKE_N_CHARACTERS:-1800}"
SMOKE_N_TITLES="${SMOKE_N_TITLES:-400}"
RERANK_BUDGET_MOVIES="${RERANK_BUDGET_MOVIES:-1200}"
KEYWORD_RERANK_BUDGET_MOVIES="${KEYWORD_RERANK_BUDGET_MOVIES:-1200}"
STEP100_DISABLE_LLM="${STEP100_DISABLE_LLM:-0}"
SKIP_DIAGNOSTIC_COLD_EDGES="${SKIP_DIAGNOSTIC_COLD_EDGES:-1}"
FAST_TITLE_TAGLINES="${FAST_TITLE_TAGLINES:-0}"
export DATA_SYS_SKIP_DIAGNOSTIC_COLD_EDGES="$SKIP_DIAGNOSTIC_COLD_EDGES"
export DATA_SYS_FAST_TITLE_TAGLINES="$FAST_TITLE_TAGLINES"

mkdir -p "$BASE_DIR/reports"

write_active_local_provider_env() {
  case "$LLM_PROVIDER" in
    local|ollama|vllm|openai|tgi|litellm)
      ;;
    *)
      return 0
      ;;
  esac
  if [[ -z "${LOCAL_LLM_MODEL:-}" || -z "${LOCAL_LLM_URL:-}" ]]; then
    return 0
  fi
  {
    printf '# Auto-written by run_lab_local_llm.sh so interrupted lab runs can continue with the same local provider.\n'
    printf 'export RUN_PROFILE=%q\n' "$RUN_PROFILE"
    if [[ -n "${PROFILE_FILE:-}" ]]; then
      printf 'export PROFILE_FILE=%q\n' "$PROFILE_FILE"
    fi
    printf 'export LLM_PROVIDER=%q\n' "$LLM_PROVIDER"
    printf 'export LOCAL_LLM_URL=%q\n' "$LOCAL_LLM_URL"
    printf 'export LOCAL_LLM_MODEL=%q\n' "$LOCAL_LLM_MODEL"
    printf 'export LOCAL_LLM_API_KEY=%q\n' "$LOCAL_LLM_API_KEY"
    printf 'export DATA_SYS_OLLAMA_NATIVE=%q\n' "$DATA_SYS_OLLAMA_NATIVE"
    printf 'export DATA_SYS_OLLAMA_THINK=%q\n' "$DATA_SYS_OLLAMA_THINK"
  } > "$BASE_DIR/reports/active_local_provider.env"
}

step100_continue_args() {
  local manifest="$BASE_DIR/_step100_resume/manifest.json"
  if [[ -d "$BASE_DIR/_step100_resume" && ! -f "$manifest" ]]; then
    printf '%s\n' --resume-step100
    return
  fi
  if [[ -f "$manifest" ]]; then
    "$PYTHON_BIN" - "$manifest" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("--resume-step100")
    raise SystemExit
status = str(data.get("status") or "").strip().lower()
if status != "complete":
    print("--resume-step100")
PY
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <target>

Targets:
  help                    Show this message.
  profile-list            List profile files under local_run_profiles/.
  show-config             Print the resolved profile/scale/local-LLM settings.
  check-provider          Verify LOCAL_LLM_URL / LOCAL_LLM_MODEL with one JSON call.
  preflight               Check whether current entities can support the active profile.
  smoke98                 Tiny end-to-end artifact/entity/graph setup through step 98.
  smoke100                Tiny movie-generation run through step 100.
  plumbing-full100-export Provider check + deterministic debug pipeline through step 100/export.
  preflight20k            Check whether current entities can support the configured 20k run.
  profile-graph80         Run graph/world setup only, stopping before movie generation.
  candidate20k-graph80    Alias for profile-graph80.
  profile-full100         Fresh local-LLM run through step 98, preflight, then step 100.
  profile-full100-export  Fresh run through step 100, sanity report, then strict JOB export.
  profile-continue        Continue from .pipeline_checkpoint.json without resetting outputs.
  profile-continue-export Continue, then write sanity report and strict JOB export.
  profile-step100         Run steps 80-100 for the active profile from existing inputs.
  resume-profile-step100  Resume step 100 for the active profile.
  candidate20k-full100    Fresh local-LLM run through step 98, preflight, then step 100.
  candidate20k-step100    Run steps 80-100 for a benchmark-first 20k movie candidate.
  resume20k-step100       Resume step 100 from the last completed year boundary.
  resume-progress         Report in-progress step-100 status from resume shards.
  sanity                  Write step-100 sanity report for the current outputs.
  runtime-report          Summarize progress, resume state, speed audit, and guardrails.
  export-only             Run strict JOB/IMDb export step 130 only.

Key environment variables:
  RUN_PROFILE=$RUN_PROFILE PROFILE_FILE=${PROFILE_FILE:-'(auto)'}
  LOCAL_LLM_URL=$LOCAL_LLM_URL
  LOCAL_LLM_MODEL=$LOCAL_LLM_MODEL
  DATA_SYS_LOCAL_CHECK_MAX_TOKENS=$DATA_SYS_LOCAL_CHECK_MAX_TOKENS
  DATA_SYS_OLLAMA_NATIVE=$DATA_SYS_OLLAMA_NATIVE
  DATA_SYS_OLLAMA_THINK=$DATA_SYS_OLLAMA_THINK
  LAB_ENABLE_SPEED_AUDIT=$LAB_ENABLE_SPEED_AUDIT
  LAB_ENABLE_MEMORY_AUDIT=$LAB_ENABLE_MEMORY_AUDIT
  N_MOVIES=$N_MOVIES START_YEAR=$START_YEAR END_YEAR=$END_YEAR
  N_PERSONS=$N_PERSONS N_CHARACTERS=$N_CHARACTERS
  ALLOW_OVERWRITE=1 is required for targets that would replace existing movie outputs.
EOF
}

show_config() {
  cat <<EOF
Profile:        $RUN_PROFILE
Profile file:   ${PROFILE_FILE:-'(none)'}
Base dir:       $BASE_DIR
Pipeline config:$DATA_SYS_PIPELINE_CONFIG
Provider:       $LLM_PROVIDER
Local URL:      $LOCAL_LLM_URL
Local model:    ${LOCAL_LLM_MODEL:-(not set)}
Movies:         $N_MOVIES
Years:          $START_YEAR -> $END_YEAR
Persons:        $N_PERSONS
Companies:      $N_COMPANIES
Keywords:       $N_KEYWORDS
Characters:     $N_CHARACTERS
Titles:         $N_TITLES
Rerank budget:  $RERANK_BUDGET_MOVIES
KW rerank bdg:  $KEYWORD_RERANK_BUDGET_MOVIES
Step100 LLM off:$STEP100_DISABLE_LLM
Skip cold diag: $SKIP_DIAGNOSTIC_COLD_EDGES
Fast taglines:  $FAST_TITLE_TAGLINES
Speed audit:    $DATA_SYS_SPEED_AUDIT
Memory audit:   $DATA_SYS_MEMORY_AUDIT
EOF
}

pipeline_common_args() {
  printf '%s\n' \
    --n-movies "$N_MOVIES" \
    --start-year "$START_YEAR" \
    --end-year "$END_YEAR" \
    --n-persons "$N_PERSONS" \
    --n-companies "$N_COMPANIES" \
    --n-keywords "$N_KEYWORDS" \
    --n-characters "$N_CHARACTERS" \
    --n-titles "$N_TITLES" \
    --seed "$SEED" \
    --mode research \
    --model "$LOCAL_LLM_MODEL" \
    --bootstrap-model "$LOCAL_LLM_MODEL" \
    --planning-model "$LOCAL_LLM_MODEL" \
    --bulk-artifact-model "$LOCAL_LLM_MODEL"
}

smoke_common_args() {
  printf '%s\n' \
    --n-movies "$SMOKE_N_MOVIES" \
    --start-year "$START_YEAR" \
    --end-year "$END_YEAR" \
    --n-persons "$SMOKE_N_PERSONS" \
    --n-companies "$SMOKE_N_COMPANIES" \
    --n-keywords "$SMOKE_N_KEYWORDS" \
    --n-characters "$SMOKE_N_CHARACTERS" \
    --n-titles "$SMOKE_N_TITLES" \
    --seed "$SEED" \
    --mode research \
    --model "$LOCAL_LLM_MODEL" \
    --bootstrap-model "$LOCAL_LLM_MODEL" \
    --planning-model "$LOCAL_LLM_MODEL" \
    --bulk-artifact-model "$LOCAL_LLM_MODEL"
}

debug_profile_common_args() {
  printf '%s\n' \
    --n-movies "$N_MOVIES" \
    --start-year "$START_YEAR" \
    --end-year "$END_YEAR" \
    --n-persons "$N_PERSONS" \
    --n-companies "$N_COMPANIES" \
    --n-keywords "$N_KEYWORDS" \
    --n-characters "$N_CHARACTERS" \
    --n-titles "$N_TITLES" \
    --seed "$SEED" \
    --mode debug \
    --model "$LOCAL_LLM_MODEL" \
    --bootstrap-model "$LOCAL_LLM_MODEL" \
    --planning-model "$LOCAL_LLM_MODEL" \
    --bulk-artifact-model "$LOCAL_LLM_MODEL"
}

require_overwrite_ok() {
  if [[ -e "$BASE_DIR/movie.csv" || -e "$BASE_DIR/movie.arrow" || -d "$BASE_DIR/_step100_resume" ]]; then
    if [[ "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
      echo "Refusing to replace existing step-100 outputs in $BASE_DIR." >&2
      echo "Set ALLOW_OVERWRITE=1 only when this run directory is intentionally disposable." >&2
      exit 3
    fi
  fi
}

require_llm_model() {
  if [[ -z "${LOCAL_LLM_MODEL:-}" ]]; then
    echo "LOCAL_LLM_MODEL is required for provider '$LLM_PROVIDER'." >&2
    echo "Set it to the model name exposed by your local server, or use a profile that sets it." >&2
    exit 2
  fi
}

run_preflight() {
  "$PYTHON_BIN" "$BASE_DIR/step100_structural_preflight.py" \
    --base-dir "$BASE_DIR" \
    --n-movies "$N_MOVIES" \
    --target-reuse-ratio 15 \
    --min-unique-cast-people "$((N_MOVIES * 7 / 10))"
}

write_runtime_report() {
  "$PYTHON_BIN" "$BASE_DIR/summarize_step100_runtime.py" \
    --base-dir "$BASE_DIR" \
    --out-dir "$BASE_DIR/reports"
}

step100_llm_args() {
  if [[ "$STEP100_DISABLE_LLM" == "1" ]]; then
    printf '%s\n' \
      --disable-llm-evolution \
      --disable-llm-critic \
      --disable-llm-world-policy \
      --disable-llm-concept-packs \
      --disable-llm-year-slates \
      --disable-llm-keyword-motifs \
      --disable-llm-rerank \
      --disable-llm-keyword-rerank
  fi
}

target="$REQUESTED_TARGET"
case "$target" in
  help|--help|-h)
    usage
    ;;

  profile-list)
    if [[ -d "$BASE_DIR/local_run_profiles" ]]; then
      find "$BASE_DIR/local_run_profiles" -maxdepth 1 -type f -name '*.env' -printf '%f\n' | sort
    else
      echo "No local_run_profiles directory found."
    fi
    ;;

  show-config)
    show_config
    ;;

  check-provider)
    require_llm_model
    write_active_local_provider_env
    "$PYTHON_BIN" "$BASE_DIR/check_local_llm.py" \
      --base-dir "$BASE_DIR" \
      --model "$LOCAL_LLM_MODEL" \
      --timeout-sec "$DATA_SYS_LLM_TIMEOUT_SEC" \
      --max-tokens "$DATA_SYS_LOCAL_CHECK_MAX_TOKENS" \
      --max-attempts 1
    ;;

  smoke98)
    require_llm_model
    write_active_local_provider_env
    require_overwrite_ok
    mapfile -t args < <(smoke_common_args)
    "$PYTHON_BIN" "$BASE_DIR/run_pipeline.py" "${args[@]}" \
      --fresh \
      --force-legacy-graph \
      --until-step 98
    ;;

  smoke100)
    require_llm_model
    write_active_local_provider_env
    require_overwrite_ok
    mapfile -t args < <(smoke_common_args)
    mapfile -t llm_args < <(step100_llm_args)
    "$PYTHON_BIN" "$BASE_DIR/run_pipeline.py" "${args[@]}" \
      "${llm_args[@]}" \
      --fresh-preserve-entities \
      --force-legacy-graph \
      --benchmark-mode \
      --reset-step100-resume \
      --rerank-budget-movies 50 \
      --keyword-rerank-budget-movies 50 \
      --until-step 100
    ;;

  plumbing-full100-export)
    require_overwrite_ok
    mapfile -t args < <(debug_profile_common_args)
    DATA_SYS_DETERMINISTIC_ENRICH=1 \
    DATA_SYS_DETERMINISTIC_LATENTS=1 \
    "$PYTHON_BIN" "$BASE_DIR/run_pipeline.py" "${args[@]}" \
      --fresh \
      --force-legacy-graph \
      --benchmark-mode \
      --reset-step100-resume \
      --disable-llm-evolution \
      --disable-llm-critic \
      --disable-llm-world-policy \
      --disable-llm-concept-packs \
      --disable-llm-year-slates \
      --disable-llm-keyword-motifs \
      --disable-llm-rerank \
      --disable-llm-keyword-rerank \
      --until-step 100
    "$PYTHON_BIN" "$BASE_DIR/step100_sanity_report.py" --base-dir "$BASE_DIR"
    write_runtime_report
    "$PYTHON_BIN" "$BASE_DIR/run_pipeline.py" "${args[@]}" --only 130 --force
    ;;

  preflight|preflight20k)
    run_preflight
    ;;

  profile-graph80|candidate20k-graph80)
    require_llm_model
    write_active_local_provider_env
    run_preflight
    mapfile -t args < <(pipeline_common_args)
    "$PYTHON_BIN" "$BASE_DIR/run_pipeline.py" "${args[@]}" \
      --fresh-preserve-entities \
      --force-legacy-graph \
      --benchmark-mode \
      --from-step 80 \
      --until-step 80 \
      --force
    ;;

  profile-step100|candidate20k-step100)
    require_llm_model
    write_active_local_provider_env
    require_overwrite_ok
    run_preflight
    mapfile -t args < <(pipeline_common_args)
    mapfile -t llm_args < <(step100_llm_args)
    "$PYTHON_BIN" "$BASE_DIR/run_pipeline.py" "${args[@]}" \
      "${llm_args[@]}" \
      --fresh-preserve-entities \
      --force-legacy-graph \
      --benchmark-mode \
      --reset-step100-resume \
      --rerank-budget-movies "$RERANK_BUDGET_MOVIES" \
      --keyword-rerank-budget-movies "$KEYWORD_RERANK_BUDGET_MOVIES" \
      --from-step 80 \
      --until-step 100
    ;;

  profile-full100|candidate20k-full100)
    require_llm_model
    write_active_local_provider_env
    require_overwrite_ok
    mapfile -t args < <(pipeline_common_args)
    "$PYTHON_BIN" "$BASE_DIR/run_pipeline.py" "${args[@]}" \
      --fresh \
      --force-legacy-graph \
      --until-step 98
    run_preflight
    mapfile -t llm_args < <(step100_llm_args)
    "$PYTHON_BIN" "$BASE_DIR/run_pipeline.py" "${args[@]}" \
      "${llm_args[@]}" \
      --force-legacy-graph \
      --benchmark-mode \
      --reset-step100-resume \
      --rerank-budget-movies "$RERANK_BUDGET_MOVIES" \
      --keyword-rerank-budget-movies "$KEYWORD_RERANK_BUDGET_MOVIES" \
      --from-step 100 \
      --until-step 100 \
      --force
    ;;

  profile-full100-export|candidate20k-full100-export)
    "$0" profile-full100
    "$0" sanity
    "$0" export-only
    ;;

  profile-continue|continue-profile|candidate20k-continue)
    require_llm_model
    write_active_local_provider_env
    mapfile -t args < <(pipeline_common_args)
    mapfile -t llm_args < <(step100_llm_args)
    mapfile -t resume_args < <(step100_continue_args)
    "$PYTHON_BIN" "$BASE_DIR/run_pipeline.py" "${args[@]}" \
      "${llm_args[@]}" \
      "${resume_args[@]}" \
      --force-legacy-graph \
      --benchmark-mode \
      --rerank-budget-movies "$RERANK_BUDGET_MOVIES" \
      --keyword-rerank-budget-movies "$KEYWORD_RERANK_BUDGET_MOVIES" \
      --from-step "${FROM_STEP:-4}" \
      --until-step "${UNTIL_STEP:-100}"
    ;;

  profile-continue-export|continue-profile-export|candidate20k-continue-export)
    "$0" profile-continue
    "$0" sanity
    "$0" export-only
    ;;

  resume-profile-step100|resume20k-step100)
    require_llm_model
    write_active_local_provider_env
    mapfile -t args < <(pipeline_common_args)
    mapfile -t llm_args < <(step100_llm_args)
    "$PYTHON_BIN" "$BASE_DIR/run_pipeline.py" "${args[@]}" \
      "${llm_args[@]}" \
      --force-legacy-graph \
      --benchmark-mode \
      --resume-step100 \
      --rerank-budget-movies "$RERANK_BUDGET_MOVIES" \
      --keyword-rerank-budget-movies "$KEYWORD_RERANK_BUDGET_MOVIES" \
      --from-step 100 \
      --until-step 100 \
      --force
    ;;

  resume-progress)
    "$PYTHON_BIN" "$BASE_DIR/step100_resume_progress_report.py" --base-dir "$BASE_DIR"
    ;;

  sanity)
    "$PYTHON_BIN" "$BASE_DIR/step100_sanity_report.py" --base-dir "$BASE_DIR"
    write_runtime_report
    ;;

  runtime-report)
    write_runtime_report
    ;;

  export-only)
    require_llm_model
    write_active_local_provider_env
    mapfile -t args < <(pipeline_common_args)
    "$PYTHON_BIN" "$BASE_DIR/run_pipeline.py" "${args[@]}" --only 130 --force
    ;;

  *)
    echo "Unknown target: $target" >&2
    usage >&2
    exit 2
    ;;
esac
