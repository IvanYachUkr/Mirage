#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-start}"

usage() {
  cat <<'EOF'
Usage:
  ./lab.sh [start|continue|help]

This is the one-command lab entrypoint.

Common wrappers:
  ./lab_smoke_100.sh start       # fresh 100-movie smoke, exports strict IMDb
  ./lab_smoke_100.sh continue    # continue/resume that smoke run
  ./lab_20k.sh start             # fresh 20k candidate from scratch
  ./lab_20k.sh continue          # continue/resume the 20k candidate
  ./lab_200k.sh start            # fresh 200k lab-scale run from scratch
  ./lab_200k.sh continue         # continue/resume the 200k run

Advanced environment:
  LAB_PROFILE=candidate50k ./lab.sh start
  LAB_RUN_DIR=../my_run ./lab_20k.sh start
  LAB_START_TARGET=profile-full100-export ./lab_20k.sh start

Provider setup:
  If LOCAL_LLM_URL and LOCAL_LLM_MODEL are set, they are used directly.
  Otherwise the script tries to detect a reachable Ollama endpoint.
  Set LAB_AUTO_INSTALL_OLLAMA=1 to let the script run the bundled Ollama
  installer when no local endpoint is reachable.
EOF
}

case "$MODE" in
  start|--start|"")
    MODE="start"
    ;;
  continue|--continue|resume|--resume)
    MODE="continue"
    ;;
  help|--help|-h)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

profile_from_scale() {
  case "${LAB_SCALE:-}" in
    "" ) return 1 ;;
    10|tiny|tiny10) printf '%s\n' tiny10_1y ;;
    100|smoke|small|small100) printf '%s\n' small100_2y ;;
    1000|1k|check|check1000) printf '%s\n' check1000_5y ;;
    20000|20k) printf '%s\n' candidate20k ;;
    50000|50k) printf '%s\n' candidate50k ;;
    100000|100k) printf '%s\n' candidate100k ;;
    200000|200k) printf '%s\n' candidate200k ;;
    *)
      echo "Unknown LAB_SCALE: $LAB_SCALE" >&2
      exit 2
      ;;
  esac
}

default_run_dir_for_profile() {
  case "$1" in
    tiny10_1y) printf '%s\n' ../lab_smoke_10 ;;
    small100_2y) printf '%s\n' ../lab_smoke_100 ;;
    check1000_5y) printf '%s\n' ../lab_check_1000 ;;
    candidate20k) printf '%s\n' ../lab_candidate_20k ;;
    candidate50k) printf '%s\n' ../lab_candidate_50k ;;
    candidate100k) printf '%s\n' ../lab_candidate_100k ;;
    candidate200k) printf '%s\n' ../lab_candidate_200k ;;
    *) printf '%s\n' "../lab_${1}" ;;
  esac
}

default_start_target_for_profile() {
  case "$1" in
    tiny10_1y|small100_2y)
      printf '%s\n' profile-full100-export
      ;;
    *)
      printf '%s\n' profile-full100
      ;;
  esac
}

default_export_after_continue_for_profile() {
  case "$1" in
    tiny10_1y|small100_2y)
      printf '%s\n' 1
      ;;
    *)
      printf '%s\n' 0
      ;;
  esac
}

if [[ -z "${LAB_PROFILE:-}" && -n "${LAB_SCALE:-}" ]]; then
  LAB_PROFILE="$(profile_from_scale)"
fi
LAB_PROFILE="${LAB_PROFILE:-${RUN_PROFILE:-small100_2y}}"
LAB_RUN_DIR="${LAB_RUN_DIR:-$(default_run_dir_for_profile "$LAB_PROFILE")}"
LAB_START_TARGET="${LAB_START_TARGET:-$(default_start_target_for_profile "$LAB_PROFILE")}"
LAB_EXPORT_AFTER_CONTINUE="${LAB_EXPORT_AFTER_CONTINUE:-$(default_export_after_continue_for_profile "$LAB_PROFILE")}"
LAB_SANITY_AFTER_CONTINUE="${LAB_SANITY_AFTER_CONTINUE:-1}"
LAB_AUTO_INSTALL_OLLAMA="${LAB_AUTO_INSTALL_OLLAMA:-0}"
LOCAL_OLLAMA_PROFILE_ID="${LOCAL_OLLAMA_PROFILE_ID:-qwen36_35b_a3b_mxfp4_moe}"

abs_path_maybe_missing() {
  local path="$1"
  local parent
  parent="$(dirname "$path")"
  local name
  name="$(basename "$path")"
  mkdir -p "$parent"
  parent="$(cd "$parent" && pwd)"
  printf '%s/%s\n' "$parent" "$name"
}

run_dir_abs="$(abs_path_maybe_missing "$LAB_RUN_DIR")"

detect_or_install_local_llm() {
  local work_dir="$1"
  cd "$work_dir"

  if [[ -n "${LOCAL_LLM_URL:-}" && -n "${LOCAL_LLM_MODEL:-}" ]]; then
    return 0
  fi

  if ./detect_ollama_endpoint.sh; then
    # shellcheck disable=SC1091
    source "$work_dir/reports/ollama_detected.env"
    export LLM_PROVIDER LOCAL_LLM_URL LOCAL_LLM_MODEL LOCAL_LLM_API_KEY
    return 0
  fi

  if [[ "$LAB_AUTO_INSTALL_OLLAMA" == "1" ]]; then
    echo "No local endpoint detected; running Ollama installer profile $LOCAL_OLLAMA_PROFILE_ID."
    LOCAL_OLLAMA_PROFILE_ID="$LOCAL_OLLAMA_PROFILE_ID" ./local_ollama_qwen35/install_all.sh
    ./detect_ollama_endpoint.sh
    # shellcheck disable=SC1091
    source "$work_dir/reports/ollama_detected.env"
    export LLM_PROVIDER LOCAL_LLM_URL LOCAL_LLM_MODEL LOCAL_LLM_API_KEY
    return 0
  fi

  echo "No usable local LLM endpoint was detected." >&2
  echo "Start vLLM/Ollama or export LOCAL_LLM_URL and LOCAL_LLM_MODEL, then rerun." >&2
  echo "To allow this script to install the bundled Ollama profile, set LAB_AUTO_INSTALL_OLLAMA=1." >&2
  exit 4
}

ensure_run_dir_ready_for_continue() {
  if [[ ! -d "$run_dir_abs" || ! -f "$run_dir_abs/run_lab_local_llm.sh" ]]; then
    echo "Run directory is not initialized: $run_dir_abs" >&2
    echo "Use start first, or set LAB_RUN_DIR to the existing run directory." >&2
    exit 3
  fi

  cd "$run_dir_abs"

  if [[ -f reports/active_local_provider.env ]]; then
    # shellcheck disable=SC1091
    source reports/active_local_provider.env
  fi

  if [[ ! -x .venv-linux/bin/python ]]; then
    echo "Python environment missing; running setup_linux_env.sh."
    ./setup_linux_env.sh
  fi

  detect_or_install_local_llm "$run_dir_abs"
  export RUN_PROFILE="${RUN_PROFILE:-$LAB_PROFILE}"
  export LLM_PROVIDER="${LLM_PROVIDER:-local}"
  export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-not-needed}"

  ./run_lab_local_llm.sh show-config
  ./run_lab_local_llm.sh check-provider
}

resume_manifest_incomplete() {
  [[ -d _step100_resume ]] || return 1
  [[ -f _step100_resume/manifest.json ]] || return 0
  ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"complete"' _step100_resume/manifest.json
}

continue_target_auto() {
  if resume_manifest_incomplete; then
    printf '%s\n' resume-profile-step100
  elif [[ -f .pipeline_checkpoint.json ]]; then
    printf '%s\n' profile-continue
  elif [[ -f movie.arrow || -f movie.csv ]]; then
    printf '%s\n' sanity
  else
    echo "No checkpoint, step-100 resume workspace, or movie output found in $run_dir_abs." >&2
    echo "Nothing obvious to continue. Use start for a fresh run." >&2
    exit 3
  fi
}

start_run() {
  if [[ -e "$run_dir_abs" && -n "$(find "$run_dir_abs" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Run directory already exists and is not empty: $run_dir_abs" >&2
    echo "Use continue, choose a new LAB_RUN_DIR, or deliberately run the lower-level tools." >&2
    exit 3
  fi

  echo "Starting fresh Mirage lab run."
  echo "Profile: $LAB_PROFILE"
  echo "Run dir: $run_dir_abs"
  echo "Target:  $LAB_START_TARGET"

  export RUN_PROFILE="$LAB_PROFILE"
  export LLM_PROVIDER="${LLM_PROVIDER:-local}"
  export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-not-needed}"

  if [[ "$LAB_AUTO_INSTALL_OLLAMA" == "1" ]]; then
    detect_or_install_local_llm "$SCRIPT_DIR"
  fi

  "$SCRIPT_DIR/lab_run_fresh_profile.sh" "$LAB_PROFILE" "$run_dir_abs" "$LAB_START_TARGET"
}

continue_run() {
  echo "Continuing Mirage lab run."
  echo "Profile hint: $LAB_PROFILE"
  echo "Run dir:      $run_dir_abs"

  ensure_run_dir_ready_for_continue

  local target
  target="${LAB_CONTINUE_TARGET:-$(continue_target_auto)}"
  echo "Continue target: $target"
  ./run_lab_local_llm.sh "$target"

  if [[ "$target" != "sanity" && "$LAB_SANITY_AFTER_CONTINUE" == "1" ]]; then
    ./run_lab_local_llm.sh sanity
  fi
  if [[ "$LAB_EXPORT_AFTER_CONTINUE" == "1" ]]; then
    ./run_lab_local_llm.sh export-only
  fi
}

case "$MODE" in
  start) start_run ;;
  continue) continue_run ;;
esac
