#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage:
  ./test30/local_ollama_qwen35/run_test30_local_linux.sh <target> [extra run_pipeline args...]

Targets:
  smoke30        Fresh research smoke run through step 30
  smoke60        Fresh research smoke run through step 60
  smoke98        Fresh research smoke run through step 98
  dense100       Fresh dense sign-off style run (100 movies, 2064-2068, through 130)
  historical100  Fresh historical validation style run (100 movies, 1950-2025, through 130)
  scale20k98     Fresh 20k scale validation through step 98

Examples:
  ./test30/local_ollama_qwen35/run_test30_local_linux.sh dense100
  LOCAL_OLLAMA_PROFILE_ID=qwen35_35b_a3b_q4_k_m ./test30/local_ollama_qwen35/run_test30_local_linux.sh smoke98
  ./test30/local_ollama_qwen35/run_test30_local_linux.sh scale20k98 --disable-llm-critic
EOF
}

run_test30_pipeline() {
  local target="$1"
  shift
  local python_bin="${VENV_DIR}/bin/python"
  [[ -x "${python_bin}" ]] || die "Python venv not found at ${python_bin}. Run install/bootstrap first."

  local -a args=(
    --fresh
    --mode research
    --model "${MODEL_NAME}"
  )

  case "${target}" in
    smoke30)
      args+=(
        --n-movies 100
        --start-year 2064
        --end-year 2068
        --until-step 30
      )
      ;;
    smoke60)
      args+=(
        --n-movies 100
        --start-year 2064
        --end-year 2068
        --until-step 60
      )
      ;;
    smoke98)
      args+=(
        --n-movies 100
        --start-year 2064
        --end-year 2068
        --until-step 98
      )
      ;;
    dense100)
      args+=(
        --n-movies 100
        --start-year 2064
        --end-year 2068
        --until-step 130
      )
      ;;
    historical100)
      args+=(
        --n-movies 100
        --start-year 1950
        --end-year 2025
        --until-step 130
      )
      ;;
    scale20k98)
      args+=(
        --n-movies 20000
        --start-year 1950
        --end-year 2025
        --until-step 98
      )
      ;;
    -h|--help|help)
      usage
      return 0
      ;;
    *)
      die "Unknown target '${target}'. Run with --help for the supported targets."
      ;;
  esac

  msg "Launching test30 target '${target}' with local model '${MODEL_NAME}'..."
  "${python_bin}" "${PROJECT_DIR}/run_pipeline.py" "${args[@]}" "$@"
}

main() {
  local target="${1:-dense100}"
  if [[ $# -gt 0 ]]; then
    shift
  fi

  require_linux
  ensure_apt_packages
  ensure_python_venv
  ensure_ollama_installed
  detect_accelerator
  export_runtime_env
  write_runtime_env
  print_accelerator_summary
  warn_if_cpu_only_memory_is_tight

  ensure_ollama_server
  ensure_model_file
  msg "About to import/register the Ollama model alias if needed..."
  ensure_ollama_model
  msg "Finished model import/register step."
  msg "About to warm the Ollama model..."
  warm_model
  msg "Finished warmup step."
  show_runtime_status

  run_test30_pipeline "${target}" "$@"
}

main "$@"
