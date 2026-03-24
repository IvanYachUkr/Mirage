#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

MODEL_REPO="unsloth/Qwen3.5-35B-A3B-GGUF"
MODEL_FILE="Qwen3.5-35B-A3B-Q4_K_M.gguf"
MODEL_NAME="qwen35-35b-a3b"
MODEL_DIR="${HOME}/models/${MODEL_NAME}"
MODELS_DIR="${HOME}/.ollama/models"
RUNTIME_ENV_FILE="${SCRIPT_DIR}/runtime_env.sh"
VENV_DIR="${PROJECT_DIR}/.venv-local-ollama"
REQ_FILE="${SCRIPT_DIR}/requirements.local_pipeline.txt"

MIN_GPU_VRAM_MB=24000
CPU_NUM_CTX=4096
GPU_NUM_CTX=8192

LOCAL_ACCELERATOR="cpu"
LOCAL_GPU_VENDOR="none"
LOCAL_GPU_ID=""
LOCAL_GPU_VRAM_MB="0"
LOCAL_ACCELERATOR_REASON="not-detected"
LOCAL_NUM_CTX="${CPU_NUM_CTX}"

msg() {
  printf '[test25-local] %s\n' "$*"
}

die() {
  printf '[test25-local][ERROR] %s\n' "$*" >&2
  exit 1
}

require_linux() {
  if [[ "${OSTYPE:-}" != linux* ]]; then
    die "These scripts are intended for Linux bash environments."
  fi
}

detect_accelerator() {
  local detection
  detection="$(python3 - <<'PY'
import re
import shutil
import subprocess
import sys

MIN_MB = 24000

def print_result(mode, vendor, device_id, vram_mb, reason):
    print(f"{mode}|{vendor}|{device_id}|{vram_mb}|{reason}")
    sys.exit(0)

def try_nvidia():
    if not shutil.which("nvidia-smi"):
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total,gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return
    best = None
    for raw in out.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 3:
            continue
        idx, mem_raw, uuid = parts[0], parts[1], parts[2]
        try:
            mem_mb = int(float(mem_raw))
        except Exception:
            continue
        device_id = uuid or idx
        if best is None or mem_mb > best[1]:
            best = (device_id, mem_mb)
    if best is None:
        return
    device_id, mem_mb = best
    if mem_mb >= MIN_MB:
        print_result("gpu", "nvidia", device_id, mem_mb, "eligible_nvidia_gpu")
    print_result("cpu", "nvidia", device_id, mem_mb, "nvidia_gpu_below_24gb")

def try_rocm():
    if not shutil.which("rocm-smi"):
        return
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return
    best = None
    for raw in out.splitlines():
        match = re.search(r"GPU\[(\d+)\].*Total Memory \(B\):\s*([0-9]+)", raw)
        if not match:
            continue
        idx = match.group(1)
        mem_bytes = int(match.group(2))
        mem_mb = mem_bytes // (1024 * 1024)
        if best is None or mem_mb > best[1]:
            best = (idx, mem_mb)
    if best is None:
        return
    device_id, mem_mb = best
    if mem_mb >= MIN_MB:
        print_result("gpu", "amd", device_id, mem_mb, "eligible_amd_gpu")
    print_result("cpu", "amd", device_id, mem_mb, "amd_gpu_below_24gb")

try_nvidia()
try_rocm()
print_result("cpu", "none", "", 0, "no_eligible_gpu_detected")
PY
)"

  IFS='|' read -r LOCAL_ACCELERATOR LOCAL_GPU_VENDOR LOCAL_GPU_ID LOCAL_GPU_VRAM_MB LOCAL_ACCELERATOR_REASON <<< "${detection}"
  if [[ "${LOCAL_ACCELERATOR}" == "gpu" ]]; then
    LOCAL_NUM_CTX="${GPU_NUM_CTX}"
  else
    LOCAL_NUM_CTX="${CPU_NUM_CTX}"
  fi
}

export_runtime_env() {
  export OLLAMA_HOST="127.0.0.1:11434"
  export OLLAMA_NO_CLOUD=1
  export OLLAMA_CONTEXT_LENGTH="${LOCAL_NUM_CTX}"
  export OLLAMA_NUM_PARALLEL=1
  export OLLAMA_MAX_LOADED_MODELS=1
  export OLLAMA_MAX_QUEUE=128

  export LLM_PROVIDER="local"
  export LOCAL_LLM_URL="http://127.0.0.1:11434/v1"
  export LOCAL_LLM_MODEL="${MODEL_NAME}"
  export LOCAL_LLM_API_KEY="not-needed"

  export TEST25_LOCAL_ACCELERATOR="${LOCAL_ACCELERATOR}"
  export TEST25_LOCAL_GPU_VENDOR="${LOCAL_GPU_VENDOR}"
  export TEST25_LOCAL_GPU_ID="${LOCAL_GPU_ID}"
  export TEST25_LOCAL_GPU_VRAM_MB="${LOCAL_GPU_VRAM_MB}"
  export TEST25_LOCAL_ACCELERATOR_REASON="${LOCAL_ACCELERATOR_REASON}"

  if [[ "${LOCAL_ACCELERATOR}" == "gpu" ]]; then
    if [[ "${LOCAL_GPU_VENDOR}" == "nvidia" ]]; then
      export CUDA_VISIBLE_DEVICES="${LOCAL_GPU_ID}"
      unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL GGML_VK_VISIBLE_DEVICES || true
    elif [[ "${LOCAL_GPU_VENDOR}" == "amd" ]]; then
      export ROCR_VISIBLE_DEVICES="${LOCAL_GPU_ID}"
      unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL || true
    fi
  else
    export CUDA_VISIBLE_DEVICES="-1"
    export ROCR_VISIBLE_DEVICES="-1"
    export HIP_VISIBLE_DEVICES="-1"
    export GPU_DEVICE_ORDINAL="-1"
    export GGML_VK_VISIBLE_DEVICES="-1"
  fi
}

write_runtime_env() {
  cat > "${RUNTIME_ENV_FILE}" <<EOF
#!/usr/bin/env bash
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_NO_CLOUD=1
export OLLAMA_CONTEXT_LENGTH="${LOCAL_NUM_CTX}"
export OLLAMA_NUM_PARALLEL="1"
export OLLAMA_MAX_LOADED_MODELS="1"
export OLLAMA_MAX_QUEUE="128"
export LLM_PROVIDER="local"
export LOCAL_LLM_URL="http://127.0.0.1:11434/v1"
export LOCAL_LLM_MODEL="${MODEL_NAME}"
export LOCAL_LLM_API_KEY="not-needed"
export TEST25_LOCAL_ACCELERATOR="${LOCAL_ACCELERATOR}"
export TEST25_LOCAL_GPU_VENDOR="${LOCAL_GPU_VENDOR}"
export TEST25_LOCAL_GPU_ID="${LOCAL_GPU_ID}"
export TEST25_LOCAL_GPU_VRAM_MB="${LOCAL_GPU_VRAM_MB}"
export TEST25_LOCAL_ACCELERATOR_REASON="${LOCAL_ACCELERATOR_REASON}"
EOF

  if [[ "${LOCAL_ACCELERATOR}" == "gpu" ]]; then
    if [[ "${LOCAL_GPU_VENDOR}" == "nvidia" ]]; then
      cat >> "${RUNTIME_ENV_FILE}" <<EOF
export CUDA_VISIBLE_DEVICES="${LOCAL_GPU_ID}"
EOF
    elif [[ "${LOCAL_GPU_VENDOR}" == "amd" ]]; then
      cat >> "${RUNTIME_ENV_FILE}" <<EOF
export ROCR_VISIBLE_DEVICES="${LOCAL_GPU_ID}"
EOF
    fi
  else
    cat >> "${RUNTIME_ENV_FILE}" <<'EOF'
export CUDA_VISIBLE_DEVICES="-1"
export ROCR_VISIBLE_DEVICES="-1"
export HIP_VISIBLE_DEVICES="-1"
export GPU_DEVICE_ORDINAL="-1"
export GGML_VK_VISIBLE_DEVICES="-1"
EOF
  fi
  chmod +x "${RUNTIME_ENV_FILE}" || true
}

print_accelerator_summary() {
  msg "Accelerator mode: ${LOCAL_ACCELERATOR}"
  msg "GPU vendor:       ${LOCAL_GPU_VENDOR}"
  msg "GPU id:           ${LOCAL_GPU_ID:-<none>}"
  msg "GPU VRAM MB:      ${LOCAL_GPU_VRAM_MB}"
  msg "Reason:           ${LOCAL_ACCELERATOR_REASON}"
  msg "num_ctx:          ${LOCAL_NUM_CTX}"
}

ensure_apt_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    die "This installer currently expects apt-get (Ubuntu/Debian-style Linux)."
  fi
  sudo apt-get update
  sudo apt-get install -y \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv
}

ensure_python_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR}/bin/pip" install -r "${REQ_FILE}"
}

ensure_ollama_installed() {
  if command -v ollama >/dev/null 2>&1; then
    msg "Ollama already installed."
    return
  fi
  msg "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
}

ollama_api_ready() {
  curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1
}

ensure_ollama_server() {
  if ollama_api_ready; then
    msg "Ollama API already available on 127.0.0.1:11434."
    return
  fi

  msg "Starting Ollama server..."
  write_runtime_env
  nohup bash -lc "source '${RUNTIME_ENV_FILE}'; ollama serve" > "${LOG_DIR}/ollama_serve.log" 2>&1 &

  local waited=0
  until ollama_api_ready; do
    sleep 1
    waited=$((waited + 1))
    if [[ ${waited} -ge 60 ]]; then
      die "Ollama API did not come up within 60s. Check ${LOG_DIR}/ollama_serve.log"
    fi
  done
  msg "Ollama server is up."
}

ensure_model_file() {
  mkdir -p "${MODEL_DIR}"
  if [[ -f "${MODEL_DIR}/${MODEL_FILE}" ]]; then
    msg "GGUF already present: ${MODEL_DIR}/${MODEL_FILE}"
    return
  fi
  msg "Downloading GGUF from Hugging Face..."
  "${VENV_DIR}/bin/hf" download "${MODEL_REPO}" "${MODEL_FILE}" --local-dir "${MODEL_DIR}"
}

write_modelfile() {
  mkdir -p "${MODEL_DIR}"
  cat > "${MODEL_DIR}/Modelfile" <<EOF
FROM ${MODEL_DIR}/${MODEL_FILE}
PARAMETER num_ctx ${LOCAL_NUM_CTX}
EOF
}

ollama_model_exists() {
  ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "${MODEL_NAME}"
}

ensure_ollama_model() {
  write_modelfile
  if ollama_model_exists; then
    msg "Ollama model already imported: ${MODEL_NAME}"
    return
  fi
  msg "Importing GGUF into Ollama as ${MODEL_NAME}..."
  ollama create "${MODEL_NAME}" -f "${MODEL_DIR}/Modelfile"
}

warm_model() {
  msg "Warming model ${MODEL_NAME}..."
  curl -fsS "http://127.0.0.1:11434/api/generate" \
    -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"ping\",\"stream\":false,\"options\":{\"num_ctx\":128}}" \
    >/dev/null
}

show_runtime_status() {
  msg "Ollama runtime status:"
  ollama ps || true
}

warn_if_cpu_only_memory_is_tight() {
  if [[ "${LOCAL_ACCELERATOR}" != "cpu" ]]; then
    return
  fi
  local mem_gb
  mem_gb="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)"
  if [[ -n "${mem_gb}" && "${mem_gb}" -lt 48 ]]; then
    msg "WARNING: CPU-only mode with ${mem_gb} GB RAM may be rough for this 22 GB GGUF."
  fi
}

run_pipeline_200k() {
  local python_bin="${VENV_DIR}/bin/python"
  [[ -x "${python_bin}" ]] || die "Python venv not found at ${python_bin}. Run install_all.sh first."

  "${python_bin}" "${PROJECT_DIR}/run_pipeline.py" \
    --fresh \
    --n-movies 200000 \
    --start-year 1950 \
    --end-year 2025 \
    --n-titles 200000 \
    --n-persons 450000 \
    --n-companies 30000 \
    --n-keywords 38000 \
    --n-characters 786000 \
    --enable-llm-evolution \
    --until-step 100 \
    "$@"
}
