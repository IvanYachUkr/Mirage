#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

PROFILES_FILE="${SCRIPT_DIR}/model_profiles.json"
MODEL_REPO=""
MODEL_FILE=""
MODEL_FILES_JOINED=""
MODEL_MERGE_OUTPUT_FILE=""
MODEL_IMPORT_PATH=""
MODEL_NAME=""
MODEL_DIR=""
MODEL_PROFILE_ID=""
MODEL_PROFILE_PUBLISHED_SIZE_GB="0"
MODEL_SELECTION_REASON=""
MODELS_DIR="${HOME}/.ollama/models"
RUNTIME_ENV_FILE="${SCRIPT_DIR}/runtime_env.sh"
VENV_DIR="${PROJECT_DIR}/.venv-local-ollama"
REQ_FILE="${SCRIPT_DIR}/requirements.local_pipeline.txt"

DEFAULT_MIN_GPU_VRAM_MB=24000
DEFAULT_CPU_NUM_CTX=4096
DEFAULT_GPU_NUM_CTX=8192
MIN_GPU_VRAM_MB="${LOCAL_OLLAMA_MIN_GPU_VRAM_MB:-${DEFAULT_MIN_GPU_VRAM_MB}}"
CPU_NUM_CTX="${LOCAL_OLLAMA_CPU_NUM_CTX:-${DEFAULT_CPU_NUM_CTX}}"
GPU_NUM_CTX="${LOCAL_OLLAMA_GPU_NUM_CTX:-${DEFAULT_GPU_NUM_CTX}}"
PROFILE_ID="${LOCAL_OLLAMA_PROFILE_ID:-}"
DISABLE_AUTO_PROFILE="${LOCAL_OLLAMA_DISABLE_AUTO_PROFILE:-}"

LOCAL_ACCELERATOR="cpu"
LOCAL_GPU_VENDOR="none"
LOCAL_GPU_ID=""
LOCAL_GPU_VRAM_MB="0"
LOCAL_GPU_COUNT="0"
LOCAL_TOTAL_GPU_VRAM_MB="0"
LOCAL_VISIBLE_GPU_IDS=""
LOCAL_ACCELERATOR_REASON="not-detected"
LOCAL_NUM_CTX="${CPU_NUM_CTX}"

msg() {
  printf '[local-ollama] %s\n' "$*"
}

die() {
  printf '[local-ollama][ERROR] %s\n' "$*" >&2
  exit 1
}

require_linux() {
  if [[ "${OSTYPE:-}" != linux* ]]; then
    die "These scripts are intended for Linux bash environments."
  fi
}

detect_accelerator() {
  local detection
  detection="$(PROFILES_FILE="${PROFILES_FILE}" PROFILE_ID="${PROFILE_ID}" DISABLE_AUTO_PROFILE="${DISABLE_AUTO_PROFILE}" CPU_NUM_CTX="${CPU_NUM_CTX}" GPU_NUM_CTX="${GPU_NUM_CTX}" DEFAULT_MIN_GPU_VRAM_MB="${DEFAULT_MIN_GPU_VRAM_MB}" MIN_GPU_VRAM_MB="${MIN_GPU_VRAM_MB}" python3 - <<'PY'
import json
import os
import re
import shutil
import subprocess
import sys

profiles_path = os.environ["PROFILES_FILE"]
profile_id = (os.environ.get("PROFILE_ID") or "").strip()
disable_auto = (os.environ.get("DISABLE_AUTO_PROFILE") or "").strip().lower() in {"1", "true", "yes", "on"}
explicit_model_repo = (os.environ.get("LOCAL_OLLAMA_MODEL_REPO") or "").strip()
explicit_model_file = (os.environ.get("LOCAL_OLLAMA_MODEL_FILE") or "").strip()
explicit_model_name = (os.environ.get("LOCAL_OLLAMA_MODEL_NAME") or "").strip()
has_explicit_model = any([explicit_model_repo, explicit_model_file, explicit_model_name])
cpu_num_ctx = os.environ.get("CPU_NUM_CTX", "4096")
gpu_num_ctx = os.environ.get("GPU_NUM_CTX", "8192")
default_min_mb = int(os.environ.get("DEFAULT_MIN_GPU_VRAM_MB", "24000"))
min_gpu_override = (os.environ.get("MIN_GPU_VRAM_MB") or "").strip()

with open(profiles_path, "r", encoding="utf-8") as fh:
    profiles_doc = json.load(fh)

profiles = sorted(profiles_doc["profiles"], key=lambda p: int(p.get("priority", 0)), reverse=True)

def detect_nvidia():
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total,name", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    items = []
    for raw in out.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 3:
            continue
        idx = parts[0]
        mem_raw = parts[1]
        name = ",".join(parts[2:]).strip()
        try:
            mem_mb = int(float(mem_raw))
        except Exception:
            continue
        items.append({"vendor": "nvidia", "device_id": idx, "vram_mb": mem_mb, "name": name})
    return items

def detect_rocm():
    if not shutil.which("rocm-smi"):
        return []
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    items = []
    for raw in out.splitlines():
        match = re.search(r"GPU\[(\d+)\].*Total Memory \(B\):\s*([0-9]+)", raw)
        if not match:
            continue
        idx = match.group(1)
        mem_bytes = int(match.group(2))
        items.append({"vendor": "amd", "device_id": idx, "vram_mb": mem_bytes // (1024 * 1024), "name": f"AMD GPU {idx}"})
    return items

inventory = detect_nvidia() + detect_rocm()

def build_summaries(items):
    grouped = {}
    for item in items:
        grouped.setdefault(item["vendor"], []).append(item)
    summaries = {}
    for vendor, entries in grouped.items():
        entries = sorted(entries, key=lambda x: (-int(x["vram_mb"]), int(x["device_id"])))
        summaries[vendor] = {
            "vendor": vendor,
            "count": len(entries),
            "total_vram_mb": sum(int(x["vram_mb"]) for x in entries),
            "best_vram_mb": int(entries[0]["vram_mb"]),
            "best_device_id": str(entries[0]["device_id"]),
            "visible_ids": ",".join(str(x["device_id"]) for x in sorted(entries, key=lambda x: int(x["device_id"]))),
            "best_name": entries[0]["name"],
        }
    return summaries

summaries = build_summaries(inventory)

def profile_by_id(profile_id_value):
    for profile in profiles:
        if profile.get("id") == profile_id_value:
            return profile
    return None

def fits(profile, summary):
    if not summary:
        return False
    min_single = int(profile.get("min_single_gpu_vram_mb", 0))
    min_total = int(profile.get("min_total_gpu_vram_mb", 0))
    min_count = int(profile.get("min_gpu_count", 1))
    visibility = profile.get("visibility", "single")
    if summary["count"] < min_count:
        return False
    if summary["best_vram_mb"] < min_single:
        return False
    if visibility == "all" and summary["total_vram_mb"] < min_total:
        return False
    return True

selected = None
selected_vendor = None
selection_reason = ""

if has_explicit_model:
    if not all([explicit_model_repo, explicit_model_file, explicit_model_name]):
        raise SystemExit("manual_model_override_requires_repo_file_name")
    selected = {
        "id": profile_id or "custom",
        "model_repo": explicit_model_repo,
        "model_file": explicit_model_file,
        "model_name": explicit_model_name,
        "published_size_gb": 0,
        "visibility": "single",
    }
    selection_reason = "manual_model_override"
elif profile_id:
    selected = profile_by_id(profile_id)
    if selected is None:
        raise SystemExit(f"unknown_profile_id:{profile_id}")
    selection_reason = f"explicit_profile:{profile_id}"
elif disable_auto:
    selected = profile_by_id(profiles_doc["default_profile_id"])
    selection_reason = f"auto_profile_disabled:{profiles_doc['default_profile_id']}"
else:
    candidates = []
    for profile in profiles:
        for vendor, summary in summaries.items():
            if fits(profile, summary):
                candidates.append((int(profile.get("priority", 0)), int(summary["total_vram_mb"]), int(summary["best_vram_mb"]), profile, vendor))
    if candidates:
        candidates.sort(reverse=True, key=lambda row: (row[0], row[1], row[2]))
        _, _, _, selected, selected_vendor = candidates[0]
        selection_reason = f"auto_profile_fit:{selected['id']}:{selected_vendor}"
    else:
        selected = profile_by_id(profiles_doc["fallback_profile_id"])
        selection_reason = f"auto_profile_fallback:{profiles_doc['fallback_profile_id']}"

if selected is None:
    raise SystemExit("no_profile_selected")

if min_gpu_override:
    min_gpu_vram_mb = int(min_gpu_override)
else:
    min_gpu_vram_mb = int(selected.get("min_single_gpu_vram_mb", default_min_mb))

summary = None
if selected_vendor and selected_vendor in summaries:
    summary = summaries[selected_vendor]
else:
    eligible = [summary for summary in summaries.values() if fits(selected, summary)]
    eligible.sort(key=lambda row: (int(row["total_vram_mb"]), int(row["best_vram_mb"])), reverse=True)
    if eligible:
        summary = eligible[0]
    elif inventory:
        best = sorted(inventory, key=lambda row: (-int(row["vram_mb"]), int(row["device_id"])))[0]
        summary = {
            "vendor": best["vendor"],
            "count": 1,
            "total_vram_mb": int(best["vram_mb"]),
            "best_vram_mb": int(best["vram_mb"]),
            "best_device_id": str(best["device_id"]),
            "visible_ids": str(best["device_id"]),
            "best_name": best["name"],
        }

visibility = selected.get("visibility", "single")
profile_fits = fits(selected, summary) if summary else False
if not summary:
    mode = "cpu"
    vendor = "none"
    gpu_id = ""
    best_vram = 0
    gpu_count = 0
    total_vram = 0
    visible_ids = ""
    accel_reason = "no_eligible_gpu_detected"
    num_ctx = cpu_num_ctx
elif not profile_fits:
    mode = "cpu"
    vendor = summary["vendor"]
    gpu_id = summary["best_device_id"]
    best_vram = summary["best_vram_mb"]
    gpu_count = summary["count"]
    total_vram = summary["total_vram_mb"]
    visible_ids = summary["visible_ids"]
    accel_reason = f"selected_profile_not_fit:{selected.get('id', '')}"
    num_ctx = cpu_num_ctx
else:
    mode = "gpu"
    vendor = summary["vendor"]
    best_vram = summary["best_vram_mb"]
    gpu_count = summary["count"]
    total_vram = summary["total_vram_mb"]
    if visibility == "all" and gpu_count > 1:
        gpu_id = summary["visible_ids"]
        visible_ids = summary["visible_ids"]
        accel_reason = f"selected_profile_all_gpus:{selected.get('id', '')}"
    else:
        gpu_id = summary["best_device_id"]
        visible_ids = summary["best_device_id"]
        accel_reason = f"selected_profile_single_gpu:{selected.get('id', '')}"
    num_ctx = gpu_num_ctx

print("|".join([
    mode,
    vendor,
    gpu_id,
    str(best_vram),
    accel_reason,
    str(num_ctx),
    visible_ids,
    str(gpu_count),
    str(total_vram),
    selected.get("model_repo", ""),
    selected.get("model_file", ""),
    ";".join(selected.get("model_files", [])),
    selected.get("merge_output_file", ""),
    selected.get("model_name", ""),
    selected.get("id", ""),
    str(selected.get("published_size_gb", 0)),
    selection_reason,
    str(min_gpu_vram_mb),
]))
PY
)"

  IFS='|' read -r \
    LOCAL_ACCELERATOR \
    LOCAL_GPU_VENDOR \
    LOCAL_GPU_ID \
    LOCAL_GPU_VRAM_MB \
    LOCAL_ACCELERATOR_REASON \
    LOCAL_NUM_CTX \
    LOCAL_VISIBLE_GPU_IDS \
    LOCAL_GPU_COUNT \
    LOCAL_TOTAL_GPU_VRAM_MB \
    MODEL_REPO \
    MODEL_FILE \
    MODEL_FILES_JOINED \
    MODEL_MERGE_OUTPUT_FILE \
    MODEL_NAME \
    MODEL_PROFILE_ID \
    MODEL_PROFILE_PUBLISHED_SIZE_GB \
    MODEL_SELECTION_REASON \
    MIN_GPU_VRAM_MB <<< "${detection}"

  MODEL_DIR="${HOME}/models/${MODEL_NAME}"
  if [[ -n "${MODEL_MERGE_OUTPUT_FILE}" ]]; then
    MODEL_IMPORT_PATH="${MODEL_DIR}/${MODEL_MERGE_OUTPUT_FILE}"
  elif [[ -n "${MODEL_FILE}" ]]; then
    MODEL_IMPORT_PATH="${MODEL_DIR}/${MODEL_FILE}"
  else
    first_model_file="${MODEL_FILES_JOINED%%;*}"
    MODEL_IMPORT_PATH="${MODEL_DIR}/${first_model_file}"
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
  export LOCAL_OLLAMA_MODEL_REPO="${MODEL_REPO}"
  export LOCAL_OLLAMA_MODEL_FILE="${MODEL_MERGE_OUTPUT_FILE:-${MODEL_FILE}}"
  export LOCAL_OLLAMA_MODEL_FILES="${MODEL_FILES_JOINED}"
  export LOCAL_OLLAMA_MODEL_NAME="${MODEL_NAME}"
  export LOCAL_OLLAMA_PROFILE_ID="${MODEL_PROFILE_ID}"
  export LOCAL_OLLAMA_MIN_GPU_VRAM_MB="${MIN_GPU_VRAM_MB}"
  export LOCAL_OLLAMA_CPU_NUM_CTX="${CPU_NUM_CTX}"
  export LOCAL_OLLAMA_GPU_NUM_CTX="${GPU_NUM_CTX}"

  export TEST25_LOCAL_ACCELERATOR="${LOCAL_ACCELERATOR}"
  export TEST25_LOCAL_GPU_VENDOR="${LOCAL_GPU_VENDOR}"
  export TEST25_LOCAL_GPU_ID="${LOCAL_GPU_ID}"
  export TEST25_LOCAL_GPU_VRAM_MB="${LOCAL_GPU_VRAM_MB}"
  export TEST25_LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT}"
  export TEST25_LOCAL_TOTAL_GPU_VRAM_MB="${LOCAL_TOTAL_GPU_VRAM_MB}"
  export TEST25_LOCAL_VISIBLE_GPU_IDS="${LOCAL_VISIBLE_GPU_IDS}"
  export TEST25_LOCAL_ACCELERATOR_REASON="${LOCAL_ACCELERATOR_REASON}"

  if [[ "${LOCAL_ACCELERATOR}" == "gpu" ]]; then
    if [[ "${LOCAL_GPU_VENDOR}" == "nvidia" ]]; then
      export CUDA_VISIBLE_DEVICES="${LOCAL_VISIBLE_GPU_IDS}"
      unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL GGML_VK_VISIBLE_DEVICES || true
    elif [[ "${LOCAL_GPU_VENDOR}" == "amd" ]]; then
      export ROCR_VISIBLE_DEVICES="${LOCAL_VISIBLE_GPU_IDS}"
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
export LOCAL_OLLAMA_MODEL_REPO="${MODEL_REPO}"
export LOCAL_OLLAMA_MODEL_FILE="${MODEL_MERGE_OUTPUT_FILE:-${MODEL_FILE}}"
export LOCAL_OLLAMA_MODEL_FILES="${MODEL_FILES_JOINED}"
export LOCAL_OLLAMA_MODEL_NAME="${MODEL_NAME}"
export LOCAL_OLLAMA_PROFILE_ID="${MODEL_PROFILE_ID}"
export LOCAL_OLLAMA_MIN_GPU_VRAM_MB="${MIN_GPU_VRAM_MB}"
export LOCAL_OLLAMA_CPU_NUM_CTX="${CPU_NUM_CTX}"
export LOCAL_OLLAMA_GPU_NUM_CTX="${GPU_NUM_CTX}"
export TEST25_LOCAL_ACCELERATOR="${LOCAL_ACCELERATOR}"
export TEST25_LOCAL_GPU_VENDOR="${LOCAL_GPU_VENDOR}"
export TEST25_LOCAL_GPU_ID="${LOCAL_GPU_ID}"
export TEST25_LOCAL_GPU_VRAM_MB="${LOCAL_GPU_VRAM_MB}"
export TEST25_LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT}"
export TEST25_LOCAL_TOTAL_GPU_VRAM_MB="${LOCAL_TOTAL_GPU_VRAM_MB}"
export TEST25_LOCAL_VISIBLE_GPU_IDS="${LOCAL_VISIBLE_GPU_IDS}"
export TEST25_LOCAL_ACCELERATOR_REASON="${LOCAL_ACCELERATOR_REASON}"
EOF

  if [[ "${LOCAL_ACCELERATOR}" == "gpu" ]]; then
    if [[ "${LOCAL_GPU_VENDOR}" == "nvidia" ]]; then
      cat >> "${RUNTIME_ENV_FILE}" <<EOF
export CUDA_VISIBLE_DEVICES="${LOCAL_VISIBLE_GPU_IDS}"
EOF
    elif [[ "${LOCAL_GPU_VENDOR}" == "amd" ]]; then
      cat >> "${RUNTIME_ENV_FILE}" <<EOF
export ROCR_VISIBLE_DEVICES="${LOCAL_VISIBLE_GPU_IDS}"
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
  msg "Model profile:     ${MODEL_PROFILE_ID}"
  msg "Profile size (GB): ${MODEL_PROFILE_PUBLISHED_SIZE_GB}"
  msg "Selection reason:  ${MODEL_SELECTION_REASON}"
  msg "Model repo:       ${MODEL_REPO}"
  msg "Model file:       ${MODEL_FILE:-<multi-file>}"
  msg "Model files:      ${MODEL_FILES_JOINED:-<single-file>}"
  msg "Model import path:${MODEL_IMPORT_PATH}"
  msg "Model alias:      ${MODEL_NAME}"
  msg "Accelerator mode: ${LOCAL_ACCELERATOR}"
  msg "GPU vendor:       ${LOCAL_GPU_VENDOR}"
  msg "Visible GPU ids:  ${LOCAL_VISIBLE_GPU_IDS:-<none>}"
  msg "GPU id:           ${LOCAL_GPU_ID:-<none>}"
  msg "GPU count:        ${LOCAL_GPU_COUNT}"
  msg "Best GPU VRAM MB: ${LOCAL_GPU_VRAM_MB}"
  msg "Total GPU VRAM MB:${LOCAL_TOTAL_GPU_VRAM_MB}"
  msg "Min GPU VRAM MB:  ${MIN_GPU_VRAM_MB}"
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
  local download_list=()
  if [[ -n "${MODEL_FILES_JOINED}" ]]; then
    IFS=';' read -r -a download_list <<< "${MODEL_FILES_JOINED}"
  elif [[ -n "${MODEL_FILE}" ]]; then
    download_list=("${MODEL_FILE}")
  fi
  [[ ${#download_list[@]} -gt 0 ]] || die "No GGUF files were configured for the selected model profile."

  local missing=()
  local entry
  for entry in "${download_list[@]}"; do
    [[ -f "${MODEL_DIR}/${entry}" ]] || missing+=("${entry}")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    msg "GGUF source files already present for ${MODEL_NAME}"
    return
  fi

  msg "Downloading GGUF from Hugging Face..."
  for entry in "${missing[@]}"; do
    "${VENV_DIR}/bin/hf" download "${MODEL_REPO}" "${entry}" --local-dir "${MODEL_DIR}"
  done
}

get_gguf_split_cmd() {
  if [[ -n "${LOCAL_OLLAMA_GGUF_SPLIT_PATH:-}" && -x "${LOCAL_OLLAMA_GGUF_SPLIT_PATH}" ]]; then
    printf '%s\n' "${LOCAL_OLLAMA_GGUF_SPLIT_PATH}"
    return 0
  fi
  if command -v gguf-split >/dev/null 2>&1; then
    command -v gguf-split
    return 0
  fi
  return 1
}

ensure_merged_model_file() {
  [[ -n "${MODEL_MERGE_OUTPUT_FILE}" ]] || return 0

  local merged_target="${MODEL_DIR}/${MODEL_MERGE_OUTPUT_FILE}"
  if [[ -f "${merged_target}" ]]; then
    msg "Merged GGUF already present: ${merged_target}"
    return
  fi

  local download_list=()
  IFS=';' read -r -a download_list <<< "${MODEL_FILES_JOINED}"
  [[ ${#download_list[@]} -gt 0 ]] || die "Profile ${MODEL_PROFILE_ID} requires sharded GGUF inputs but none were configured."

  local first_shard="${MODEL_DIR}/${download_list[0]}"
  [[ -f "${first_shard}" ]] || die "First GGUF shard missing: ${first_shard}"

  local gguf_split
  gguf_split="$(get_gguf_split_cmd)" || die "Profile ${MODEL_PROFILE_ID} requires GGUF shard merging. Install gguf-split or set LOCAL_OLLAMA_GGUF_SPLIT_PATH."
  msg "Merging GGUF shards into ${merged_target} ..."
  "${gguf_split}" --merge "${first_shard}" "${merged_target}"
}

write_modelfile() {
  mkdir -p "${MODEL_DIR}"
  cat > "${MODEL_DIR}/Modelfile" <<EOF
FROM ${MODEL_IMPORT_PATH}
PARAMETER num_ctx ${LOCAL_NUM_CTX}
EOF
}

ollama_model_exists() {
  ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "${MODEL_NAME}"
}

ensure_ollama_model() {
  ensure_merged_model_file
  write_modelfile
  msg "Checking whether Ollama model alias already exists: ${MODEL_NAME}"
  if ollama_model_exists; then
    msg "Ollama model already imported: ${MODEL_NAME}"
    return
  fi
  msg "Importing GGUF into Ollama as ${MODEL_NAME}..."
  ollama create "${MODEL_NAME}" -f "${MODEL_DIR}/Modelfile"
  msg "Ollama create finished for ${MODEL_NAME}"
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

pipeline_model_args() {
  printf '%s\n' \
    --model "${MODEL_NAME}" \
    --bootstrap-model "${MODEL_NAME}" \
    --planning-model "${MODEL_NAME}" \
    --bulk-artifact-model "${MODEL_NAME}"
}

pipeline_step100_resume_args() {
  local manifest="${PROJECT_DIR}/_step100_resume/manifest.json"
  if [[ -d "${PROJECT_DIR}/_step100_resume" && ! -f "${manifest}" ]]; then
    printf '%s\n' --resume-step100
    return
  fi
  if [[ -f "${manifest}" ]]; then
    "${VENV_DIR}/bin/python" - "${manifest}" <<'PY'
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

warn_if_cpu_only_memory_is_tight() {
  if [[ "${LOCAL_ACCELERATOR}" != "cpu" ]]; then
    return
  fi
  local mem_gb
  mem_gb="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)"
  if [[ -n "${mem_gb}" && "${mem_gb}" -lt 48 ]]; then
    msg "WARNING: CPU-only mode with ${mem_gb} GB RAM may be rough for ${MODEL_FILE}."
  fi
}

run_pipeline_20k() {
  local python_bin="${VENV_DIR}/bin/python"
  [[ -x "${python_bin}" ]] || die "Python venv not found at ${python_bin}. Run install_all.sh first."
  local model_args=()
  mapfile -t model_args < <(pipeline_model_args)

  "${python_bin}" "${PROJECT_DIR}/run_pipeline.py" \
    --fresh \
    --n-movies 20000 \
    --start-year 1950 \
    --end-year 2025 \
    --n-titles 20000 \
    --n-persons 45000 \
    --n-companies 1000 \
    --n-keywords 15000 \
    --n-characters 100000 \
    "${model_args[@]}" \
    --enable-llm-evolution \
    --until-step 130 \
    "$@"
}

run_pipeline_continue_20k() {
  local python_bin="${VENV_DIR}/bin/python"
  [[ -x "${python_bin}" ]] || die "Python venv not found at ${python_bin}. Run install_all.sh first."
  local model_args=()
  local resume_args=()
  mapfile -t model_args < <(pipeline_model_args)
  mapfile -t resume_args < <(pipeline_step100_resume_args)

  "${python_bin}" "${PROJECT_DIR}/run_pipeline.py" \
    --n-movies 20000 \
    --start-year 1950 \
    --end-year 2025 \
    --n-titles 20000 \
    --n-persons 45000 \
    --n-companies 1000 \
    --n-keywords 15000 \
    --n-characters 100000 \
    "${model_args[@]}" \
    "${resume_args[@]}" \
    --enable-llm-evolution \
    --until-step 130 \
    "$@"
}

run_pipeline_200k() {
  local python_bin="${VENV_DIR}/bin/python"
  [[ -x "${python_bin}" ]] || die "Python venv not found at ${python_bin}. Run install_all.sh first."
  local model_args=()
  mapfile -t model_args < <(pipeline_model_args)

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
    "${model_args[@]}" \
    --enable-llm-evolution \
    --until-step 130 \
    "$@"
}

run_pipeline_continue_200k() {
  local python_bin="${VENV_DIR}/bin/python"
  [[ -x "${python_bin}" ]] || die "Python venv not found at ${python_bin}. Run install_all.sh first."
  local model_args=()
  local resume_args=()
  mapfile -t model_args < <(pipeline_model_args)
  mapfile -t resume_args < <(pipeline_step100_resume_args)

  "${python_bin}" "${PROJECT_DIR}/run_pipeline.py" \
    --n-movies 200000 \
    --start-year 1950 \
    --end-year 2025 \
    --n-titles 200000 \
    --n-persons 450000 \
    --n-companies 30000 \
    --n-keywords 38000 \
    --n-characters 786000 \
    "${model_args[@]}" \
    "${resume_args[@]}" \
    --enable-llm-evolution \
    --until-step 130 \
    "$@"
}
