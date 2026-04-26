#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

candidate_bases=()
if [[ -n "${OLLAMA_BASE_URL:-}" ]]; then
  candidate_bases+=("${OLLAMA_BASE_URL%/}")
fi
candidate_bases+=("http://127.0.0.1:11434" "http://localhost:11434")
if command -v ip >/dev/null 2>&1; then
  route_gateway="$(ip route show default 2>/dev/null | awk '/^default / {print $3; exit}' || true)"
  if [[ -n "${route_gateway:-}" ]]; then
    candidate_bases+=("http://$route_gateway:11434")
  fi
fi
if [[ -r /etc/resolv.conf ]]; then
  gateway="$(awk '/^nameserver / {print $2; exit}' /etc/resolv.conf || true)"
  if [[ -n "${gateway:-}" ]]; then
    candidate_bases+=("http://$gateway:11434")
  fi
fi

mkdir -p "$BASE_DIR/reports"

for base in "${candidate_bases[@]}"; do
  if response="$(curl -fsS --max-time 3 "$base/api/tags" 2>/dev/null)"; then
    model="$(
      printf '%s' "$response" | "$PYTHON_BIN" -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print("")
    raise SystemExit
models = data.get("models") or []
names = [str(item.get("name") or item.get("model") or "") for item in models if isinstance(item, dict)]
preferred_prefixes = (
    "qwen36",
    "qwen3.6",
    "qwen35-35b",
    "qwen3.5-35b",
    "qwen35",
    "qwen3.5",
    "qwen",
)
for prefix in preferred_prefixes:
    for name in names:
        if name.lower().startswith(prefix):
            print(name)
            raise SystemExit
print(names[0] if names else "")
'
    )"
    env_path="$BASE_DIR/reports/ollama_detected.env"
    {
      printf 'export LLM_PROVIDER=local\n'
      printf 'export LOCAL_LLM_URL=%q\n' "$base/v1"
      printf 'export LOCAL_LLM_API_KEY=not-needed\n'
      if [[ -n "$model" ]]; then
        printf 'export LOCAL_LLM_MODEL=%q\n' "$model"
      fi
    } > "$env_path"
    echo "Ollama reachable: $base"
    if [[ -n "$model" ]]; then
      echo "First model: $model"
    else
      echo "No models reported by Ollama."
    fi
    echo "Wrote $env_path"
    exit 0
  fi
done

win_ps="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
if [[ -f "$win_ps" ]]; then
  if response="$("$win_ps" -NoProfile -Command 'try { $r = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3; $r | ConvertTo-Json -Depth 5 } catch { exit 1 }' 2>/dev/null | tr -d '\r')" && [[ -n "$response" ]]; then
    model="$(
      printf '%s' "$response" | "$PYTHON_BIN" -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print("")
    raise SystemExit
models = data.get("models") or []
names = [str(item.get("name") or item.get("model") or "") for item in models if isinstance(item, dict)]
preferred_prefixes = (
    "qwen36",
    "qwen3.6",
    "qwen35-35b",
    "qwen3.5-35b",
    "qwen35",
    "qwen3.5",
    "qwen",
)
for prefix in preferred_prefixes:
    for name in names:
        if name.lower().startswith(prefix):
            print(name)
            raise SystemExit
print(names[0] if names else "")
'
    )"
    env_path="$BASE_DIR/reports/ollama_windows_seen.env"
    {
      printf '# Windows Ollama is running, but this WSL shell could not reach it directly.\n'
      printf '# Restart Ollama with OLLAMA_HOST=0.0.0.0:11434, then rerun detect_ollama_endpoint.sh.\n'
      printf 'export LLM_PROVIDER=local\n'
      printf 'export LOCAL_LLM_API_KEY=not-needed\n'
      if [[ -n "$model" ]]; then
        printf 'export LOCAL_LLM_MODEL=%q\n' "$model"
      fi
    } > "$env_path"
    echo "Windows Ollama is running, but it is not reachable from WSL."
    if [[ -n "$model" ]]; then
      echo "Detected Windows model: $model"
    fi
    echo "Likely fix for local WSL smoke testing:"
    echo "  restart Ollama with OLLAMA_HOST=0.0.0.0:11434, then rerun ./detect_ollama_endpoint.sh"
    echo "Wrote $env_path"
    exit 2
  fi
fi

echo "Ollama was not reachable from WSL on the common endpoints." >&2
echo "Start Ollama on Windows, then rerun this script." >&2
exit 1
