"""
V17 LLM Provider & Caching
===========================
Centralizes all LLM interactions behind a single ``LLMManager`` class that:

- Provides disk-based caching (``DiskLLMCache``) for deterministic replays
  and cost savings during iterative development.
- Routes calls through ``api_resilience.generate_content_resilient`` for
  automatic timeout, retry, and backoff handling.
- Supports per-role configurations (structured, creative, critic) loaded
  from the v17_config.json via ``WorkspacePaths``.

Usage::

    from v17_runtime import resolve_workspace
    from llm_provider import LLMManager

    ws = resolve_workspace(script_dir=Path(__file__).parent)
    llm = LLMManager(ws)
    resp, inp, out, cost, was_cached = llm.generate(
        role="structured",
        contents="Generate a movie concept...",
        schema_name="movie_concept",
    )
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

from api_resilience import generate_content_resilient, usage_tokens_and_cost
from v17_runtime import WorkspacePaths


# ═══════════════════════════════════════════════════════════════════════
# Serialization helpers (prefer orjson if available, else stdlib json)
# ═══════════════════════════════════════════════════════════════════════

def _dumps(payload: dict[str, Any]) -> bytes:
    if orjson is not None:
        return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _loads(raw: bytes) -> dict[str, Any]:
    if orjson is not None:
        return dict(orjson.loads(raw))
    return json.loads(raw.decode("utf-8"))


# ═══════════════════════════════════════════════════════════════════════
# Disk-based LLM response cache
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CachedResponse:
    """Lightweight stand-in for the real API response object."""
    text: str
    usage_metadata: Any = None


class DiskLLMCache:
    """SHA-256-keyed disk cache for LLM responses.

    Each cached response is stored as a JSON file named ``{sha256}.json``
    under the cache root directory.  Cache keys are deterministic: same
    prompt + config + model → same cache file.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, payload: dict[str, Any]) -> str:
        raw = _dumps(payload)
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, payload: dict[str, Any]) -> CachedResponse | None:
        """Return cached response if it exists, else None."""
        key = self._cache_key(payload)
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            row = _loads(path.read_bytes())
            return CachedResponse(text=str(row.get("text", "")))
        except Exception:
            return None

    def put(self, payload: dict[str, Any], text: str):
        """Store a response text under the computed cache key."""
        key = self._cache_key(payload)
        path = self._cache_path(key)
        path.write_bytes(_dumps({"text": str(text)}))


# ═══════════════════════════════════════════════════════════════════════
# LLM Providers
# ═══════════════════════════════════════════════════════════════════════

class GeminiProvider:
    """Wraps the Google GenAI client, delegating to api_resilience."""

    def __init__(self, api_key: str):
        from google import genai  # type: ignore
        self.client = genai.Client(api_key=api_key)

    def generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: dict[str, Any],
        timeout_sec: float,
        max_attempts: int,
    ):
        return generate_content_resilient(
            client=self.client,
            model=model,
            contents=contents,
            config=config,
            timeout_sec=timeout_sec,
            max_attempts=max_attempts,
            base_delay_sec=4,
            max_delay_sec=30,
        )


# ═══════════════════════════════════════════════════════════════════════
# LLM Manager (main entry point)
# ═══════════════════════════════════════════════════════════════════════

class LLMManager:
    """Unified LLM interface with caching, role configs, and cost tracking.

    Parameters
    ----------
    workspace : WorkspacePaths
        Resolved workspace from ``v17_runtime.resolve_workspace()``.

    Example::

        llm = LLMManager(workspace)
        resp, inp_tokens, out_tokens, cost, was_cached = llm.generate(
            role="structured",
            contents="...",
            schema_name="movie_concept",
        )
    """

    def __init__(self, workspace: WorkspacePaths):
        self.workspace = workspace
        self.cache = DiskLLMCache(workspace.cache_path("responses"))
        self.provider_name = str(workspace.config.llm_provider or "gemini").lower()
        self._provider = None

    def _provider_instance(self):
        """Lazy-initialize the LLM provider."""
        if self._provider is not None:
            return self._provider
        if self.provider_name != "gemini":
            raise RuntimeError(f"Unsupported llm provider: {self.provider_name}")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("missing GEMINI_API_KEY/GOOGLE_API_KEY")
        self._provider = GeminiProvider(api_key=api_key)
        return self._provider

    def _role_cfg(self, role: str) -> Any:
        """Resolve per-role LLM config from v17_config."""
        role = str(role).lower()
        llm_cfg = self.workspace.config.llm
        if role == "creative":
            return llm_cfg.creative
        if role == "critic":
            return llm_cfg.critic
        return llm_cfg.structured

    def generate(
        self,
        *,
        role: str,
        contents: Any,
        schema_name: str,
        seed: int | None = None,
        model: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        timeout_sec: float = 70.0,
        max_attempts: int = 5,
        cache: bool = True,
        input_price_per_million: float = 0.0,
        output_price_per_million: float = 0.0,
    ) -> tuple[Any, int, int, float, bool]:
        """Generate LLM content with caching and cost tracking.

        Returns
        -------
        tuple of (response, input_tokens, output_tokens, cost_usd, was_cached)
            If ``was_cached`` is True, the response is a ``CachedResponse``
            and token counts / cost are zero.
        """
        role_cfg = self._role_cfg(role)

        # Build config from role defaults + overrides
        config = {
            "temperature": float(role_cfg.temperature),
            "thinking_config": {"thinking_budget": 0},
        }
        if role_cfg.response_mime_type:
            config["response_mime_type"] = role_cfg.response_mime_type
        if role_cfg.max_output_tokens:
            config["max_output_tokens"] = int(role_cfg.max_output_tokens)
        if config_overrides:
            config.update(config_overrides)

        effective_model = str(model or role_cfg.model)

        # Cache key includes everything that affects the response
        cache_payload = {
            "provider": self.provider_name,
            "role": role,
            "schema_name": schema_name,
            "model": effective_model,
            "seed": seed,
            "contents": contents,
            "config": config,
        }

        # Check cache first
        if cache:
            cached = self.cache.get(cache_payload)
            if cached is not None:
                return cached, 0, 0, 0.0, True

        # Call provider
        provider = self._provider_instance()
        resp, _stats = provider.generate_content(
            model=effective_model,
            contents=contents,
            config=config,
            timeout_sec=timeout_sec,
            max_attempts=max_attempts,
        )

        # Cache the response text
        text = str(getattr(resp, "text", "") or "")
        if cache:
            self.cache.put(cache_payload, text)

        # Compute token usage and cost
        inp, out, cost = usage_tokens_and_cost(
            resp,
            input_per_million=float(input_price_per_million),
            output_per_million=float(output_price_per_million),
        )
        return resp, inp, out, cost, False
