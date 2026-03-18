"""
V17 Runtime Configuration System
=================================
Centralizes all pipeline parameters (calibration targets, modeling priors,
LLM settings, runtime thresholds) into a single dataclass-driven config
loaded from v17_config.json.  Provides workspace path management so every
script resolves input/output/cache paths consistently.

Usage:
    from v17_runtime import resolve_workspace, bootstrap_env_from_argv
    bootstrap_env_from_argv()                         # parse --data-dir etc.
    ws = resolve_workspace(script_dir=Path(__file__).parent)
    person_csv = ws.input_path("entities", "person.csv")
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ── Environment variable names ────────────────────────────────────────────────
ENV_DATA_DIR = "DATA_SYS_V17_DATA_DIR"
ENV_OUTPUT_DIR = "DATA_SYS_V17_OUTPUT_DIR"
ENV_CONFIG = "DATA_SYS_V17_CONFIG"
ENV_LLM_PROVIDER = "DATA_SYS_V17_LLM_PROVIDER"
ENV_LLM_CACHE_DIR = "DATA_SYS_V17_LLM_CACHE_DIR"


def _as_path(raw: str | os.PathLike[str] | None, default: Path) -> Path:
    if raw is None:
        return default.resolve()
    return Path(raw).expanduser().resolve()


# ═══════════════════════════════════════════════════════════════════════
# Dataclass configuration hierarchy
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CalibrationTargets:
    """Numeric targets that calibration loops try to hit."""
    cast_gini_min: float = 0.48
    cast_gini_max: float = 0.60
    pop_cast_corr_min: float = 0.32
    edge_giant_component_min: float = 0.84
    best_friend_same_comm_min: float = 0.76
    bridge_ratio_min: float = 0.03
    bridge_ratio_max: float = 0.12
    blockbuster_tail_target: int = 110


@dataclass
class ModelingPriors:
    """Statistical priors that shape generation distributions."""
    # Entity load targets (avg movies per entity)
    target_actor_load: float = 6.2
    target_director_load: float = 7.5
    target_company_load: float = 10.5
    role_scarcity_power: float = 1.35
    # Shortlist / exploration
    shortlist_size: int = 96
    # Graph generation
    graph_candidate_k: int = 160
    graph_bridge_budget: float = 0.055
    graph_same_block_share: float = 0.78
    graph_closure_budget: float = 0.12
    graph_closure_top_k: int = 12
    graph_director_candidate_k: int = 48
    # Financial regime modeling
    financial_regime_amplitude: float = 0.18
    financial_slate_pressure: float = 0.075
    financial_momentum_decay: float = 0.68
    financial_recent_horizon: int = 6
    financial_genre_memory_weight: float = 0.11
    # Temporal evolution
    temporal_macro_event_budget: int = 18
    temporal_micro_event_budget: int = 48
    # Post-generation critic
    critic_sample_size: int = 12
    critic_max_actions: int = 20
    critic_max_repairs_per_title: int = 2


@dataclass
class LLMRoleConfig:
    """Config for a single LLM role (structured, creative, critic)."""
    provider: str = "gemini"
    model: str = "gemini-3.1-flash-lite-preview"
    temperature: float = 0.25
    response_mime_type: str | None = "application/json"
    max_output_tokens: int | None = None


@dataclass
class LLMSettings:
    """All LLM-related settings, with per-role configurations."""
    provider: str = "gemini"
    cache_namespace: str = "v17"
    structured: LLMRoleConfig = field(default_factory=LLMRoleConfig)
    creative: LLMRoleConfig = field(
        default_factory=lambda: LLMRoleConfig(
            provider="gemini",
            model="gemini-3.1-flash-lite-preview",
            temperature=0.75,
            response_mime_type="application/json",
        )
    )
    critic: LLMRoleConfig = field(
        default_factory=lambda: LLMRoleConfig(
            provider="gemini",
            model="gemini-3.1-flash-lite-preview",
            temperature=0.15,
            response_mime_type="application/json",
        )
    )


@dataclass
class RuntimeSettings:
    """Thresholds and limits for pipeline execution."""
    lazy_world_threshold_movies: int = 250
    full_cache_threshold_movies: int = 1000
    # TV series scaling
    tv_series_floor_small: int = 8
    tv_series_sqrt_scale: float = 6.0
    tv_series_large_ratio: float = 0.06
    tv_series_max: int = 8000
    # Contract background fill
    contract_background_min_people: int = 180
    contract_background_sqrt_scale: float = 40.0
    contract_background_linear_ratio: float = 0.20
    contract_background_max_people: int = 20000
    # Media links
    media_links_min_total: int = 12
    media_links_target_ratio: float = 2.0
    media_links_per_movie: int = 4
    media_links_bucket_cap: int = 96
    media_links_max_total: int = 0


@dataclass
class V17Config:
    """Top-level config combining all sub-configs."""
    version: str = "17"
    data_dir: str | None = None
    output_dir: str | None = None
    llm_provider: str = "gemini"
    llm_cache_dir: str | None = None
    calibration: CalibrationTargets = field(default_factory=CalibrationTargets)
    priors: ModelingPriors = field(default_factory=ModelingPriors)
    llm: LLMSettings = field(default_factory=LLMSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)


# ═══════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════

def _merge_dataclass(dc: Any, payload: dict[str, Any]) -> Any:
    """Recursively merge a dict payload into a nested dataclass."""
    for key, value in payload.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_v17_config(config_path: str | os.PathLike[str] | None = None) -> V17Config:
    """Load config from JSON file (if exists), then overlay env vars."""
    cfg = V17Config()
    raw_path = config_path or os.getenv(ENV_CONFIG)
    if raw_path:
        path = Path(raw_path).expanduser().resolve()
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                _merge_dataclass(cfg, payload)

    # Environment variables override JSON values
    if os.getenv(ENV_LLM_PROVIDER):
        cfg.llm_provider = str(os.getenv(ENV_LLM_PROVIDER))
        cfg.llm.provider = cfg.llm_provider
        cfg.llm.structured.provider = cfg.llm_provider
        cfg.llm.creative.provider = cfg.llm_provider
        cfg.llm.critic.provider = cfg.llm_provider
    if os.getenv(ENV_LLM_CACHE_DIR):
        cfg.llm_cache_dir = str(os.getenv(ENV_LLM_CACHE_DIR))
    if os.getenv(ENV_DATA_DIR):
        cfg.data_dir = str(os.getenv(ENV_DATA_DIR))
    if os.getenv(ENV_OUTPUT_DIR):
        cfg.output_dir = str(os.getenv(ENV_OUTPUT_DIR))

    return cfg


# ═══════════════════════════════════════════════════════════════════════
# Workspace path management
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WorkspacePaths:
    """Manages data_dir (read from), output_dir (write to), cache_dir.

    input_path() prefers output_dir if the file exists there, otherwise
    falls back to data_dir.  This lets the pipeline read base data from
    a shared location while writing all outputs to a separate directory.
    """
    data_dir: Path
    output_dir: Path
    cache_dir: Path
    config: V17Config

    def input_path(self, *parts: str) -> Path:
        """Resolve an input path: prefer output_dir, fall back to data_dir."""
        rel = Path(*parts)
        out = self.output_dir / rel
        src = self.data_dir / rel
        if out.exists():
            if out.is_file():
                return out
            try:
                if any(out.iterdir()) or not src.exists():
                    return out
            except OSError:
                return out
        return src

    def output_path(self, *parts: str) -> Path:
        """Resolve an output path, creating parent dirs as needed."""
        path = self.output_dir / Path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def cache_path(self, *parts: str) -> Path:
        """Resolve a cache path, creating parent dirs as needed."""
        path = self.cache_dir / Path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_dirs(self):
        """Create standard output subdirectories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for rel in ("entities", "graph", "checkpoints", "_dev"):
            (self.output_dir / rel).mkdir(parents=True, exist_ok=True)


def resolve_workspace(
    *,
    script_dir: str | os.PathLike[str] | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> WorkspacePaths:
    """Build a fully resolved WorkspacePaths from args + config + env."""
    default_dir = Path(script_dir or Path(__file__).resolve().parent).resolve()
    cfg = load_v17_config(config_path)

    data_root = _as_path(data_dir or cfg.data_dir, default_dir)
    output_root = _as_path(output_dir or cfg.output_dir, default_dir)
    cache_default = output_root / ".cache" / "llm"
    cache_root = _as_path(cfg.llm_cache_dir, cache_default)

    ws = WorkspacePaths(
        data_dir=data_root,
        output_dir=output_root,
        cache_dir=cache_root,
        config=cfg,
    )
    ws.ensure_dirs()
    return ws


# ═══════════════════════════════════════════════════════════════════════
# CLI & environment helpers
# ═══════════════════════════════════════════════════════════════════════

def export_workspace_env(workspace: WorkspacePaths, config_path: str | None = None) -> dict[str, str]:
    """Export workspace paths as environment variable dict for subprocesses."""
    env = {
        ENV_DATA_DIR: str(workspace.data_dir),
        ENV_OUTPUT_DIR: str(workspace.output_dir),
        ENV_LLM_PROVIDER: str(workspace.config.llm_provider),
        ENV_LLM_CACHE_DIR: str(workspace.cache_dir),
    }
    if config_path:
        env[ENV_CONFIG] = str(Path(config_path).expanduser().resolve())
    elif os.getenv(ENV_CONFIG):
        env[ENV_CONFIG] = str(Path(os.getenv(ENV_CONFIG, "")).expanduser().resolve())
    return env


def bootstrap_env_from_argv(argv: list[str] | None = None):
    """Parse shared CLI args (--data-dir, --config, etc.) into env vars.

    Call this early in any script that uses the v17 config system, before
    importing modules that read env vars at import time.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--config")
    parser.add_argument("--llm-provider")
    ns, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    if ns.data_dir:
        os.environ[ENV_DATA_DIR] = str(Path(ns.data_dir).expanduser().resolve())
    if ns.output_dir:
        os.environ[ENV_OUTPUT_DIR] = str(Path(ns.output_dir).expanduser().resolve())
    if ns.config:
        os.environ[ENV_CONFIG] = str(Path(ns.config).expanduser().resolve())
    if ns.llm_provider:
        os.environ[ENV_LLM_PROVIDER] = str(ns.llm_provider)


def add_shared_runtime_args(parser: argparse.ArgumentParser):
    """Add --data-dir, --output-dir, --config, --llm-provider to an existing parser."""
    parser.add_argument("--data-dir", default=os.getenv(ENV_DATA_DIR))
    parser.add_argument("--output-dir", default=os.getenv(ENV_OUTPUT_DIR))
    parser.add_argument("--config", default=os.getenv(ENV_CONFIG))
    parser.add_argument("--llm-provider", default=os.getenv(ENV_LLM_PROVIDER))


def effective_model_config(workspace: WorkspacePaths) -> dict[str, Any]:
    """Return a serializable snapshot of the active configuration."""
    return {
        "version": workspace.config.version,
        "data_dir": str(workspace.data_dir),
        "output_dir": str(workspace.output_dir),
        "llm_provider": workspace.config.llm_provider,
        "llm_cache_dir": str(workspace.cache_dir),
        "calibration": asdict(workspace.config.calibration),
        "priors": asdict(workspace.config.priors),
        "llm": asdict(workspace.config.llm),
        "runtime": asdict(workspace.config.runtime),
    }
