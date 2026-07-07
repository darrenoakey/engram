# =============================================================================
#  config — single frozen source of truth for every tunable
#  why: local/config.toml overlays code defaults; no environment variables,
#  ever (DESIGN.md §3); unknown keys fail loudly instead of silently no-oping
# =============================================================================
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelConfig:
    serve_path: str = "/Volumes/Gumby/models/ornith-9b-4bit"
    master_path: str = "/Volumes/Gumby/models/ornith-9b-bf16"
    test_path: str = "/Volumes/Gumby/models/qwen3.5-0.8b-4bit"
    base_generations_dir: str = "/Volumes/Gumby/models/engram-generations"


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8500


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 4096


@dataclass(frozen=True)
class PlasticityConfig:
    enabled: bool = True
    self_reinforce: str = "gated"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_scope: str = "mid_mlp_full_attn"
    mid_layers: tuple = (8, 28)
    lr_reinforce: float = 1e-6
    lr_reward: float = 5e-6
    lambda_neg: float = 0.5
    beta_kl: float = 0.05
    max_span_tokens: int = 256
    replay_spans: int = 1
    topk_grad_fraction: float = 0.3
    grad_clip_norm: float = 1.0
    delta_frobenius_cap: float = 0.05
    adapter_norm_ceiling: float = 5.0
    update_kl_budget: float = 0.5
    include_think_tokens: bool = False


@dataclass(frozen=True)
class GuardsConfig:
    canary_every: int = 20
    canary_kl_budget: float = 0.15
    canary_breaches_to_rollback: int = 2
    checkpoint_every: int = 10
    checkpoint_ring: int = 20


@dataclass(frozen=True)
class FeedbackConfig:
    auto_tool_scoring: bool = True
    tool_success_reward: float = 0.3
    tool_failure_reward: float = -0.5


@dataclass(frozen=True)
class EngramConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    guards: GuardsConfig = field(default_factory=GuardsConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)


def _merged(defaults, section: dict, name: str):
    known = {f.name for f in fields(defaults)}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"unknown config keys in [{name}]: {sorted(unknown)}")
    cleaned = {k: tuple(v) if isinstance(v, list) else v for k, v in section.items()}
    return replace(defaults, **cleaned)


def load_config(path: Path | None = None) -> EngramConfig:
    toml_path = path if path is not None else repo_root() / "local" / "config.toml"
    base = EngramConfig()
    if not toml_path.exists():
        return base
    raw = tomllib.loads(toml_path.read_text())
    known_sections = {f.name for f in fields(base)}
    unknown = set(raw) - known_sections
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    merged = {name: _merged(getattr(base, name), raw[name], name) for name in raw}
    return replace(base, **merged)
