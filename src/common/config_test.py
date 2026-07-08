# =============================================================================
#  config_test — real round-trips of default + overlaid configuration
#  why: a silently-misread config would mistune every plasticity update
# =============================================================================
from pathlib import Path

import pytest

from common.config import EngramConfig, load_config, repo_root


def test_defaults_when_no_file(tmp_path: Path):
    cfg = load_config(tmp_path / "absent.toml")
    assert cfg == EngramConfig()
    assert cfg.plasticity.lambda_neg == 0.5
    assert cfg.plasticity.mid_layers == (8, 28)
    assert cfg.plasticity.lr_absorb == 5e-6
    assert cfg.individuation.enabled is False
    assert cfg.individuation.surprise_percentile == 0.7


def test_individuation_overlay(tmp_path: Path):
    toml = tmp_path / "config.toml"
    toml.write_text("[individuation]\nenabled = true\nsurprise_percentile = 0.85\n")
    cfg = load_config(toml)
    assert cfg.individuation.enabled is True
    assert cfg.individuation.surprise_percentile == 0.85
    assert cfg.individuation.selfedit_paraphrases == 4


def test_toml_overlay(tmp_path: Path):
    toml = tmp_path / "config.toml"
    toml.write_text('[server]\nport = 9999\n\n[plasticity]\nlr_reward = 1e-5\nmid_layers = [4, 30]\n')
    cfg = load_config(toml)
    assert cfg.server.port == 9999
    assert cfg.plasticity.lr_reward == 1e-5
    assert cfg.plasticity.mid_layers == (4, 30)
    assert cfg.sampling.temperature == 0.6


def test_unknown_key_rejected(tmp_path: Path):
    toml = tmp_path / "config.toml"
    toml.write_text("[plasticity]\nlearning_rate_typo = 1.0\n")
    with pytest.raises(ValueError, match="learning_rate_typo"):
        load_config(toml)


def test_unknown_section_rejected(tmp_path: Path):
    toml = tmp_path / "config.toml"
    toml.write_text("[plasticty]\nenabled = true\n")
    with pytest.raises(ValueError, match="plasticty"):
        load_config(toml)


def test_repo_root_is_engram_checkout():
    assert (repo_root() / "DESIGN.md").exists()


def test_forced_path_isolates_from_local_config(tmp_path: Path):
    from common.config import set_forced_config_path

    live = tmp_path / "live.toml"
    live.write_text("[individuation]\nenabled = true\n")
    set_forced_config_path(live)
    try:
        assert load_config().individuation.enabled is True
        set_forced_config_path(tmp_path / "absent.toml")
        assert load_config() == EngramConfig()
    finally:
        set_forced_config_path(tmp_path / "absent.toml")
