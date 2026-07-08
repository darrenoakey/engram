# =============================================================================
#  conftest — session-wide test isolation of the on-disk data tree
#  why: canary baselines, traces, checkpoints, journal and replay all resolve
#  through common.store; if tests wrote to the live local/data they would
#  pollute production state — a leftover canary subset silently skips the real
#  boot baseline, and a leftover 0.8B checkpoint could be restored into the 9B
#  overlay. Redirecting the store root for the whole test session prevents it.
# =============================================================================
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from common import config, store


@pytest.fixture(scope="session", autouse=True)
def _isolate_data_root():
    root = Path("output/testing") / f"data-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    store.set_data_root(root)
    config.set_forced_config_path(root / "no-such-config.toml")
    yield
    store.set_data_root(None)
    config.set_forced_config_path(None)
