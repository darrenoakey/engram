# =============================================================================
#  checkpoints_test — real overlay checkpoint ring on the 0.8B model
#  why: auto-rollback depends on durable checkpoints, last-known-good selection,
#  and ring pruning behaving exactly as specified
# =============================================================================
from __future__ import annotations

import threading

import mlx.core as mx
import pytest
from mlx_lm import load

from common.config import load_config
from plasticity.adapter import attach_overlay
from plasticity.checkpoints import Checkpoints


# ##################################################################
# overlay fixture
# one real model + overlay reused across the checkpoint tests
@pytest.fixture(scope="module")
def overlay():
    config = load_config()
    model, _tok = load(config.model.test_path)
    return attach_overlay(model, config.plasticity)


# ##################################################################
# set marker
# stamp adapter[0].b with a recognisable constant so a restore is verifiable
def _set_marker(overlay, value):
    overlay.reset()
    module = overlay.adapters[0][1]
    module.b = (mx.ones(module.b.shape) * value).astype(mx.bfloat16)
    mx.eval(module.b)
    return float(value)


# ##################################################################
# save and restore by id
# a checkpoint restores the exact overlay tensors it captured
def test_save_and_restore(overlay, tmp_path):
    store = Checkpoints(tmp_path / "cp", ring=20)
    _set_marker(overlay, 0.02)
    checkpoint_id = store.save(overlay, updates_at_save=5, canary_clean=True)
    _set_marker(overlay, 0.09)
    store.restore(overlay, checkpoint_id)
    assert abs(float(overlay.adapters[0][1].b[0, 0]) - 0.02) < 1e-3


# ##################################################################
# restore last good skips unclean
# with no id, restore selects the most recent canary-clean checkpoint
def test_restore_last_good(overlay, tmp_path):
    store = Checkpoints(tmp_path / "cp", ring=20)
    _set_marker(overlay, 0.01)
    store.save(overlay, 1, canary_clean=True)
    threading.Event().wait(0.02)
    _set_marker(overlay, 0.03)
    store.save(overlay, 2, canary_clean=True)
    threading.Event().wait(0.02)
    _set_marker(overlay, 0.05)
    store.save(overlay, 3, canary_clean=False)
    store.restore(overlay, None)
    assert abs(float(overlay.adapters[0][1].b[0, 0]) - 0.03) < 1e-3


# ##################################################################
# ring prunes oldest
# saving past the ring keeps only the newest checkpoints
def test_ring_prune(overlay, tmp_path):
    store = Checkpoints(tmp_path / "cp", ring=2)
    for value in (0.01, 0.02, 0.03, 0.04):
        _set_marker(overlay, value)
        store.save(overlay, 0, canary_clean=True)
        threading.Event().wait(0.02)
    metas = store.list()
    assert len(metas) == 2
    assert len(list((tmp_path / "cp").glob("*.safetensors"))) == 2
