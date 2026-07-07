# =============================================================================
#  adapter_test — real overlay behaviour on the 0.8B qwen3_5 test model
#  why: the overlay is the substrate every update writes to; verify targeting,
#  trainable-param isolation, enable toggling, exact snapshot/save round-trips,
#  and — critically — that merge_deltas orientation matches the base weight
# =============================================================================
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm import load

from common.config import load_config
from plasticity.adapter import PlasticLinear, attach_overlay, band


# ##################################################################
# overlay fixture
# load the real quantized test model once and attach a fresh overlay; mutating
# tests call overlay.reset() to return to cold start
@pytest.fixture(scope="module")
def overlay_bundle():
    config = load_config()
    model, tok = load(config.model.test_path)
    overlay = attach_overlay(model, config.plasticity)
    return model, tok, overlay, config


# ##################################################################
# attach targets the right modules
# MLP on every band layer, attention q/k/v/o only on full-attention band layers,
# DeltaNet in_proj_* left untouched
def test_attach_targets(overlay_bundle):
    model, _tok, overlay, config = overlay_bundle
    lo, hi = band(config.plasticity.mid_layers, len(model.layers))
    full_attn = [i for i in range(lo, hi + 1) if not model.layers[i].is_linear]
    expected = (hi - lo + 1) * 3 + len(full_attn) * 4
    assert len(overlay.adapters) == expected
    linear_layer = next(model.layers[i] for i in range(lo, hi + 1) if model.layers[i].is_linear)
    assert not isinstance(linear_layer.linear_attn.in_proj_qkv, PlasticLinear)
    assert isinstance(model.layers[full_attn[0]].self_attn.q_proj, PlasticLinear)


# ##################################################################
# only adapters are trainable
# after freezing the base, the model's trainable params are exactly the overlay
# A/B tensors — nothing from the quantized base leaks in
def test_only_adapters_trainable(overlay_bundle):
    model, _tok, overlay, _config = overlay_bundle
    model_keys = {k for k, _ in tree_flatten(model.trainable_parameters())}
    assert model_keys == set(overlay.trainable_parameters())
    assert len(model_keys) == 2 * len(overlay.adapters)
    assert all(k.endswith(".a") or k.endswith(".b") for k in model_keys)


# ##################################################################
# enable toggle and disabled context
# a nonzero adapter changes logits when enabled; disabled() forwards the base
def test_enabled_toggle(overlay_bundle):
    model, tok, overlay, _config = overlay_bundle
    overlay.reset()
    module = overlay.adapters[0][1]
    module.b = (mx.random.normal(module.b.shape) * 0.05).astype(mx.bfloat16)
    ids = mx.array([tok.encode("Hello there friend")])
    enabled = model(ids)
    with overlay.disabled():
        base = model(ids)
    assert float(mx.abs(enabled - base).max()) > 0.0
    overlay.enabled = True
    assert bool(mx.allclose(model(ids), enabled))


# ##################################################################
# snapshot restore is bit exact
# proof #5 (in-memory): restore returns every adapter tensor to its exact bytes
def test_snapshot_restore_bit_exact(overlay_bundle):
    _model, _tok, overlay, _config = overlay_bundle
    overlay.reset()
    snap = overlay.snapshot()
    module = overlay.adapters[0][1]
    module.a = module.a + 1.0
    overlay.restore(snap)
    restored = overlay.trainable_parameters()
    for key, value in snap.items():
        assert bool(mx.all(restored[key] == value))


# ##################################################################
# save load round trips
# proof #5 (on disk): safetensors save then load reproduces the params exactly
def test_save_load_roundtrip(overlay_bundle, tmp_path):
    _model, _tok, overlay, _config = overlay_bundle
    overlay.reset()
    for _weight_path, module in overlay.adapters[:3]:
        module.b = (mx.random.normal(module.b.shape) * 0.03).astype(mx.bfloat16)
    saved = {k: mx.array(v) for k, v in overlay.trainable_parameters().items()}
    path = str(tmp_path / "overlay.safetensors")
    overlay.save(path)
    overlay.reset()
    overlay.load(path)
    loaded = overlay.trainable_parameters()
    for key, value in saved.items():
        assert bool(mx.all(loaded[key] == value))


# ##################################################################
# merge deltas orientation
# proof #6: dequantize a wrapped base, add the merged delta, and confirm the
# result matches the enabled-adapter forward — the delta layout is correct
def test_merge_deltas_orientation(overlay_bundle):
    model, _tok, overlay, _config = overlay_bundle
    overlay.reset()
    weight_path, module = overlay.adapters[0]
    module.b = (mx.random.normal(module.b.shape) * 0.05).astype(mx.bfloat16)
    base = module.base
    dequant = mx.dequantize(base.weight, base.scales, base.biases,
                            group_size=base.group_size, bits=base.bits, mode=base.mode)
    merged_weight = dequant.astype(mx.float32) + overlay.merge_deltas()[weight_path]
    in_dims = merged_weight.shape[1]
    x = mx.random.normal((2, in_dims)).astype(mx.float32)
    via_merge = x @ merged_weight.T
    via_adapter = module(x.astype(mx.bfloat16)).astype(mx.float32)
    assert bool(mx.allclose(via_merge, via_adapter, atol=5e-2, rtol=5e-2))


# ##################################################################
# reset and norms
# cold start has zero delta norm; a nonzero B produces positive norms; reset
# zeroes B again and re-randomises A
def test_reset_and_norms(overlay_bundle):
    _model, _tok, overlay, _config = overlay_bundle
    overlay.reset()
    assert overlay.total_norm() == 0.0
    module = overlay.adapters[0][1]
    module.b = (mx.random.normal(module.b.shape) * 0.05).astype(mx.bfloat16)
    assert overlay.total_norm() > 0.0
    overlay.reset()
    assert float(mx.abs(module.b).max()) == 0.0
    assert float(module.a.std()) > 0.0
    assert isinstance(module.base, nn.QuantizedLinear)
