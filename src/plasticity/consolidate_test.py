# =============================================================================
#  consolidate_test — the full dream path on the 0.8B model
#  why: consolidation folds the overlay into the bf16 master and requantizes a
#  serving base; verify the merged weight equals base+delta and that the
#  requantized model actually loads and generates. Heavy but real: it builds a
#  bf16 master fixture by dequantizing the 4-bit test model.
# =============================================================================
from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm import generate, load
from mlx_lm.convert import convert

from common.config import load_config
from plasticity import consolidate
from plasticity.adapter import attach_overlay


# ##################################################################
# consolidated fixture
# run the whole heavy pipeline once: dequantize a bf16 master, apply a nonzero
# overlay delta, dream it into a merged master, and requantize to a serving base
@pytest.fixture(scope="module")
def consolidated(tmp_path_factory):
    config = load_config()
    work = tmp_path_factory.mktemp("consolidate")
    master = work / "master"
    convert(config.model.test_path, str(master), dequantize=True, dtype="bfloat16")
    model, tok = load(config.model.test_path)
    overlay = attach_overlay(model, config.plasticity)
    overlay.reset()
    weight_path, module = overlay.adapters[0]
    module.b = (mx.random.normal(module.b.shape) * 0.05).astype(mx.bfloat16)
    mx.eval(module.b)
    delta = overlay.merge_deltas()[weight_path]
    merged = work / "merged"
    consolidate.dream(str(master), overlay, str(merged), [weight_path])
    quant = work / "quant"
    consolidate.quantize_targets(str(merged), str(quant))
    return {"master": master, "merged": merged, "quant": quant, "path": weight_path, "delta": delta}


# ##################################################################
# merge adds the delta to the targeted weight only
# the merged master weight equals base+delta at the targeted path, while an
# untargeted weight is copied through unchanged
def test_merge_adds_delta(consolidated):
    base = mx.load(str(consolidated["master"] / "model.safetensors"))
    merged = mx.load(str(consolidated["merged"] / "model.safetensors"))
    path = consolidated["path"]
    expected = base[path].astype(mx.float32) + consolidated["delta"]
    assert bool(mx.allclose(merged[path].astype(mx.float32), expected, atol=1e-2, rtol=1e-2))
    other = "language_model.model.norm.weight"
    assert bool(mx.all(base[other] == merged[other]))


# ##################################################################
# requantized base loads and generates
# the 4-bit serving copy produced from the merged master loads through mlx-lm
# and produces finite logits + generated text
def test_requantized_loads_and_generates(consolidated):
    model, tok = load(str(consolidated["quant"]))
    logits = model(mx.array([tok.encode("Hello")]))
    mx.eval(logits)
    assert logits.shape[-1] == model.language_model.args.vocab_size
    assert bool(mx.all(mx.isfinite(logits[0, -1].astype(mx.float32))))
    text = generate(model, tok, "The sky is", max_tokens=4, verbose=False)
    assert isinstance(text, str)
