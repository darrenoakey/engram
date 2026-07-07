# =============================================================================
#  guards_test — real checks of the update bounds
#  why: the masking/clipping/capping guards are the difference between a bounded
#  online update and a norm blowup; verify each does exactly what it claims
# =============================================================================
from __future__ import annotations

import mlx.core as mx

from plasticity import guards


# ##################################################################
# topk mask leaves the rest exactly zero
# proof #3: masking fraction f keeps ~f of the entries and zeros exactly the
# remaining (1-f); kept entries are untouched
def test_topk_mask_fraction_and_exact_zeros():
    mx.random.seed(0)
    grad = mx.random.normal((80, 40))
    masked = guards.mask_one(grad, 0.3)
    nonzero = int((masked != 0).sum())
    fraction = nonzero / grad.size
    assert abs(fraction - 0.3) < 0.02
    kept = masked != 0
    assert bool(mx.all(mx.where(kept, masked == grad, True)))


# ##################################################################
# topk mask over a tree
# the tree helper masks every tensor independently
def test_topk_mask_tree():
    grads = {"a": mx.random.normal((50, 10)), "b": mx.random.normal((20,))}
    masked = guards.topk_mask(grads, 0.5)
    for key, value in masked.items():
        frac = int((value != 0).sum()) / grads[key].size
        assert abs(frac - 0.5) < 0.06


# ##################################################################
# clip global rescales only when over budget
# a tree whose global norm exceeds the cap is rescaled to exactly the cap; a
# small tree is returned unchanged
def test_clip_global():
    big = {"w": mx.ones((10, 10)) * 5.0}
    clipped, norm = guards.clip_global(big, 1.0)
    assert norm > 1.0
    assert abs(guards.global_norm(clipped) - 1.0) < 1e-3
    small = {"w": mx.ones((2, 2)) * 0.01}
    unchanged, _ = guards.clip_global(small, 1.0)
    assert abs(guards.global_norm(unchanged) - guards.global_norm(small)) < 1e-6


# ##################################################################
# cap delta bounds the per-tensor move
# a large post-step move is rescaled to the cap; a small move is left alone
def test_cap_delta():
    before = {"w": mx.zeros((4, 4))}
    after = {"w": mx.ones((4, 4))}
    capped = guards.cap_delta(before, after, 0.1)
    move = float(mx.linalg.norm(capped["w"] - before["w"]))
    assert abs(move - 0.1) < 1e-4
    tiny_after = {"w": mx.ones((4, 4)) * 0.001}
    tiny = guards.cap_delta(before, tiny_after, 0.1)
    assert bool(mx.allclose(tiny["w"], tiny_after["w"]))


# ##################################################################
# pause flag trips on ceiling
# the flag raises with a reason above the ceiling and stays clear below it
def test_pause_flag():
    flag = guards.PauseFlag()
    assert flag.check(1.0, 5.0) is False
    assert flag.paused is False
    assert flag.check(6.0, 5.0) is True
    assert flag.paused is True
    assert "ceiling" in flag.reason
