# =============================================================================
#  losses_test — real numeric checks of the update objective
#  why: these formulas are load-bearing (sign, saturation, KL zero point);
#  verify them against a numpy reference and against their intended behaviour
# =============================================================================
from __future__ import annotations

import mlx.core as mx
import numpy as np

from plasticity import losses


# ##################################################################
# log1mexp matches reference
# the stable log(1-e^x) must equal the naive numpy computation across both
# branches (near zero and far negative), including around the log(0.5) split
def test_log1mexp_matches_reference():
    xs = np.array([-0.001, -0.05, -0.3, -0.6931, -0.7, -1.0, -3.0, -8.0], dtype=np.float32)
    got = np.array(losses.log1mexp(mx.array(xs)))
    ref = np.log1p(-np.exp(xs))
    assert np.allclose(got, ref, atol=1e-4), f"{got} vs {ref}"


# ##################################################################
# positive loss rewards confidence
# raising p(target) must lower the reward-scaled cross-entropy loss
def test_positive_loss_rewards_confidence():
    targets = mx.array([0, 1, 2])
    weights = mx.ones(3)
    confident = mx.array([[9.0, 0, 0, 0], [0, 9.0, 0, 0], [0, 0, 9.0, 0]])
    unsure = mx.zeros((3, 4))
    loss_conf = float(losses.positive_loss(confident, targets, weights, 1.0))
    loss_unsure = float(losses.positive_loss(unsure, targets, weights, 1.0))
    assert loss_conf < loss_unsure
    assert loss_conf > 0.0


# ##################################################################
# negative loss saturates
# unlikelihood is ~0 when the punished token is already unlikely and large when
# it is likely — the bounded, self-saturating behaviour DESIGN relies on
def test_negative_loss_saturates():
    targets = mx.array([0])
    weights = mx.ones(1)
    likely = mx.array([[9.0, 0.0, 0.0, 0.0]])
    unlikely = mx.array([[-9.0, 3.0, 3.0, 3.0]])
    loss_likely = float(losses.negative_loss(likely, targets, weights, -0.5, 0.5))
    loss_unlikely = float(losses.negative_loss(unlikely, targets, weights, -0.5, 0.5))
    assert loss_likely > loss_unlikely
    assert loss_unlikely >= 0.0
    assert loss_unlikely < 0.05


# ##################################################################
# kl anchor zero at identity
# KL to an identical distribution is zero; a shifted distribution is positive
def test_kl_anchor_zero_and_positive():
    logits = mx.array([[2.0, 0.5, -1.0, 0.0], [1.0, 1.0, 1.0, 3.0]])
    other = logits + mx.array([[3.0, -1.0, 0.0, 0.0], [0.0, 2.0, -1.0, 0.0]])
    assert abs(float(losses.kl_anchor(logits, logits))) < 1e-5
    assert float(losses.kl_anchor(logits, other)) > 0.0
