# =============================================================================
#  losses — the online update objective (DESIGN.md §4)
#  why: batch=1 online learning needs reward-scaled cross-entropy for the
#  positive case and bounded, self-saturating unlikelihood for the negative
#  case, both anchored by a KL term to the base distribution so a single
#  update can't drift the model. All math in float32 for numerical safety.
# =============================================================================
from __future__ import annotations

import mlx.core as mx


# ##################################################################
# log softmax
# stable per-row log-softmax used everywhere so probabilities are computed
# once, in fp32
def log_softmax(logits: mx.array) -> mx.array:
    x = logits.astype(mx.float32)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


# ##################################################################
# log1mexp
# numerically stable log(1 - e^x) for x <= 0: the two-branch trick avoids
# catastrophic cancellation near x=0 and underflow for very negative x
def log1mexp(x: mx.array) -> mx.array:
    x = mx.minimum(x.astype(mx.float32), -1e-7)
    near_zero = mx.log(-mx.expm1(x))
    far = mx.log1p(-mx.exp(x))
    return mx.where(x > -0.6931471805599453, near_zero, far)


# ##################################################################
# gather logprob
# per-position log p(target) over a span given logits and target ids
def gather_logprob(logits: mx.array, targets: mx.array) -> mx.array:
    logp = log_softmax(logits)
    return mx.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]


# ##################################################################
# weighted mean
# span-masked mean of a per-position quantity with per-token weights w_t
def weighted_mean(values: mx.array, weights: mx.array) -> mx.array:
    weights = weights.astype(mx.float32)
    return (values * weights).sum() / mx.maximum(weights.sum(), 1e-8)


# ##################################################################
# positive loss
# reward-scaled cross-entropy: pushes p(target) up, scaled by how good the
# outcome was. -reward * mean_w log p(y)
def positive_loss(logits: mx.array, targets: mx.array, weights: mx.array, reward: float) -> mx.array:
    logp = gather_logprob(logits, targets)
    return -reward * weighted_mean(logp, weights)


# ##################################################################
# negative loss
# bounded unlikelihood: pushes p(target) down via -log(1-p), scaled by
# |reward| * lambda_neg. Self-saturating — gradient vanishes as p -> 0, so it
# can't displace mass onto unrelated tokens (the squeezing pathology)
def negative_loss(logits: mx.array, targets: mx.array, weights: mx.array, reward: float, lambda_neg: float) -> mx.array:
    logp = gather_logprob(logits, targets)
    return -abs(reward) * lambda_neg * weighted_mean(log1mexp(logp), weights)


# ##################################################################
# kl anchor
# mean over span positions of KL(live || base) across the full vocab; bounds
# how far one update moves the distribution from the frozen base
def kl_anchor(live_logits: mx.array, base_logits: mx.array) -> mx.array:
    logp_live = log_softmax(live_logits)
    logp_base = log_softmax(base_logits)
    p_live = mx.exp(logp_live)
    per_position = (p_live * (logp_live - logp_base)).sum(axis=-1)
    return per_position.mean()
