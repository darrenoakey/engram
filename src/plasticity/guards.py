# =============================================================================
#  guards — the bounds that keep an online update from collapsing the model
#  why: ROME/MEMIT-style edits blow up on norm; masked, clipped, capped updates
#  drift far less. These are pure array ops over the adapter gradient / param
#  trees plus a pause flag when total overlay norm exceeds its ceiling.
# =============================================================================
from __future__ import annotations

import mlx.core as mx


# ##################################################################
# mask one
# keep only the top `fraction` of a tensor's entries by magnitude, zero the
# rest. Threshold is the k-th largest |value| via sort (mx has no quantile)
def mask_one(grad: mx.array, fraction: float) -> mx.array:
    flat = mx.abs(grad).reshape(-1)
    keep = max(1, round(fraction * flat.size))
    ordered = mx.sort(flat)
    threshold = ordered[flat.size - keep]
    return mx.where(mx.abs(grad) >= threshold, grad, mx.zeros_like(grad))


# ##################################################################
# topk mask
# per-tensor top-k magnitude masking over a whole gradient tree; MoFO/FGGM show
# masked updates drift less, and this leaves (1-fraction) of each grad exactly 0
def topk_mask(grads: dict, fraction: float) -> dict:
    return {key: mask_one(value, fraction) for key, value in grads.items()}


# ##################################################################
# global norm
# L2 norm over every tensor in a tree (treated as one flat vector)
def global_norm(grads: dict) -> float:
    total = 0.0
    for value in grads.values():
        total += float((value.astype(mx.float32) ** 2).sum())
    return total ** 0.5


# ##################################################################
# clip global
# rescale the whole gradient tree so its global L2 norm never exceeds max_norm
def clip_global(grads: dict, max_norm: float) -> tuple[dict, float]:
    norm = global_norm(grads)
    if norm <= max_norm or norm == 0.0:
        return grads, norm
    factor = max_norm / norm
    return {key: value * factor for key, value in grads.items()}, norm


# ##################################################################
# cap delta
# per-tensor post-step delta cap: if a param moved further than `cap` in this
# update, rescale its move back onto the cap sphere (keeps direction)
def cap_delta(before: dict, after: dict, cap: float) -> dict:
    capped: dict = {}
    for key, new_value in after.items():
        move = new_value - before[key]
        norm = float(mx.linalg.norm(move.astype(mx.float32)))
        if norm > cap and norm > 0.0:
            move = move * (cap / norm)
        capped[key] = before[key] + move
    return capped


# ##################################################################
# pause flag
# raised when total overlay norm passes the ceiling; the server surfaces this
# in /v1/brain and stops accepting updates until consolidation resets deltas
class PauseFlag:
    def __init__(self) -> None:
        self.paused = False
        self.reason: str | None = None

    # ##################################################################
    # check
    # set (or clear) the pause based on the overlay's current total norm
    def check(self, total_norm: float, ceiling: float) -> bool:
        if total_norm > ceiling:
            self.paused = True
            self.reason = f"overlay total_norm {total_norm:.4f} exceeds ceiling {ceiling:.4f}"
        return self.paused
