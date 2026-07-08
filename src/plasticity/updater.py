# =============================================================================
#  updater — the guarded online update pipeline (DESIGN.md §4)
#  why: this is where inference outcomes become weight changes. Every step is
#  snapshotted, gradient-masked, clipped, delta-capped and KL-gated so a single
#  bad signal can't damage the model — a breach restores the snapshot. Accepts
#  the raw mlx model + overlay + token ids (no engine coupling); the server
#  wires stored traces to this signature.
# =============================================================================
from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.trainer import grad_checkpoint

from plasticity import guards, losses

REPLAY_REWARD = 0.3


# ##################################################################
# span
# one teacher-forced training window: which input prefix to forward, which
# logit range predicts which targets, and how it should be scored
@dataclass
class Span:
    input_ids: list
    lo: int
    hi: int
    targets: mx.array
    reward: float
    negative: bool


# ##################################################################
# update report
# the numeric record of one update attempt (journaled and returned)
@dataclass
class UpdateReport:
    kind: str
    reward: float
    span_tokens: int
    loss: float
    grad_norm: float
    delta_norm: float
    span_kl: float
    accepted: bool
    wall_ms: float


# ##################################################################
# updater
# holds config and a fresh AdamW (no optimizer-state persistence per boot);
# apply() runs the whole guarded pipeline under the caller's gpu lock
class Updater:
    def __init__(self, config) -> None:
        self.config = config
        self.optimizer = optim.AdamW(learning_rate=config.lr_reward)

    # ##################################################################
    # apply
    # snapshot -> grads over spans -> guard -> step -> delta cap -> KL gate.
    # a KL breach restores the snapshot; either way a report is journaled
    def apply(self, model, overlay, token_ids, gen_start, credit_spans, reward, kind, replay_buffer, journal):
        start = time.time()
        effective_reward = REPLAY_REWARD if kind == "reinforce" else reward
        spans = self._build_spans(token_ids, gen_start, credit_spans, effective_reward, replay_buffer)
        if not spans:
            return self._empty_report(kind, effective_reward, start)
        snapshot = overlay.snapshot()
        pre_logits, bases = self._eval_logits(model, overlay, spans)
        loss, grad_norm = self._grad_step(model, overlay, spans, bases, kind)
        delta_norm = self._cap(overlay, snapshot)
        return self._finalize(model, overlay, spans, snapshot, pre_logits, journal, kind, effective_reward, loss,
                              grad_norm, delta_norm, start)

    # ##################################################################
    # build spans
    # primary credit spans (newest first, capped) plus one positive replay span
    # per config, each accumulated sequentially — never concatenated
    def _build_spans(self, token_ids, gen_start, credit_spans, reward, replay_buffer):
        cap = self.config.max_span_tokens
        spans: list[Span] = []
        for start, end in credit_spans:
            start = max(gen_start, start, end - cap, 1)
            if end - start <= 0:
                continue
            targets = mx.array(token_ids[start:end])
            spans.append(Span(token_ids[:end], start - 1, end - 1, targets, reward, reward < 0))
        spans.extend(self._replay_spans(replay_buffer, cap))
        return spans

    # ##################################################################
    # replay spans
    # sample positive spans from the buffer and teacher-force each as a whole
    # standalone sequence (predict token t from t-1) with a fixed positive reward
    def _replay_spans(self, replay_buffer, cap):
        spans: list[Span] = []
        if replay_buffer is None:
            return spans
        for tokens in replay_buffer.sample(self.config.replay_spans):
            tokens = tokens[:cap]
            if len(tokens) < 2:
                continue
            targets = mx.array(tokens[1:])
            spans.append(Span(tokens, 0, len(tokens) - 1, targets, REPLAY_REWARD, False))
        return spans

    # ##################################################################
    # eval logits
    # capture pre-step live logits of the primary span (for the post-step KL
    # gate) and the overlay-disabled base logits of every span (for kl_anchor),
    # all in eval mode with gradients stopped
    def _eval_logits(self, model, overlay, spans):
        model.eval()
        primary = spans[0]
        pre_logits = mx.stop_gradient(self._span_logits(model, primary))
        bases = []
        with overlay.disabled():
            for span in spans:
                bases.append(mx.stop_gradient(self._span_logits(model, span)))
        mx.eval(pre_logits, *bases)
        return pre_logits, bases

    # ##################################################################
    # span logits
    # teacher-forced forward of one span's input prefix, sliced to the logit
    # range that predicts its targets, in fp32
    def _span_logits(self, model, span: Span) -> mx.array:
        logits = model(mx.array(span.input_ids)[None])[0]
        return logits[span.lo:span.hi].astype(mx.float32)

    # ##################################################################
    # grad step
    # train-mode grad accumulation across spans with layer grad-checkpointing,
    # then mask/clip and one AdamW step; returns primary loss and pre-clip norm
    def _grad_step(self, model, overlay, spans, bases, kind):
        cls, original = _checkpoint_layers(model)
        model.train()
        try:
            loss, accum = self._accumulate(model, spans, bases)
        finally:
            cls.__call__ = original
            model.eval()
        masked = guards.topk_mask(accum, self.config.topk_grad_fraction)
        clipped, grad_norm = guards.clip_global(masked, self.config.grad_clip_norm)
        self.optimizer.learning_rate = self._learning_rate(kind)
        self.optimizer.update(model, tree_unflatten(list(clipped.items())))
        mx.eval(model.parameters(), self.optimizer.state)
        return loss, grad_norm

    # ##################################################################
    # learning rate
    # each update kind has its own step size: gentle self-reinforcement, a
    # stronger reward step, and the user-token absorb step (individuation)
    def _learning_rate(self, kind: str) -> float:
        rates = {"reinforce": self.config.lr_reinforce, "reward": self.config.lr_reward,
                 "absorb": self.config.lr_absorb}
        return rates.get(kind, self.config.lr_reward)

    # ##################################################################
    # accumulate
    # sum per-span gradients into one flat tree; the first span's loss is the
    # reported loss (it is the actual credit span, replay spans are auxiliary)
    def _accumulate(self, model, spans, bases):
        accum: dict = {}
        primary_loss = 0.0
        for index, span in enumerate(spans):
            loss, grads = self._span_grads(model, span, bases[index])
            for key, value in tree_flatten(grads):
                accum[key] = value if key not in accum else accum[key] + value
            if index == 0:
                primary_loss = float(loss)
        return primary_loss, accum

    # ##################################################################
    # span grads
    # value_and_grad over overlay params only: class loss (positive or bounded
    # unlikelihood) plus the KL-to-base anchor on the same span
    def _span_grads(self, model, span: Span, base_logits: mx.array):
        weights = mx.ones(span.targets.shape[0])

        def loss_fn(_model):
            logits = self._span_logits(_model, span)
            if span.negative:
                cls = losses.negative_loss(logits, span.targets, weights, span.reward, self.config.lambda_neg)
            else:
                cls = losses.positive_loss(logits, span.targets, weights, span.reward)
            return cls + self.config.beta_kl * losses.kl_anchor(logits, base_logits)

        return nn.value_and_grad(model, loss_fn)(model)

    # ##################################################################
    # cap
    # per-tensor post-step delta cap; returns the global norm of the actual
    # (capped) move applied to the overlay
    def _cap(self, overlay, snapshot) -> float:
        after = overlay.trainable_parameters()
        mx.eval(list(after.values()))
        capped = guards.cap_delta(snapshot, after, self.config.delta_frobenius_cap)
        overlay.restore(capped)
        mx.eval(list(capped.values()))
        return guards.global_norm({k: capped[k] - snapshot[k] for k in capped})

    # ##################################################################
    # finalize
    # post-step KL gate against the pre-step distribution: within budget the
    # update is journaled and kept; over budget it is restored and rejected
    def _finalize(self, model, overlay, spans, snapshot, pre_logits, journal, kind, reward, loss,
                  grad_norm, delta_norm, start):
        model.eval()
        post = self._span_logits(model, spans[0])
        span_kl = float(losses.kl_anchor(post, pre_logits))
        accepted = span_kl <= self.config.update_kl_budget
        if not accepted:
            overlay.restore(snapshot)
            mx.eval(list(snapshot.values()))
        report = UpdateReport(kind, reward, int(spans[0].targets.shape[0]), loss, grad_norm, delta_norm,
                              span_kl, accepted, (time.time() - start) * 1000.0)
        journal.record("update" if accepted else "rejected_update", **vars(report))
        return report

    # ##################################################################
    # empty report
    # nothing creditable in this trace: record a rejected no-op
    def _empty_report(self, kind, reward, start) -> UpdateReport:
        return UpdateReport(kind, reward, 0, 0.0, 0.0, 0.0, 0.0, False, (time.time() - start) * 1000.0)


# ##################################################################
# checkpoint layers
# enable gradient checkpointing on the decoder layer class for the duration of
# one update, returning the class and its original __call__ so it can be undone
def _checkpoint_layers(model):
    layer = model.layers[0]
    cls = type(layer)
    original = cls.__call__
    grad_checkpoint(layer)
    return cls, original
