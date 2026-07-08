# =============================================================================
#  surprise — the per-turn gate that decides which user turns carry signal
#  why: INDIVIDUATION.md §4 — the user's own tokens are ground truth. The model's
#  cross-entropy on the latest user message is its surprise; a rolling high
#  percentile is the adaptive threshold, so ~95% of already-predictable turns are
#  skipped and only the individuating residual reaches the learner. Locating the
#  user's content tokens is done from the END of the chat-templated sequence
#  because the qwen template's assistant rendering depends on the whole message
#  list, so a naive prefix of an earlier render is NOT a token-prefix of the full.
# =============================================================================
from __future__ import annotations

from collections import deque

import numpy as np

from engine.trace import Span


# ##################################################################
# special ids
# resolve the two turn markers to their single token ids and the length of the
# "user\n" role header that follows an opening marker, all for this tokenizer
def _special_ids(tokenizer) -> tuple[int, int, int]:
    im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    header = tokenizer.encode("user\n", add_special_tokens=False)
    return int(im_start[0]), int(im_end[0]), len(header)


# ##################################################################
# template
# render the full conversation to token ids with no generation prompt; the last
# user message is the final turn, so its content sits at the tail of this
def _template(tokenizer, messages: list) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, enable_thinking=True)
    return [int(t) for t in ids]


# ##################################################################
# last user span
# the content-token range of the final user message: the last opening marker
# begins its "user\n" header, the first closing marker after the content ends it
def _last_user_span(tokenizer, full: list[int]) -> tuple[int, int] | None:
    im_start, im_end, header_len = _special_ids(tokenizer)
    starts = [i for i, t in enumerate(full) if t == im_start]
    if not starts:
        return None
    content_start = starts[-1] + 1 + header_len
    later_ends = [i for i, t in enumerate(full) if t == im_end and i >= content_start]
    if not later_ends or later_ends[0] <= content_start:
        return None
    return content_start, later_ends[0]


# ##################################################################
# user message tokens
# full chat-templated sequence plus the [start,end) span of the latest user
# message's content; None when the last turn is not a user turn or is too short
def user_message_tokens(host, messages: list):
    if not messages or messages[-1].get("role") != "user":
        return None
    full = _template(host.tokenizer, messages)
    span = _last_user_span(host.tokenizer, full)
    if span is None:
        return None
    start, end = span
    if end - start < host.config.individuation.min_user_tokens:
        return None
    return full, (start, end)


# ##################################################################
# surprise
# mean cross-entropy (nats) the current model assigns to the user's content span
# given its prior context — the implicit label the user never states explicitly
def surprise(host, messages: list, config) -> float | None:
    packed = user_message_tokens(host, messages)
    if packed is None:
        return None
    full, (start, end) = packed
    logp = host.span_logprobs(full, Span("user", start, end), adapters_enabled=True)
    return float(-logp.mean())


# ##################################################################
# surprise gate
# adaptive rolling-percentile threshold over recent surprise values; fires only
# once warmed up and only for a value above the recent window's percentile. In
# memory for v1 (re-warms after a restart), pure and deterministic to test.
class SurpriseGate:
    def __init__(self, config) -> None:
        settings = config.individuation
        self.window: deque = deque(maxlen=settings.surprise_window)
        self.percentile = settings.surprise_percentile
        self.warmup = settings.surprise_warmup

    # ##################################################################
    # warm
    # true once enough values have been observed for the threshold to be trusted
    @property
    def warm(self) -> bool:
        return len(self.window) >= self.warmup

    # ##################################################################
    # threshold
    # the current rolling percentile of observed surprise, or None before any
    def threshold(self) -> float | None:
        if not self.window:
            return None
        return float(np.percentile(np.array(self.window), self.percentile * 100.0))

    # ##################################################################
    # consider
    # decide on this value against the window as it stood BEFORE observing it,
    # then record it; a warm gate fires only when the value beats the percentile
    def consider(self, value: float) -> bool:
        fired = self.warm and self._exceeds(value)
        self.window.append(value)
        return fired

    # ##################################################################
    # exceeds
    # strictly above the current rolling percentile (never fires on an empty one)
    def _exceeds(self, value: float) -> bool:
        threshold = self.threshold()
        return threshold is not None and value > threshold
