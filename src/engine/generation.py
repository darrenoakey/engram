# =============================================================================
#  generation — prompt building, streamed decoding, and token-level span parsing
#  why: keeps ModelHost thin; span parsing works on atomic marker token ids
#  (verified single-token in qwen3_5), so no fragile char-offset mapping
# =============================================================================
from __future__ import annotations

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from engine.trace import Span

_MARKER_TEXTS = (
    ("think_open", "<think>"),
    ("think_close", "</think>"),
    ("tool_open", "<tool_call>"),
    ("tool_close", "</tool_call>"),
)


# =============================================================================
#  marker ids
#  why: resolve the four span markers to their token ids once; each must be a
#  single atomic token or token-level span parsing is impossible — fail loud
def marker_ids(tokenizer) -> dict:
    resolved: dict = {}
    for label, text in _MARKER_TEXTS:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"marker {text!r} is not a single token: {encoded}")
        resolved[label] = int(encoded[0])
    return resolved


# =============================================================================
#  build prompt
#  why: the qwen3_5 chat template renders tools natively and, with thinking on,
#  injects the opening <think> into the prompt itself
def build_prompt(tokenizer, messages: list, tools: list | None) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages, tools=tools or None, add_generation_prompt=True, enable_thinking=True
    )
    return [int(t) for t in ids]


# =============================================================================
#  starts in think
#  why: generation begins inside whichever block the prompt left open; the last
#  think marker in the prompt decides whether token 0 of the output is thinking
def starts_in_think(prompt_ids: list[int], markers: dict) -> bool:
    for token in reversed(prompt_ids):
        if token == markers["think_open"]:
            return True
        if token == markers["think_close"]:
            return False
    return False


def build_sampler(sampling):
    return make_sampler(temp=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k)


# =============================================================================
#  stream generation
#  why: generate_step hands back the full log-softmax per step for free; record
#  only the chosen-token logprob, decode deltas for the on_token callback
def stream_generation(model, tokenizer, prompt_ids, sampler, max_tokens, eos_ids, on_token):
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    gen_ids: list[int] = []
    logprobs: list[float] = []
    for token, step_logprobs in generate_step(mx.array(prompt_ids), model, max_tokens=max_tokens, sampler=sampler):
        token_id = int(token)
        if token_id in eos_ids:
            break
        gen_ids.append(token_id)
        logprobs.append(float(step_logprobs[token_id]))
        detokenizer.add_token(token_id)
        _emit(on_token, detokenizer.last_segment)
    detokenizer.finalize()
    _emit(on_token, detokenizer.last_segment)
    return gen_ids, logprobs


def _emit(on_token, segment: str) -> None:
    if on_token is not None and segment:
        on_token(segment)


# =============================================================================
#  parse spans
#  why: partition the generated region into think/answer/tool_call spans so the
#  updater can credit answer + tool_call tokens only; offset maps to full ids
def parse_spans(gen_ids: list[int], markers: dict, start_in_think: bool, offset: int = 0) -> list[Span]:
    return _coalesce(_classify(gen_ids, markers, start_in_think), offset)


def _classify(gen_ids: list[int], markers: dict, start_in_think: bool) -> list[str]:
    kinds: list[str] = []
    state = "think" if start_in_think else "answer"
    for token in gen_ids:
        state, kind = _advance(state, token, markers)
        kinds.append(kind)
    return kinds


# =============================================================================
#  advance
#  why: one state step of the span machine; the closing/opening marker token
#  belongs to the block it delimits, and the transition applies to the next one
def _advance(state: str, token: int, markers: dict) -> tuple[str, str]:
    if state == "think":
        return ("answer" if token == markers["think_close"] else "think"), "think"
    if state == "tool_call":
        return ("answer" if token == markers["tool_close"] else "tool_call"), "tool_call"
    if token == markers["think_open"]:
        return "think", "think"
    if token == markers["tool_open"]:
        return "tool_call", "tool_call"
    return "answer", "answer"


def _coalesce(kinds: list[str], offset: int) -> list[Span]:
    spans: list[Span] = []
    index = 0
    while index < len(kinds):
        end = index
        while end < len(kinds) and kinds[end] == kinds[index]:
            end += 1
        spans.append(Span(kinds[index], offset + index, offset + end))
        index = end
    return spans
