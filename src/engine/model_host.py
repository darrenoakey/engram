# =============================================================================
#  model_host — the single owner of the MLX model, GPU lock, and eval/train mode
#  why: all Metal work funnels through one lock so generation, probes, and
#  updates never overlap; generation runs eval mode (DeltaNet fast kernel),
#  and teacher-forced logprobs are the measuring stick for probes and KL
# =============================================================================
from __future__ import annotations

import threading

import mlx.core as mx
from mlx_lm import load

from engine import generation
from engine.trace import Span, Trace


def _sampling_dict(sampling) -> dict:
    return {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "max_tokens": sampling.max_tokens,
    }


# =============================================================================
#  model host
#  why: loads Ornith/the test model once, holds the plastic overlay the server
#  attaches later, and serializes every Metal call behind gpu_lock
class ModelHost:
    def __init__(self, config, model_path: str):
        self.config = config
        self.model, self.tokenizer = load(model_path)
        self.model.eval()
        self.overlay = None
        self.gpu_lock = threading.Lock()
        self.markers = generation.marker_ids(self.tokenizer)
        self.eos_ids = set(self.tokenizer.eos_token_ids)

    # =========================================================================
    #  generate
    #  why: one turn of inference recorded as a Trace; eval mode keeps DeltaNet
    #  on its fast kernel, and the lock guarantees exclusive GPU access
    def generate(self, messages, tools=None, sampling=None, on_token=None, enable_thinking=True) -> Trace:
        sampling = sampling if sampling is not None else self.config.sampling
        with self.gpu_lock:
            return self._generate_locked(messages, tools, sampling, on_token, enable_thinking)

    def _generate_locked(self, messages, tools, sampling, on_token, enable_thinking) -> Trace:
        self.model.eval()
        prompt_ids = generation.build_prompt(self.tokenizer, messages, tools, enable_thinking)
        in_think = generation.starts_in_think(prompt_ids, self.markers)
        sampler = generation.build_sampler(sampling)
        gen_ids, logprobs = generation.stream_generation(
            self.model, self.tokenizer, prompt_ids, sampler, sampling.max_tokens, self.eos_ids, on_token
        )
        spans = generation.parse_spans(gen_ids, self.markers, in_think, offset=len(prompt_ids))
        return Trace.create(prompt_ids + gen_ids, len(prompt_ids), logprobs, spans, _sampling_dict(sampling))

    # =========================================================================
    #  span logprobs
    #  why: teacher-forced eval-mode forward over a stored prefix+span, returning
    #  the chosen-token logprob per position; adapters_enabled toggles the overlay
    #  when one is attached (server wiring), otherwise it is ignored
    def span_logprobs(self, token_ids: list[int], span: Span, adapters_enabled: bool = True) -> mx.array:
        with self.gpu_lock:
            self.model.eval()
            if self.overlay is not None and not adapters_enabled:
                with self.overlay.disabled():
                    return self._chosen_logprobs(token_ids, span)
            return self._chosen_logprobs(token_ids, span)

    # =========================================================================
    #  chosen logprobs
    #  why: logit at position t predicts token t+1, so the logprob of the token
    #  at index i comes from row i-1; log_softmax is computed in float32
    def _chosen_logprobs(self, token_ids: list[int], span: Span) -> mx.array:
        prefix = mx.array([token_ids[: span.end]])
        logits = self.model(prefix)[0].astype(mx.float32)
        rows = logits[span.start - 1 : span.end - 1]
        logp = rows - mx.logsumexp(rows, axis=-1, keepdims=True)
        targets = mx.array(token_ids[span.start : span.end])
        chosen = mx.take_along_axis(logp, targets[:, None], axis=1)[:, 0]
        mx.eval(chosen)
        return chosen
