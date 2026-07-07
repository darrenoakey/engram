# =============================================================================
#  canary — cumulative-drift detector for the live model vs the original base
#  why: consolidations fold the plastic overlay into the base weights again and
#  again; each fold can nudge behavior, and the nudges accumulate. A full-vocab
#  logit baseline is far too large to store, so at first boot we store a TRUNCATED
#  reference: for each probe we greedy-generate a short continuation with the
#  overlay DISABLED (the original base's own words), and record per position the
#  top-K token ids + fp32 logprobs of the base's predictive distribution. At probe
#  time we teacher-force the SAME continuation through the LIVE model (overlay
#  enabled) and compute per-position KL restricted to the stored top-K support:
#  renormalize the stored top-K to a proper distribution, gather+renormalize the
#  live logprobs at those same K ids, KL(stored || live), mean over positions then
#  prompts. Restricting to the top-K support is an APPROXIMATION of the true full
#  KL — mass the base put outside its own top-K is ignored — but it captures the
#  drift that matters (where the base was confident) at a tiny storage cost. Pure
#  evaluation: no journaling here, the server records the report.
# =============================================================================
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace

import mlx.core as mx

from common import store
from evaluation.canary_prompts import EXPECTED, PROBES

TOP_K = 64
CONTINUATION_TOKENS = 32


# =============================================================================
#  canary report
#  why: the single value the guard rail reads (mean_kl), the count that hard-gates
#  a clean verdict (match_failures), and the per-prompt breakdown for the journal
@dataclass
class CanaryReport:
    mean_kl: float
    per_prompt: list = field(default_factory=list)
    match_failures: int = 0


# =============================================================================
#  overlay off
#  why: base-reference forwards and baseline generation must exclude the plastic
#  delta; when no overlay is attached yet (host.overlay is None) the base model
#  is already the plain forward, so the context is a no-op
@contextlib.contextmanager
def _overlay_off(host):
    if host.overlay is None:
        yield
    else:
        with host.overlay.disabled():
            yield


def _greedy(sampling):
    return replace(sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=CONTINUATION_TOKENS)


# =============================================================================
#  greedy generate
#  why: one deterministic ≤32-token continuation; disable_overlay picks the base
#  (baseline reference) or the live model as-is (probe-time expected-match check)
def _greedy_generate(host, messages, disable_overlay: bool):
    sampling = _greedy(host.config.sampling)
    if disable_overlay:
        with _overlay_off(host):
            return host.generate(messages, sampling=sampling)
    return host.generate(messages, sampling=sampling)


# =============================================================================
#  logp rows
#  why: teacher-forced log-softmax (fp32) of the positions that predict each
#  continuation token — logit row i-1 predicts the token at index i
def _logp_rows(model, token_ids: list[int], gen_start: int, end: int) -> mx.array:
    logits = model(mx.array([token_ids[:end]]))[0].astype(mx.float32)
    rows = logits[gen_start - 1 : end - 1]
    return rows - mx.logsumexp(rows, axis=-1, keepdims=True)


# =============================================================================
#  top k
#  why: keep only the K most probable ids per position (argpartition leaves the K
#  largest in the tail slice, unsorted — order is irrelevant for a set-based KL)
def _top_k(logp_rows: mx.array, k: int) -> tuple[mx.array, mx.array]:
    part = mx.argpartition(logp_rows, kth=logp_rows.shape[-1] - k, axis=-1)
    ids = part[:, -k:]
    vals = mx.take_along_axis(logp_rows, ids, axis=-1)
    return ids, vals


# =============================================================================
#  capture record
#  why: the stored baseline for one probe — the base's own continuation plus its
#  top-K predictive distribution per position, all overlay-disabled
def _capture_record(host, pid: str, messages: list[dict]) -> dict:
    trace = _greedy_generate(host, messages, disable_overlay=True)
    end = len(trace.token_ids)
    record = {"id": pid, "token_ids": trace.token_ids, "gen_start": trace.gen_start,
              "topk_ids": [], "topk_logp": []}
    if end - trace.gen_start < 1:
        return record
    with host.gpu_lock:
        host.model.eval()
        with _overlay_off(host):
            rows = _logp_rows(host.model, trace.token_ids, trace.gen_start, end)
        ids, vals = _top_k(rows, TOP_K)
        mx.eval(ids, vals)
    record["topk_ids"] = ids.tolist()
    record["topk_logp"] = vals.tolist()
    return record


def _record_path(pid: str):
    return store.canary_dir() / f"{pid}.json.gz"


def _write_index(ids: list[str]) -> None:
    payload = {"ids": ids, "top_k": TOP_K, "continuation_tokens": CONTINUATION_TOKENS}
    store.atomic_write_json(store.canary_dir() / "index.json", payload)


# =============================================================================
#  baseline
#  why: capture the reference state on first boot (base model, overlay disabled),
#  one gzipped record per probe plus an index; overwrites existing files so a
#  re-baseline is idempotent-safe. Returns a small summary dict for the caller.
def baseline(host, probes=None) -> dict:
    probes = probes if probes is not None else PROBES
    ids: list[str] = []
    for pid, messages in probes:
        record = _capture_record(host, pid, messages)
        store.write_json_gz(_record_path(pid), record)
        ids.append(pid)
    _write_index(ids)
    return {"ids": ids, "top_k": TOP_K, "continuation_tokens": CONTINUATION_TOKENS}


# =============================================================================
#  live rows
#  why: teacher-force the stored continuation through the LIVE model (overlay
#  as-is) under the GPU lock, returning the fp32 log-softmax rows to gather from
def _live_rows(host, token_ids: list[int], gen_start: int, end: int) -> mx.array:
    with host.gpu_lock:
        host.model.eval()
        rows = _logp_rows(host.model, token_ids, gen_start, end)
        mx.eval(rows)
    return rows


# =============================================================================
#  truncated kl
#  why: KL(stored || live) over the stored top-K support only — both sides are
#  renormalized to proper distributions over those K ids, then averaged over
#  positions. This is the top-K approximation of the full-vocab KL (see header).
def _truncated_kl(stored_ids: list, stored_logp: list, live_rows: mx.array) -> float:
    ids = mx.array(stored_ids)
    stored = mx.array(stored_logp)
    stored = stored - mx.logsumexp(stored, axis=-1, keepdims=True)
    live = mx.take_along_axis(live_rows, ids, axis=-1)
    live = live - mx.logsumexp(live, axis=-1, keepdims=True)
    per_position = (mx.exp(stored) * (stored - live)).sum(axis=-1)
    return float(per_position.mean())


# =============================================================================
#  record kl
#  why: the drift of one probe — teacher-force its stored continuation live and
#  compare to the stored base top-K; an empty continuation contributes zero drift
def _record_kl(host, record: dict) -> float:
    token_ids = record["token_ids"]
    gen_start = record["gen_start"]
    end = len(token_ids)
    if end - gen_start < 1 or not record["topk_ids"]:
        return 0.0
    live = _live_rows(host, token_ids, gen_start, end)
    return _truncated_kl(record["topk_ids"], record["topk_logp"], live)


# =============================================================================
#  match expected
#  why: coarse behavioral check for an answer-bearing probe — greedy-generate
#  under the live model and confirm the expected substring still appears
def _match_expected(host, pid: str, messages: list[dict]) -> bool:
    trace = _greedy_generate(host, messages, disable_overlay=False)
    text = host.tokenizer.decode(trace.token_ids[trace.gen_start :])
    return EXPECTED[pid].lower() in text.lower()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


# =============================================================================
#  probe
#  why: measure current drift and answer-fidelity against the stored baseline —
#  per-prompt truncated KL plus, for the 12 EXPECTED prompts, a live substring
#  match. Returns the report the server journals and the guard rail reads.
def probe(host, probes=None) -> CanaryReport:
    probes = probes if probes is not None else PROBES
    per_prompt: list[dict] = []
    for pid, messages in probes:
        record = store.read_json_gz(_record_path(pid))
        kl = _record_kl(host, record)
        matched = _match_expected(host, pid, messages) if pid in EXPECTED else None
        per_prompt.append({"id": pid, "kl": kl, "matched": matched})
    mean_kl = _mean([entry["kl"] for entry in per_prompt])
    failures = sum(1 for entry in per_prompt if entry["matched"] is False)
    return CanaryReport(mean_kl=mean_kl, per_prompt=per_prompt, match_failures=failures)


# =============================================================================
#  is clean
#  why: the guard rail's verdict — drift within the configured KL budget AND no
#  expected answer lost; a single lost answer fails the whole probe
def is_clean(report: CanaryReport, guards_config) -> bool:
    return report.mean_kl <= guards_config.canary_kl_budget and report.match_failures == 0
