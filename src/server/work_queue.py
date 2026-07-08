# =============================================================================
#  work_queue — the single background update worker (DESIGN.md §4 server/)
#  why: inference must always win the GPU, so exactly one worker thread applies
#  queued updates and only claims the host lock when no chat request is waiting.
#  It reads live state fields (overlay, updater) so a consolidation swap is
#  picked up automatically, and every canary path is gated so unit tests that
#  set a huge canary_every never touch the concurrently-built evaluation package.
# =============================================================================
from __future__ import annotations

import queue
import threading

import mlx.core as mx

from engine.trace import Span, Trace
from individuation.experience import Experience

CANARY_DISABLED_THRESHOLD = 10 ** 9


# ##################################################################
# canary enabled
# a canary_every at or above the disabled threshold means "no canary at all";
# tests use this to keep the concurrently-written evaluation package out
def canary_enabled(config) -> bool:
    return config.guards.canary_every < CANARY_DISABLED_THRESHOLD


# ##################################################################
# materialize
# force every model/overlay tensor to a concrete array. mlx builds arrays lazily
# and binds the unevaluated graph to the creating thread; a forward run on a
# different thread (uvicorn's worker pool, this worker) then faults with "no
# Stream in current thread". Evaluating after any weight reassignment keeps the
# tensors thread-agnostic so every thread can read them
def materialize(host) -> None:
    mx.eval(host.model.parameters())


# ##################################################################
# in flight
# a tiny counter of chat requests currently in the handler; the worker refuses
# to claim the GPU while this is non-zero so generation always has priority
class InFlight:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0

    def enter(self) -> None:
        with self._lock:
            self._count += 1

    def leave(self) -> None:
        with self._lock:
            self._count -= 1

    def waiting(self) -> int:
        with self._lock:
            return self._count


# ##################################################################
# work queue
# owns the job queue and the worker thread; jobs are dicts
# {kind, trace_id, reward, source}. Fairness, graceful hold for consolidation,
# checkpoint cadence and canary/rollback bookkeeping all live here
class WorkQueue:
    def __init__(self, state) -> None:
        self.state = state
        self.jobs: queue.Queue = queue.Queue()
        self.accepted_updates = 0
        self.consecutive_breaches = 0
        self.last_clean = True
        self._stop = threading.Event()
        self._held = threading.Event()
        self._busy = threading.Event()
        self.thread = threading.Thread(target=self._run, name="engram-worker", daemon=True)

    # ##################################################################
    # lifecycle
    # start/stop the worker and enqueue/measure jobs
    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def enqueue(self, job: dict) -> None:
        self.jobs.put(job)

    def depth(self) -> int:
        return self.jobs.qsize()

    # ##################################################################
    # hold / release
    # consolidation quiesces the worker: hold stops it claiming new jobs and
    # waits for any in-flight job to finish; release lets it resume
    def hold(self) -> None:
        self._held.set()
        self.wait_idle()

    def release(self) -> None:
        self._held.clear()

    def wait_idle(self) -> None:
        while self._busy.is_set():
            threading.Event().wait(0.02)

    # ##################################################################
    # run
    # the worker loop: pull a job, wait until it is fair to claim the GPU, then
    # process it under the host lock. A failing job must NEVER kill this thread
    # (the learning loop would die silently while the queue grows) — it is
    # journaled loudly as worker_error and the loop continues
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._claim_and_process(job)
            except Exception as error:
                self.state.journal.record("worker_error", job=job, error=repr(error))

    # ##################################################################
    # claim and process
    # block (never busy-spin) while consolidation holds the worker or any chat
    # request is in flight, then process exactly one job
    def _claim_and_process(self, job: dict) -> None:
        while (self._held.is_set() or self.state.in_flight.waiting() > 0) and not self._stop.is_set():
            threading.Event().wait(0.05)
        if self._stop.is_set():
            return
        self._busy.set()
        try:
            self._process(job)
        finally:
            self._busy.clear()

    # ##################################################################
    # process
    # load the trace, build its credit spans, and run the guarded update under
    # the host GPU lock; accepted updates trigger the post-update bookkeeping.
    # A trace with no creditable span (a truly empty generation) is journaled as
    # skipped_update, never silently dropped — a lost reward that leaves no record
    # would hang any caller draining the queue and hide feedback that did nothing
    def _process(self, job: dict) -> None:
        if job["kind"] == "absorb_candidate":
            self._process_candidate(job)
            return
        if "token_ids" in job:
            self._process_direct(job)
            return
        trace = Trace.load(job["trace_id"])
        credit_spans = self._credit_spans(trace)
        if not credit_spans:
            self.state.journal.record("skipped_update", trace_id=job["trace_id"], kind=job["kind"],
                                      reward=job["reward"], reason="no creditable span")
            return
        with self.state.host.gpu_lock:
            report = self.state.updater.apply(
                self.state.host.model, self.state.overlay, trace.token_ids, trace.gen_start,
                credit_spans, job["reward"], job["kind"], self.state.replay, self.state.journal,
            )
        if report.accepted:
            self._after_accept(trace, credit_spans, report, job)

    # ##################################################################
    # process direct
    # an absorb job carries its own token_ids + credit span (the USER's tokens,
    # or a self-edit answer span) rather than a trace of the model's own output —
    # this is how engram trains on what the user said, not on what it replied
    def _process_direct(self, job: dict) -> None:
        with self.state.host.gpu_lock:
            report = self.state.updater.apply(
                self.state.host.model, self.state.overlay, job["token_ids"], job["gen_start"],
                job["credit_spans"], job["reward"], job["kind"], self.state.replay, self.state.journal,
            )
        self._absorb_accepted(report)

    # ##################################################################
    # process candidate
    # the per-turn individuation gate, off the response path: score the user's
    # message surprise, pass it through the adaptive gate, and only on a surprising
    # turn log the experience and absorb the user's tokens into the volatile overlay
    def _process_candidate(self, job: dict) -> None:
        span = job["credit_spans"][0]
        # span_logprobs takes the gpu lock itself — do NOT wrap it (the lock is
        # not reentrant, and a double-acquire deadlocks the worker holding it)
        logp = self.state.host.span_logprobs(job["token_ids"], Span("user", span[0], span[1]), True)
        surprise = float(-logp.mean())
        if not self.state.surprise_gate.consider(surprise):
            return
        experience = Experience.create(job["user_text"], job["context_digest"], surprise,
                                       self.state.model_path, self.accepted_updates)
        self.state.experience_log.record(experience)
        self.state.journal.record("experience", experience_id=experience.id, surprise=surprise)
        if not self.state.config.individuation.absorb_overlay or self.state.pause_flag.paused:
            return
        with self.state.host.gpu_lock:
            report = self.state.updater.apply(
                self.state.host.model, self.state.overlay, job["token_ids"], job["gen_start"],
                job["credit_spans"], 1.0, "absorb", None, self.state.journal,
            )
        self._absorb_accepted(report)

    # ##################################################################
    # absorb accepted
    # bookkeeping shared by the direct and candidate absorb paths: count the update
    # and re-check the adapter-norm ceiling that pauses runaway plasticity
    def _absorb_accepted(self, report) -> None:
        if not report.accepted:
            return
        self.accepted_updates += 1
        self.state.pause_flag.check(self.state.overlay.total_norm(),
                                    self.state.config.plasticity.adapter_norm_ceiling)

    # ##################################################################
    # credit spans
    # the answer and tool_call spans (and think when configured), newest first
    # and capped at three — the regions an outcome actually credits. When a turn
    # produced NO answer or tool call (a reasoning model that spent the whole
    # budget inside <think>), fall back to the reasoning span so the feedback is
    # never wasted — the reasoning IS the decision when nothing else was emitted
    def _credit_spans(self, trace: Trace) -> list:
        spans = self._spans_of_kinds(trace, self._primary_kinds())
        if not spans:
            spans = self._spans_of_kinds(trace, {"think"})
        return list(reversed(spans))[:3]

    def _primary_kinds(self) -> set:
        kinds = {"tool_call", "answer"}
        if self.state.config.plasticity.include_think_tokens:
            kinds.add("think")
        return kinds

    def _spans_of_kinds(self, trace: Trace, kinds: set) -> list:
        return [(span.start, span.end) for span in trace.spans if span.kind in kinds]

    # ##################################################################
    # after accept
    # replay the primary positive span, re-check the pause ceiling, checkpoint on
    # cadence, and run the canary when a negative or scheduled probe is due
    def _after_accept(self, trace: Trace, credit_spans: list, report, job: dict) -> None:
        self.accepted_updates += 1
        config = self.state.config
        if report.reward > 0:
            start, end = credit_spans[0]
            self.state.replay.add(trace.token_ids[start:end])
        self.state.pause_flag.check(self.state.overlay.total_norm(), config.plasticity.adapter_norm_ceiling)
        if self.accepted_updates % config.guards.checkpoint_every == 0:
            self.state.checkpoints.save(self.state.overlay, self.accepted_updates, self.last_clean)
        self._maybe_canary(job)

    # ##################################################################
    # maybe canary
    # probe after every accepted negative update and every canary_every-th update
    # when canary is enabled at all (disabled entirely under the test threshold)
    def _maybe_canary(self, job: dict) -> None:
        config = self.state.config
        if not canary_enabled(config):
            return
        negative = job["kind"] == "reward" and job["reward"] < 0
        if negative or self.accepted_updates % config.guards.canary_every == 0:
            self._run_canary()

    # ##################################################################
    # run canary
    # lazily import the evaluation package, probe, journal the result, and roll
    # back to the last good checkpoint on two consecutive breaches
    def _run_canary(self) -> None:
        from evaluation import canary

        report = canary.probe(self.state.host)
        self.state.journal.record("canary", mean_kl=report.mean_kl, match_failures=report.match_failures)
        self.last_clean = canary.is_clean(report, self.state.config.guards)
        if self.last_clean:
            self.consecutive_breaches = 0
            return
        self.consecutive_breaches += 1
        if self.consecutive_breaches >= self.state.config.guards.canary_breaches_to_rollback:
            self._rollback()

    # ##################################################################
    # rollback
    # restore the last canary-clean overlay, journal it, and clear the streak
    def _rollback(self) -> None:
        with self.state.host.gpu_lock:
            checkpoint_id = self.state.checkpoints.restore(self.state.overlay)
            materialize(self.state.host)
        self.state.journal.record("rollback", checkpoint_id=checkpoint_id)
        self.consecutive_breaches = 0
