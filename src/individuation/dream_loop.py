# =============================================================================
#  dream_loop — the continuous background learner (LEARNING.md §11.5)
#  why: v1's dream is manual — a human must click Consolidate or POST
#  /v1/brain/dream, so facts the surprise gate NOTICED were never durably learned
#  unless someone remembered to ask. This module is the fixed loop that makes
#  learning continuous: dream → finish (commit or revert) → check for new work →
#  if work exists, loop again immediately; if not, sleep and re-check. During
#  sustained use it learns continuously; it sleeps only when idle.
#
#  It reuses the existing atomic dream() for NEW experiences unchanged, and adds
#  a low-speed repolish pass for already-learned (probe) facts that have gone
#  stale (>repolish_after_h since last trained). One queue, one loop, one GPU:
#  each cycle holds the update worker (so chat always wins the GPU) exactly as
#  brain_api._consolidate does, and every weight change still flows through the
#  same snapshot → probe/sentinels → commit/revert gate the manual dream uses.
# =============================================================================
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from individuation import dream as D


# ##################################################################
# dream loop
# one daemon thread per process. Reads live state each cycle so newly-noted
# experiences and in-flight wake-absorbs are picked up without restart. A failing
# cycle is journaled loudly and the loop continues — a dead learning loop is the
# silent-failure case that leaves noticed facts forever unconsolidated
class DreamLoop:
    def __init__(self, state) -> None:
        self.state = state
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="engram-dream-loop", daemon=True)
        # observable status, copied under _lock by the brain snapshot
        self.state_label = "idle"
        self.last_outcome = "none"
        self.last_dream_at = ""
        self.last_dream_committed_at = ""
        self.cycles = 0
        self._next_check_at = 0.0

    # ##################################################################
    # lifecycle
    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        # join only if called from a thread other than the loop itself
        if threading.current_thread() is not self.thread and self.thread.is_alive():
            self.thread.join(timeout=30.0)

    # ##################################################################
    # status
    # a read-only snapshot of what the loop is doing, for /v1/brain and the UI.
    # next_check_in_s is clamped at 0 so a sleeping loop reads as "imminent"
    def status(self) -> dict:
        with self._lock:
            remaining = max(0.0, self._next_check_at - time.time())
        return {
            "running": not self._stop.is_set() and self.thread.is_alive(),
            "state": self.state_label,
            "last_outcome": self.last_outcome,
            "last_dream_at": self.last_dream_at,
            "last_committed_at": self.last_dream_committed_at,
            "cycles": self.cycles,
            "next_check_in_s": int(remaining),
        }

    # ##################################################################
    # run
    # the fixed loop: one cycle, then sleep when there is nothing to do. The
    # sleep is interruptible (stop() wakes it immediately). A cycle that throws
    # is journaled dream_loop_error and swallowed — the loop must never die
    def _run(self) -> None:
        sleep_s = max(1.0, self.state.config.individuation.dream_idle_sleep_s)
        while not self._stop.is_set():
            try:
                did = self._cycle()
            except Exception as error:
                self.state.journal.record("dream_loop_error", error=repr(error))
                with self._lock:
                    self.last_outcome = "error"
                    self.last_dream_at = _now_iso()
                did = False
            # new work loops immediately; idle sleeps before re-checking
            wait = 0.0 if did else sleep_s
            with self._lock:
                self._next_check_at = time.time() + wait
                if wait == 0.0:
                    self.state_label = "dreaming"
            if self._stop.wait(wait):
                return

    # ##################################################################
    # cycle
    # one decision: dream new experiences (priority), else repolish stale facts,
    # else nothing. Holds the update worker for exclusivity exactly as the manual
    # /v1/brain/dream endpoint does. Returns True if it did work (loop again
    # immediately), False if idle (sleep before re-checking)
    def _cycle(self) -> bool:
        config = self.state.config
        log = self.state.experience_log
        probe = self.state.individuation_probe
        if log.unconsolidated():
            self._run_dream()
            return True
        stale = _stale_probes(probe, config.individuation.repolish_after_h,
                              config.individuation.repolish_min_batch)
        if stale:
            batch = stale[: config.individuation.repolish_max_batch]
            self._run_repolish(batch)
            return True
        with self._lock:
            self.state_label = "idle"
        return False

    # ##################################################################
    # run dream
    # quiesce the worker, run the existing atomic dream on the NEW experiences,
    # record the outcome. A committed dream drains the unconsolidated set; a
    # reverted one leaves it (the snapshot was restored) — either way the loop
    # re-checks next cycle
    def _run_dream(self) -> None:
        with self._lock:
            self.state_label = "dreaming"
        self.state.queue.hold()
        try:
            report = D.dream(self.state.host, self.state.overlay, self.state.updater, self.state.journal,
                             self.state.experience_log, self.state.individuation_probe, self.state.config)
        finally:
            self.state.queue.release()
        self._record_outcome("dream", report)

    # ##################################################################
    # run repolish
    # the refresher for stale learned facts: same hold/release, but the gentler
    # repolish path. Re-polish is lower priority than new experiences by
    # construction (_cycle tries new first)
    def _run_repolish(self, probes) -> None:
        with self._lock:
            self.state_label = "repolishing"
        self.state.queue.hold()
        try:
            report = D.repolish(self.state.host, self.state.overlay, self.state.updater, self.state.journal,
                                self.state.individuation_probe, self.state.config, probes)
        finally:
            self.state.queue.release()
        self._record_outcome("repolish", report)

    # ##################################################################
    # record outcome
    # stamp the observable status from a dream/repolish report under the lock
    def _record_outcome(self, kind: str, report) -> None:
        when = _now_iso()
        outcome = kind + ("_committed" if report.committed else "_reverted")
        with self._lock:
            self.cycles += 1
            self.last_outcome = outcome
            self.last_dream_at = when
            self.state_label = "idle"
            if report.committed:
                self.last_dream_committed_at = when


# ##################################################################
# stale probes
# the re-polish selection: learned facts older than repolish_after_h, but only if
# at least repolish_min_batch qualify (avoid a cycle for a single stale fact when
# the operator set a higher floor). Cutoff computed in UTC
def _stale_probes(probe, after_h: float, min_batch: int) -> list:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=after_h)).isoformat()
    stale = probe.stale(cutoff)
    return stale if len(stale) >= max(1, min_batch) else []


# ##################################################################
# now iso
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
