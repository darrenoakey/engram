# =============================================================================
#  journal — append-only JSONL system of record (DESIGN.md §4)
#  why: weights are derived artifacts; the journal is the truth. Every update,
#  rejection, rollback, checkpoint, consolidation and canary is recorded so the
#  learned history survives restarts and can be audited. Writes go through the
#  common store's atomic append (single process, open a+, fsync).
# =============================================================================
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from common import store

EVENT_TYPES = (
    "update", "rejected_update", "rollback", "checkpoint", "consolidate", "canary", "worker_error",
    "consolidate_reverted",
)


# ##################################################################
# journal
# thin recorder over the append-only JSONL; the path is injectable so tests
# write to output/testing instead of the live data dir
class Journal:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else store.journal_path()

    # ##################################################################
    # record
    # append one typed event with a wall-clock timestamp; unknown types are a
    # programming error and fail loudly rather than silently polluting stats
    def record(self, event_type: str, **fields) -> dict:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown journal event type: {event_type}")
        event = {"type": event_type, "at": time.time(), **fields}
        store.append_jsonl(self.path, event)
        return event

    # ##################################################################
    # tail
    # the most recent n events, oldest-first
    def tail(self, n: int) -> list:
        return store.read_jsonl(self.path, limit=n)

    # ##################################################################
    # stats
    # counts by type, the last canary event, and cumulative reward across all
    # accepted updates — the numbers /v1/brain reports
    def stats(self) -> dict:
        rows = store.read_jsonl(self.path)
        counts = Counter(row.get("type") for row in rows)
        canaries = [row for row in rows if row.get("type") == "canary"]
        reward = sum(row.get("reward", 0.0) for row in rows if row.get("type") == "update")
        return {
            "counts": dict(counts),
            "last_canary": canaries[-1] if canaries else None,
            "cumulative_reward": reward,
            "total_events": len(rows),
        }
