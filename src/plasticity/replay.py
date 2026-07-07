# =============================================================================
#  replay — ring buffer of positive credit spans (DESIGN.md §4)
#  why: mixing >=1 positive replay span into every update measurably cuts
#  catastrophic forgetting and counters the negative-update squeezing pathology
#  (a mixed-sign batch). Spans are token-id lists, persisted as JSON so the
#  buffer survives restarts. Capped so it never grows without bound.
# =============================================================================
from __future__ import annotations

import random
from pathlib import Path

from common import store

DEFAULT_CAP = 200


# ##################################################################
# replay buffer
# a bounded FIFO of positive spans (token-id lists); every trace with reward>0
# contributes its credit span, and updates sample k of them without replacement
class ReplayBuffer:
    def __init__(self, path: Path | None = None, cap: int = DEFAULT_CAP) -> None:
        self.path = path if path is not None else store.replay_path()
        self.cap = cap
        self.spans: list[list[int]] = self._read()

    # ##################################################################
    # read
    # load the persisted spans, tolerating a missing file (cold start)
    def _read(self) -> list[list[int]]:
        if not self.path.exists():
            return []
        return store.read_json(self.path)

    # ##################################################################
    # add
    # append a positive span, drop the oldest beyond the cap, persist atomically
    def add(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        self.spans.append(list(token_ids))
        if len(self.spans) > self.cap:
            self.spans = self.spans[-self.cap:]
        store.atomic_write_json(self.path, self.spans)

    # ##################################################################
    # seed
    # bulk-load reference spans (canary continuations) so the very first
    # updates still get a positive replay span before any traffic arrives
    def seed(self, spans: list[list[int]]) -> None:
        for span in spans:
            self.add(span)

    # ##################################################################
    # sample
    # up to k spans chosen uniformly without replacement (fewer if the buffer
    # is smaller); empty buffer yields an empty list
    def sample(self, k: int) -> list[list[int]]:
        if not self.spans:
            return []
        return random.sample(self.spans, min(k, len(self.spans)))

    def __len__(self) -> int:
        return len(self.spans)
