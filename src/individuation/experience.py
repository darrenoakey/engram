# =============================================================================
#  experience — the immutable-ish log of surprising turns (INDIVIDUATION.md §4.2)
#  why: the experience log is the source of record and the provenance spine — the
#  night's dream selects from it, and every durable weight change traces back to
#  the logged interaction(s) that caused it. Records are appended as JSONL through
#  the common store; consolidation flips a flag (a whole-file atomic rewrite) so a
#  processed night is never re-dreamed, but the user text and provenance stay.
# =============================================================================
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from common import store


# ##################################################################
# context digest
# a short stable hash of the prior conversation, so an experience carries which
# context provoked the surprise without storing the whole history verbatim
def context_digest(messages: list) -> str:
    joined = "\n".join(f"{m.get('role', '')}:{m.get('content', '')}" for m in messages)
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


# ##################################################################
# experience
# one logged high-surprise user turn plus the provenance needed to reproject it:
# the user's text, a digest of its context, the surprise value, and the model
# generation / learner version it was captured under
@dataclass
class Experience:
    id: str
    created_at: str
    user_text: str
    context_digest: str
    surprise: float
    serving_generation: str
    learner_version: int
    consolidated: bool = False

    # ##################################################################
    # create
    # mint a fresh experience with a uuid id and a utc timestamp
    @staticmethod
    def create(user_text: str, context_digest: str, surprise: float, serving_generation: str,
               learner_version: int) -> "Experience":
        return Experience(uuid.uuid4().hex, datetime.now(timezone.utc).isoformat(), user_text,
                          context_digest, float(surprise), serving_generation, int(learner_version))

    def to_dict(self) -> dict:
        return {"id": self.id, "created_at": self.created_at, "user_text": self.user_text,
                "context_digest": self.context_digest, "surprise": float(self.surprise),
                "serving_generation": self.serving_generation, "learner_version": int(self.learner_version),
                "consolidated": bool(self.consolidated)}

    @staticmethod
    def from_dict(data: dict) -> "Experience":
        return Experience(data["id"], data["created_at"], data["user_text"], data["context_digest"],
                          float(data["surprise"]), data["serving_generation"], int(data["learner_version"]),
                          bool(data.get("consolidated", False)))


# ##################################################################
# experience log
# append-only JSONL over the common store; the path is injectable so tests write
# to output/testing. mark_consolidated is the one mutation — a whole-file atomic
# rewrite that flips flags without touching the recorded provenance.
class ExperienceLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else store.data_root() / "experience.jsonl"

    # ##################################################################
    # record
    # append one experience as a JSONL row through the store's fsynced append
    def record(self, exp: Experience) -> None:
        store.append_jsonl(self.path, exp.to_dict())

    # ##################################################################
    # all
    # every logged experience in write order
    def all(self) -> list[Experience]:
        return [Experience.from_dict(row) for row in store.read_jsonl(self.path)]

    # ##################################################################
    # recent
    # the most recent n experiences, oldest-first
    def recent(self, n: int) -> list[Experience]:
        return self.all()[-n:]

    # ##################################################################
    # unconsolidated
    # experiences not yet folded into the base — the candidates a dream selects
    def unconsolidated(self) -> list[Experience]:
        return [exp for exp in self.all() if not exp.consolidated]

    # ##################################################################
    # mark consolidated
    # flip the consolidated flag for the given ids and atomically rewrite the log
    def mark_consolidated(self, ids: list[str]) -> None:
        wanted = set(ids)
        rows = store.read_jsonl(self.path)
        for row in rows:
            if row.get("id") in wanted:
                row["consolidated"] = True
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        store.atomic_write_bytes(self.path, payload.encode())
