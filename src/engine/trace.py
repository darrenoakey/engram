# =============================================================================
#  trace — the record of one generation (tokens, per-token logprobs, spans)
#  why: the journal and every weight update are keyed off traces; they must
#  survive a restart, so mx arrays become plain lists and persist via store
# =============================================================================
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from common import store

VALID_KINDS = ("think", "answer", "tool_call")


# =============================================================================
#  span — a contiguous token region of one kind within the generated tokens
#  why: credit assignment operates on spans; it serializes as [kind, start, end]
#  so the on-disk form stays compact and obvious
@dataclass
class Span:
    kind: str
    start: int
    end: int

    def as_list(self) -> list:
        return [self.kind, self.start, self.end]

    @staticmethod
    def from_list(value: list) -> "Span":
        return Span(str(value[0]), int(value[1]), int(value[2]))


def _plain_ints(values) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(v) for v in values]


def _plain_floats(values) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(v) for v in values]


# =============================================================================
#  trace — the full immutable-ish record produced by ModelHost.generate
#  why: one row of the system of record; feedback is appended over its life,
#  everything else is fixed at generation time
@dataclass
class Trace:
    trace_id: str
    created_at: str
    token_ids: list[int]
    gen_start: int
    logprobs: list[float]
    spans: list[Span]
    tool_call_ids: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)
    feedback: list = field(default_factory=list)

    @staticmethod
    def create(token_ids, gen_start: int, logprobs, spans: list[Span], sampling: dict) -> "Trace":
        return Trace(
            trace_id=uuid.uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(),
            token_ids=_plain_ints(token_ids),
            gen_start=int(gen_start),
            logprobs=_plain_floats(logprobs),
            spans=list(spans),
            sampling=dict(sampling),
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "token_ids": _plain_ints(self.token_ids),
            "gen_start": int(self.gen_start),
            "logprobs": _plain_floats(self.logprobs),
            "spans": [s.as_list() for s in self.spans],
            "tool_call_ids": dict(self.tool_call_ids),
            "sampling": dict(self.sampling),
            "feedback": list(self.feedback),
        }

    @staticmethod
    def from_dict(data: dict) -> "Trace":
        return Trace(
            trace_id=data["trace_id"],
            created_at=data["created_at"],
            token_ids=[int(t) for t in data["token_ids"]],
            gen_start=int(data["gen_start"]),
            logprobs=[float(x) for x in data["logprobs"]],
            spans=[Span.from_list(s) for s in data["spans"]],
            tool_call_ids=dict(data.get("tool_call_ids", {})),
            sampling=dict(data.get("sampling", {})),
            feedback=list(data.get("feedback", [])),
        )

    def path(self) -> Path:
        return store.traces_dir() / f"{self.trace_id}.json.gz"

    def save(self) -> Path:
        target = self.path()
        store.write_json_gz(target, self.to_dict())
        return target

    @staticmethod
    def load(trace_id: str) -> "Trace":
        return Trace.from_dict(store.read_json_gz(store.traces_dir() / f"{trace_id}.json.gz"))

    @staticmethod
    def list_recent(limit: int) -> list["Trace"]:
        files = sorted(store.traces_dir().glob("*.json.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [Trace.from_dict(store.read_json_gz(p)) for p in files[:limit]]
