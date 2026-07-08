# =============================================================================
#  corroborate — the truth/durability gate (INDIVIDUATION.md §5.2)
#  why: with no labels a fact, a joke and a role-play instruction are the same
#  token stream. Before anything is consolidated, the model itself judges whether
#  a user turn states a DURABLE fact or preference ABOUT THE USER worth keeping,
#  versus transient/role-play/hypothetical. v1 is a content durability classifier
#  (recurrence across sessions is v1.1); a durable verdict carries a canonical
#  third-person statement that becomes the self-edit's target.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, replace

ALLOWED_KINDS = ("fact", "preference", "role_play", "hypothetical", "transient", "other")

CLASSIFY_MAX_TOKENS = 110

_SYSTEM = (
    "You judge whether a user's message states a DURABLE fact or preference ABOUT THE USER worth "
    "remembering long-term (their identity, situation, tastes, or constraints). Role-play, "
    "hypotheticals, jokes, and one-off commands are NOT durable. Reply in EXACTLY this format and "
    "nothing else:\nDURABLE: yes or no\nKIND: fact, preference, role_play, hypothetical, transient, "
    "or other\nSTATEMENT: a short third-person sentence about the user, or none"
)


# ##################################################################
# verdict
# the durability decision for one user turn: whether to keep it, its category,
# and (only when durable) the canonical statement the self-edit will teach
@dataclass
class Verdict:
    durable: bool
    kind: str
    statement: str


# ##################################################################
# classify
# ask the model to judge one user message and parse its structured reply into a
# Verdict; greedy decoding keeps the judgement deterministic and reproducible
def classify(host, user_text: str, config) -> Verdict:
    text = _judge(host, user_text)
    return _parse_verdict(text)


# ##################################################################
# judge
# greedy, thinking-off generation of the structured judgement for one message;
# the answer tokens are decoded back to text for defensive parsing
def _judge(host, user_text: str) -> str:
    sampling = replace(host.config.sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=CLASSIFY_MAX_TOKENS)
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": f'Message: "{user_text}"'}]
    trace = host.generate(messages, sampling=sampling, enable_thinking=False)
    return host.tokenizer.decode(trace.token_ids[trace.gen_start:])


# ##################################################################
# parse verdict
# a durable verdict requires BOTH an affirmative DURABLE line and a non-empty
# statement, so the invariant durable <=> statement holds for the caller
def _parse_verdict(text: str) -> Verdict:
    affirmative = _affirmative(_field(text, "durable"))
    statement = _statement(_field(text, "statement"))
    kind = _kind(_field(text, "kind"))
    durable = affirmative and statement != ""
    return Verdict(durable=durable, kind=kind, statement=statement if durable else "")


# ##################################################################
# field
# the text after "<key>:" on the first line that starts with that key
# (case-insensitive); empty string when the model omitted the line
def _field(text: str, key: str) -> str:
    for line in text.splitlines():
        if line.strip().lower().startswith(key) and ":" in line:
            return line.split(":", 1)[1].strip()
    return ""


# ##################################################################
# affirmative
# whether a DURABLE line reads as yes/true rather than no
def _affirmative(raw: str) -> bool:
    low = raw.strip().lower()
    return low.startswith("y") or low.startswith("true")


# ##################################################################
# statement
# a cleaned canonical statement, or empty when the model wrote none/nothing
def _statement(raw: str) -> str:
    cleaned = raw.strip().strip('"').strip()
    return "" if cleaned.lower() in ("", "none", "n/a") else cleaned


# ##################################################################
# kind
# map the free-form KIND text onto the allowed vocabulary, defaulting to other
def _kind(raw: str) -> str:
    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")
    for candidate in ALLOWED_KINDS:
        if candidate in normalized:
            return candidate
    return "other"
