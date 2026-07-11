# =============================================================================
#  atomize — split a multi-fact user turn into atomic statements (INDIVIDUATION §5)
#  why: corroboration and self-edit operate per-STATEMENT, but a single user turn
#  can carry several facts ("my wife is Arlene, my son is Leo"). Fed the whole
#  turn, the classifier emits one compound statement and the self-edit loses every
#  proper noun — so nothing trainable reaches the overlay and the model later
#  confabulates. Atomize decomposes such a turn into one short third-person
#  sentence per fact BEFORE corroboration, so each atomic fact gets its own
#  self-edit target and its own cold-recall probe. A single-fact turn yields one
#  atom and behaves exactly as before; a failed split falls back to [user_text].
# =============================================================================
from __future__ import annotations

from dataclasses import replace

_SYSTEM = (
    "Break the user's message into a list of ATOMIC facts about them. Each fact is one short "
    "third-person sentence (start with 'The user') stating exactly one relationship, trait, or "
    "preference. Do not combine facts into one sentence. If the message states only one fact, "
    "output exactly one line. Do not add commentary. Output one fact per line and nothing else.\n"
    "Example input:  \"my wife is Arlene and my son is Leo\"\n"
    "Example output:\nThe user's wife is Arlene.\nThe user's son is Leo."
)


# ##################################################################
# atomize
# decompose one user turn into atomic third-person statements. Greedy decoding
# keeps the split deterministic. Returns [user_text] (the original, as one atom)
# when the model yields nothing parseable — so the caller never gets an empty list
# and a single-fact turn degrades cleanly to today's single-statement path
def atomize(host, user_text: str, config) -> list[str]:
    text = _generate(host, user_text, config.individuation.atomize_max_tokens)
    atoms = _parse_atoms(text)
    return atoms or [user_text]


# ##################################################################
# generate
# greedy, thinking-off generation of the atomic-fact list for one user turn;
# the lines are decoded back to text for defensive line-based parsing
def _generate(host, user_text: str, max_tokens: int) -> str:
    sampling = replace(host.config.sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=max_tokens)
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": f'Message: "{user_text}"'}]
    trace = host.generate(messages, sampling=sampling, enable_thinking=False)
    return host.tokenizer.decode(trace.token_ids[trace.gen_start:])


# ##################################################################
# parse atoms
# one statement per non-empty stripped line; drop explicit none/negations so a
# model that decides the turn has no facts returns [] (caller falls back). A line
# is kept only if it reads as a statement about the user (contains "user") OR is
# a reasonable sentence — this keeps the few-shot "Example output:" out and drops
# stray commentary, while still admitting "The user's dog is Peanut."
def _parse_atoms(text: str) -> list[str]:
    atoms: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().strip('"').strip()
        low = cleaned.lower()
        if not cleaned or low in ("none", "n/a", "no facts."):
            continue
        # skip the few-shot example header / echoed input
        if low.startswith("example") or low.startswith("message:"):
            continue
        atoms.append(cleaned)
    return atoms
