# =============================================================================
#  selfedit — turn a corroborated statement into assistant-knowledge (§5.3)
#  why: teacher-forcing the user's raw tokens makes the model MIMIC the user's
#  voice more than KNOW the fact (the predict-vs-know gap, §11). The self-edit
#  reformulates a durable statement into assistant-knowledge QA pairs with
#  paraphrase augmentation, and training targets the ANSWER tokens — that is what
#  yields cold recall instead of memorising one phrasing. This module produces
#  the pairs and the exact teacher-forcing material the absorb update consumes.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, replace

_SYSTEM = (
    "You convert a fact about the user into assistant knowledge. Produce {count} distinct "
    "question-and-answer pairs. Each QUESTION is one a user might ask that should make you recall the "
    "fact; each ANSWER is a short reply in your own voice that states the fact (address the user as "
    "'you'). Use EXACTLY this format, one pair per two lines:\nQ: <question>\nA: <answer>"
)


# ##################################################################
# qa pair
# one assistant-knowledge reformulation: a question that should elicit the fact
# and a short answer that states it in the assistant's own voice
@dataclass
class QAPair:
    question: str
    answer: str


# ##################################################################
# synthesize
# generate up to selfedit_paraphrases assistant-knowledge QA pairs for a durable
# statement; greedy decoding, and [] when the model yields nothing parseable
def synthesize(host, statement: str, config) -> list[QAPair]:
    count = config.individuation.selfedit_paraphrases
    text = _generate(host, statement, count, config.individuation.selfedit_max_tokens)
    return _parse_pairs(text, count)


# ##################################################################
# generate
# greedy, thinking-off generation of the QA block for one statement; the answer
# tokens are decoded back to text for defensive line-based parsing
def _generate(host, statement: str, count: int, max_tokens: int) -> str:
    sampling = replace(host.config.sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=max_tokens)
    messages = [{"role": "system", "content": _SYSTEM.format(count=count)},
                {"role": "user", "content": f"Fact: {statement}"}]
    trace = host.generate(messages, sampling=sampling, enable_thinking=False)
    return host.tokenizer.decode(trace.token_ids[trace.gen_start:])


# ##################################################################
# parse pairs
# walk the lines pairing each "Q:" with the next "A:"; only complete, non-empty
# pairs are kept, up to the requested count
def _parse_pairs(text: str, limit: int) -> list[QAPair]:
    pairs: list[QAPair] = []
    pending: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("q:"):
            pending = stripped[2:].strip()
        elif low.startswith("a:") and pending:
            answer = stripped[2:].strip()
            if answer:
                pairs.append(QAPair(pending, answer))
                pending = None
            if len(pairs) >= limit:
                break
    return pairs


# ##################################################################
# training example
# teacher-forcing material for one QA pair in assistant-knowledge form: the
# prompt is the question with the assistant turn opened (thinking off, so no
# think block), the answer tokens are appended, and the span covers ONLY the
# answer — the tokens the absorb update trains on to install knowledge
def training_example(host, qa: QAPair) -> tuple[list[int], int, tuple[int, int]]:
    prompt = host.tokenizer.apply_chat_template(
        [{"role": "user", "content": qa.question}], add_generation_prompt=True, enable_thinking=False
    )
    prompt = [int(t) for t in prompt]
    answer = [int(t) for t in host.tokenizer.encode(qa.answer, add_special_tokens=False)]
    token_ids = prompt + answer
    gen_start = len(prompt)
    return token_ids, gen_start, (gen_start, len(token_ids))
