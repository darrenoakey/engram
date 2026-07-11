# =============================================================================
#  name_recall — the dedicated end-to-end "did it actually learn me" test
#  why: this is the single most important behaviour to keep honest — tell engram
#  your name once, and a FRESH conversation recalls it. It is NOT part of the
#  pytest suite (it needs the slow background dream loop and a whole dedicated
#  server); run it on demand with:  ./run name-recall
#
#  It builds its OWN isolated server on the 0.8B test model with a clean data
#  dir, a cold overlay, auto_dream ON, a short idle sleep, and a low surprise
#  warmup so the gate fires quickly. Then over plain HTTP:
#    1. ask "what's my name?" cold             -> assert it does NOT know
#    2. say "my name is darren"                -> wait for the wake gate to NOTICE
#    3. wait for the dream loop to CONSOLIDATE -> learned count rises, dream journaled
#    4. start a FRESH conversation, ask again  -> assert it answers "darren"
#  Everything real: a real model, real updates, the real background DreamLoop.
# =============================================================================
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

from common import config, store
from common.config import load_config
from plasticity.checkpoints import Checkpoints
from plasticity.journal import Journal
from plasticity.replay import ReplayBuffer
from server.app import create_app, serve_in_thread, stop_state

NAME = "darren"
TELL = f"My name is {NAME}."
ASK = "What is my name?"
CHAT_TIMEOUT = 300.0
TIMEOUT = 60.0
PROBE_MAX_TOKENS = 64
MAX_ATTEMPTS = 6   # a real user repeats a fact; give the loop that many nights to commit


# ##################################################################
# name recall result
# the four step outcomes plus the single pass/fail the caller exits on
@dataclass
class NameRecallResult:
    cold_answer: str         # what it said before being told
    noticed: bool            # did the wake gate flag the name turn?
    consolidated: bool       # did a dream commit and learn the fact?
    fresh_answer: str        # what it said in a fresh conversation after learning
    passed: bool


# ##################################################################
# build server
# an isolated engram on the 0.8B with a fresh data root, cold overlay, auto_dream
# ON, a short idle sleep, and a low surprise warmup so the gate fires fast even
# from a cold start (the default warmup of 8 would need 8 priming turns)
def build_server() -> tuple:
    root = Path("output/testing") / f"name-recall-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    store.set_data_root(root)
    base = load_config()
    tuned = replace(
        base,
        model=replace(base.model),  # keep the 0.8B test_path as the serving model
        guards=replace(base.guards, canary_every=10 ** 9),  # keep eval pkg untouched
        plasticity=replace(base.plasticity, lr_consolidate=4e-5),  # stronger, to overcome the 0.8B identity prior
        sampling=replace(base.sampling, temperature=0.0, top_p=0.0, top_k=0, max_tokens=PROBE_MAX_TOKENS),
        individuation=replace(
            base.individuation,
            enabled=True,
            auto_dream=True,
            absorb_overlay=True,
            surprise_warmup=1,        # fire after a single observed turn
            surprise_percentile=0.1,  # very permissive gate
            min_user_tokens=3,
            selfedit_paraphrases=8,   # the 0.8B mis-edits ~5/6; more tries = more on-topic pairs
            dream_epochs=4,           # extra passes so the one good pair dominates the garbage ones
            dream_idle_sleep_s=1.0,   # loop re-checks every second
            probe_recall_target=0.34,  # the 0.8B's recall is genuinely weaker than the 9B
        ),
    )
    config.set_forced_config_path(root / "no-config.toml")
    app = create_app(
        tuned, model_path=tuned.model.test_path,
        journal=Journal(root / "journal.jsonl"),
        checkpoints=Checkpoints(root / "checkpoints", ring=20),
        replay=ReplayBuffer(root / "replay.json"),
    )
    handle, thread, url = serve_in_thread(app)
    return app, handle, thread, url, root


# ##################################################################
# chat
# one turn of real inference (thinking off for snappy direct answers); returns
# the assistant's reply text
def chat(url: str, messages: list, max_tokens: int = PROBE_MAX_TOKENS) -> str:
    body = {"messages": messages, "enable_thinking": False, "max_tokens": max_tokens}
    response = httpx.post(f"{url}/v1/chat/completions", json=body, timeout=CHAT_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    return message.get("content") or message.get("reasoning_content") or ""


# ##################################################################
# brain / journal
def _brain(url: str) -> dict:
    response = httpx.get(f"{url}/v1/brain", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def _journal_counts(url: str) -> dict:
    response = httpx.get(f"{url}/v1/brain/journal?limit=1000", timeout=TIMEOUT)
    response.raise_for_status()
    counts: dict = {}
    for event in response.json()["events"]:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    return counts


# ##################################################################
# wait noticed
# block until the wake gate has flagged a name turn — an experience event is
# journaled beyond the baseline count
def wait_noticed(url: str, since_exp: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _journal_counts(url).get("experience", 0) > since_exp:
            return True
        threading.Event().wait(0.5)
    return False


# ##################################################################
# wait dream settles
# block until the dream loop finishes ONE night past the baseline — returns the
# committed/reverted verdict and the new learned count. Key: a dream
# OPTIMISTICALLY adds probes then truncates them on revert, so peeking at
# `learned` mid-dream is racy; the journal's dream/dream_reverted event is the
# honest "the night is over" signal
def wait_dream_settles(url: str, since_dreams: int, timeout: float = 300.0) -> tuple:
    deadline = time.time() + timeout
    while time.time() < deadline:
        counts = _journal_counts(url)
        dreams = counts.get("dream", 0) + counts.get("dream_reverted", 0)
        if dreams > since_dreams:
            committed = counts.get("dream", 0) > 0
            return committed, _brain(url)["individuation"]["learned"]
        threading.Event().wait(1.0)
    return False, _brain(url)["individuation"]["learned"]


# ##################################################################
# run name recall
# the whole test: cold check, then tell→notice→consolidate repeated until a dream
# commits (a real user repeats a fact), then fresh recall. Each attempt tells the
# name once, waits for the wake gate to notice, then waits for the dream to settle
def run_name_recall() -> NameRecallResult:
    app, handle, thread, url, root = build_server()
    state = app.state.engram
    try:
        # 1. COLD — it should not know the name yet
        cold = chat(url, [{"role": "user", "content": ASK}]).lower()
        cold_knows = NAME in cold
        print(f"[1] cold answer:  {cold[:120]!r}")
        if cold_knows:
            print(f"    ! '{NAME}' already in the cold answer — model seeded; test inconclusive")

        # 2-4. TELL → NOTICE → CONSOLIDATE, repeating until a dream commits
        noticed = False
        consolidated = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            since_exp = _journal_counts(url).get("experience", 0)
            chat(url, [{"role": "user", "content": TELL}])
            noticed = noticed or wait_noticed(url, since_exp)
            print(f"[2] told (try {attempt}/{MAX_ATTEMPTS}): noticed={noticed}")
            since_dreams = _journal_counts(url).get("dream", 0) + _journal_counts(url).get("dream_reverted", 0)
            committed, learned = wait_dream_settles(url, since_dreams)
            print(f"[4]   dream settled: committed={committed} learned={learned}")
            if committed:
                consolidated = True
                break
            print("    reverted (recall below target) — repeating, as a real user would")

        # 5. FRESH RECALL — brand new conversation. Ask the learned probe's
        # question if one exists (what the dream actually trained on), else ASK
        memory = httpx.get(f"{url}/v1/brain/memory", timeout=TIMEOUT).json()
        question = memory["learned"][0]["question"] if memory["learned"] else ASK
        fresh = chat(url, [{"role": "user", "content": question}]).lower()
        recalls = NAME in fresh
        print(f"[5] fresh answer: {fresh[:120]!r}  (asked: {question!r})")

        passed = (not cold_knows) and noticed and consolidated and recalls
        return NameRecallResult(cold, noticed, consolidated, fresh, passed)
    finally:
        stop_state(state)
        handle.should_exit = True
        thread.join(timeout=10)
        store.set_data_root(None)
        config.set_forced_config_path(None)


# ##################################################################
# scoreboard
def scoreboard(result: NameRecallResult) -> str:
    cold_ok = NAME not in result.cold_answer.lower()
    recalls = NAME in result.fresh_answer.lower()
    return "\n".join([
        "engram name-recall (did it actually learn me?)",
        f"  [{'PASS' if cold_ok else 'FAIL'}] cold check     did NOT already know '{NAME}'",
        f"  [{'PASS' if result.noticed else 'FAIL'}] noticed        wake gate flagged the name turn",
        f"  [{'PASS' if result.consolidated else 'FAIL'}] consolidated   dream committed the learned fact",
        f"  [{'PASS' if recalls else 'FAIL'}] fresh recall   a new conversation says '{NAME}'",
        f"RESULT: {'PASS' if result.passed else 'FAIL'}",
    ])
