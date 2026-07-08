# =============================================================================
#  proof — the end-to-end demonstration instrument (DESIGN.md §4 / §6)
#  why: prove the whole learning loop against a RUNNING engram over plain HTTP,
#  so the same module measures the 0.8B in pytest and the live 9B in deployment.
#  Reinforcement pushes a self-produced continuation's logprob up, punishment
#  pushes a different one down, stability confirms the brain never paused, and
#  the caller restarts the server to show the learned overlay reloads intact.
# =============================================================================
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import httpx

TIMEOUT = 60.0
CHAT_TIMEOUT = 600.0
EPS = 1e-3
REINFORCE_PROMPT = "In one short sentence, what is two plus two?"
PUNISH_PROMPT = "In one short sentence, name a primary color."


# ##################################################################
# proof result
# per-phase numbers and verdicts plus the single pass/fail the caller exits on
@dataclass
class ProofResult:
    reinforcement: dict
    punishment: dict
    stability: dict
    passed: bool


# ##################################################################
# auth / user
# the bearer header the brain endpoints require and a one-message chat body
def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _user(prompt: str) -> list:
    return [{"role": "user", "content": prompt}]


# ##################################################################
# brain / processed
# fetch the full brain snapshot and count every job the worker has retired
# (accepted, rejected, or errored) — the monotone signal a drain waits on
def _brain(url: str) -> dict:
    response = httpx.get(f"{url}/v1/brain", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def _processed(brain: dict) -> int:
    counts = brain["updates"]["counts"]
    return counts.get("update", 0) + counts.get("rejected_update", 0) + counts.get("worker_error", 0)


# ##################################################################
# probe
# teacher-forced logprob of a continuation under the live overlay; the numeric
# handle every phase watches move
def probe(url: str, token: str, prompt: str, continuation: str) -> dict:
    body = {"prompt": prompt, "continuation": continuation}
    response = httpx.post(f"{url}/v1/brain/probe", json=body, headers=_auth(token), timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


# ##################################################################
# chat
# one turn of real inference; returns the model's own reply text (its answer,
# or its reasoning when a short generation never closed <think>) and the trace id
def chat(url: str, messages: list, max_tokens=None, temperature=None) -> tuple:
    body: dict = {"messages": messages}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    response = httpx.post(f"{url}/v1/chat/completions", json=body, timeout=CHAT_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    text = message.get("content") or message.get("reasoning_content") or ""
    return text, data["engram"]["trace_id"]


# ##################################################################
# feedback
# submit a reward for a trace; the server enqueues the guarded weight update
def feedback(url: str, token: str, trace_id: str, reward: float, source: str = "proof") -> dict:
    body = {"trace_id": trace_id, "reward": reward, "source": source}
    response = httpx.post(f"{url}/v1/feedback", json=body, headers=_auth(token), timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


# ##################################################################
# checkpoint / rollback
# capture the current overlay as a named baseline and later restore it exactly;
# the proof uses these to run each phase from an identical clean starting point
def checkpoint(url: str, token: str) -> str:
    response = httpx.post(f"{url}/v1/brain/checkpoint", headers=_auth(token), timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()["checkpoint_id"]


def rollback(url: str, token: str, checkpoint_id: str) -> None:
    body = {"checkpoint_id": checkpoint_id}
    response = httpx.post(f"{url}/v1/brain/rollback", json=body, headers=_auth(token), timeout=TIMEOUT)
    response.raise_for_status()


# ##################################################################
# wait drained
# block until the queue is empty and the worker has gone quiet. `since` is the
# processed count captured BEFORE the jobs were enqueued: requiring the count to
# rise past it means a long in-flight update (queue already empty, not yet
# counted) is always awaited to completion rather than mistaken for idle
def wait_drained(url: str, timeout: float = 600.0, since=None, poll: float = 0.25, quiet: float = 0.5) -> None:
    deadline = time.time() + timeout
    last = _processed(_brain(url))
    changed_at = time.time()
    while time.time() < deadline:
        brain = _brain(url)
        count = _processed(brain)
        if count != last:
            last, changed_at = count, time.time()
        risen = since is None or count > since
        if brain["queue_depth"] == 0 and risen and time.time() - changed_at >= quiet:
            return
        threading.Event().wait(poll)
    raise TimeoutError(f"queue did not drain within {timeout:.0f}s")


# ##################################################################
# continuation
# the model's own reply IS the probe target, kept verbatim (newlines intact so
# its tokens match the reinforced span) and bounded so a long 9B answer stays a
# reasonable probe. The full self-produced span — not a short prefix — is what
# carries a measurable, correctly-signed logprob shift through the plain-prompt
# probe: a short reasoning-boilerplate prefix is too context-bound to transfer
def _continuation(text: str) -> str:
    return text.strip()[:400]


# ##################################################################
# learn phase
# derive a fixed self-produced continuation, probe it, then repeatedly re-elicit
# and reward it, draining after each round; return the before/after logprobs
def _learn_phase(url: str, token: str, prompt: str, reward: float, rounds: int) -> dict:
    text, _ = chat(url, _user(prompt))
    continuation = _continuation(text)
    before = probe(url, token, prompt, continuation)["logprob_sum"]
    for _ in range(rounds):
        since = _processed(_brain(url))
        _, trace_id = chat(url, _user(prompt))
        feedback(url, token, trace_id, reward)
        wait_drained(url, since=since)
    after = probe(url, token, prompt, continuation)["logprob_sum"]
    return {"prompt": prompt, "continuation": continuation, "before": before, "after": after, "rounds": rounds}


# ##################################################################
# reinforcement phase
# reward the model's own continuation; its probed logprob must rise
def reinforcement_phase(url: str, token: str, rounds: int = 6) -> dict:
    result = _learn_phase(url, token, REINFORCE_PROMPT, 1.0, rounds)
    result["verdict"] = result["after"] > result["before"] + EPS
    return result


# ##################################################################
# punishment phase
# punish a different self-produced continuation; its probed logprob must fall
def punishment_phase(url: str, token: str, rounds: int = 6) -> dict:
    result = _learn_phase(url, token, PUNISH_PROMPT, -1.0, rounds)
    result["verdict"] = result["after"] < result["before"] - EPS
    return result


# ##################################################################
# stability phase
# read the brain: overlay magnitude, update/rejection counts and the last canary
# KL if one was journaled; the loop is healthy only if plasticity never paused
def stability_phase(url: str, token: str) -> dict:
    brain = _brain(url)
    counts = brain["updates"]["counts"]
    last_canary = brain["updates"]["last_canary"]
    paused = brain["paused"]["flag"]
    return {
        "total_norm": brain["overlay"]["total_norm"],
        "paused": paused,
        "updates": counts.get("update", 0),
        "rejections": counts.get("rejected_update", 0),
        "last_canary_mean_kl": last_canary["mean_kl"] if last_canary else None,
        "verdict": not paused,
    }


# ##################################################################
# self probe
# elicit the model's own continuation for a fixed prompt and probe it — used
# either side of a restart to show the learned overlay reloaded from checkpoint
def _self_probe(url: str, token: str, prompt: str) -> float:
    text, _ = chat(url, _user(prompt))
    continuation = _continuation(text)
    return probe(url, token, prompt, continuation)["logprob_sum"]


# ##################################################################
# persistence probe
# both phase measurements again; the caller compares these across a server
# restart (the restart itself is orchestrated by the caller, not this module)
def persistence_probe(url: str, token: str) -> dict:
    return {"reinforce": _self_probe(url, token, REINFORCE_PROMPT),
            "punish": _self_probe(url, token, PUNISH_PROMPT)}


# ##################################################################
# run proof
# the whole demonstration: punishment, reinforcement and stability, printing a
# compact scoreboard and returning a result that passes only if all verdicts
# hold. Each phase runs from an IDENTICAL clean baseline — the overlay is
# checkpointed up front and rolled back to it between phases — so the two
# directions are proven independently and can never cross-contaminate. That
# matters on small models whose reasoning is near-identical boilerplate across
# prompts (both continuations share most tokens, so without the reset punishing
# one would drag the other down); on the 9B the phases are naturally distinct.
# Punishment runs first so reinforcement's positive replay spans never leak into
# it. The reinforcement result is left in place so learned state is observable.
def run_proof(url: str, token: str, rounds: int = 6) -> ProofResult:
    baseline = checkpoint(url, token)
    punishment = punishment_phase(url, token, rounds)
    rollback(url, token, baseline)
    reinforcement = reinforcement_phase(url, token, rounds)
    stability = stability_phase(url, token)
    passed = reinforcement["verdict"] and punishment["verdict"] and stability["verdict"]
    result = ProofResult(reinforcement=reinforcement, punishment=punishment, stability=stability, passed=passed)
    print(_scoreboard(result))
    return result


# ##################################################################
# scoreboard / verdict line
# a compact human rendering of the phase numbers and the overall verdict
def _scoreboard(result: ProofResult) -> str:
    reinforce, punish, stability = result.reinforcement, result.punishment, result.stability
    return "\n".join([
        "engram proof of life",
        _verdict_line("reinforcement", reinforce["verdict"],
                      f"logprob {reinforce['before']:.3f} -> {reinforce['after']:.3f}"),
        _verdict_line("punishment", punish["verdict"],
                      f"logprob {punish['before']:.3f} -> {punish['after']:.3f}"),
        _verdict_line("stability", stability["verdict"],
                      f"norm {stability['total_norm']:.3f} paused={stability['paused']} "
                      f"updates={stability['updates']} rejections={stability['rejections']}"),
        f"RESULT: {'PASS' if result.passed else 'FAIL'}",
    ])


def _verdict_line(name: str, ok: bool, detail: str) -> str:
    return f"  [{'PASS' if ok else 'FAIL'}] {name:<14} {detail}"
