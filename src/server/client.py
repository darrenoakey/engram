# =============================================================================
#  client — the thin httpx client behind `engram status` and later tooling
#  why: operator commands and the proof harness talk to the live service over
#  real HTTP; this keeps the URL building, auth header and status formatting in
#  one reusable place so nothing hand-rolls requests.
# =============================================================================
from __future__ import annotations

import json

import httpx

from common.config import load_config

TIMEOUT = 15.0


# ##################################################################
# base url
# the served address from config; there is exactly one engram per config
def _base_url(config) -> str:
    return f"http://{config.server.host}:{config.server.port}"


# ##################################################################
# auth
# the bearer header the authenticated endpoints require
def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ##################################################################
# get brain
# fetch the full brain snapshot
def get_brain(base_url: str) -> dict:
    response = httpx.get(f"{base_url}/v1/brain", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


# ##################################################################
# probe
# measure a continuation's logprob under the live model
def probe(base_url: str, prompt: str, continuation: str, token: str) -> dict:
    body = {"prompt": prompt, "continuation": continuation}
    response = httpx.post(f"{base_url}/v1/brain/probe", json=body, headers=_auth(token), timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


# ##################################################################
# send feedback
# submit a human reward for a trace
def send_feedback(base_url: str, trace_id: str, reward: float, token: str,
                  source: str | None = None, note: str | None = None) -> dict:
    body = {"trace_id": trace_id, "reward": reward, "source": source, "note": note}
    response = httpx.post(f"{base_url}/v1/feedback", json=body, headers=_auth(token), timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


# ##################################################################
# format status
# a short human-readable rendering of the brain snapshot
def format_status(data: dict) -> str:
    overlay = data["overlay"]
    paused = data["paused"]
    updates = data["updates"]
    return "\n".join([
        f"model: {data['model_path']}",
        f"updates: {updates['counts']}",
        f"cumulative_reward: {updates['cumulative_reward']:.4f}",
        f"queue_depth: {data['queue_depth']}",
        f"overlay_norm: {overlay['total_norm']:.4f} across {overlay['adapter_count']} adapters",
        f"paused: {paused['flag']} ({paused['reason']})",
        f"uptime_s: {data['uptime_s']:.1f}",
    ])


# ##################################################################
# show status
# the `engram status` command: fetch the brain and print it as text or json
def show_status(args) -> int:
    base_url = _base_url(load_config())
    try:
        data = get_brain(base_url)
    except httpx.HTTPError as error:
        print(f"could not reach engram at {base_url}: {error}")
        return 1
    print(json.dumps(data, indent=2) if getattr(args, "json", False) else format_status(data))
    return 0
