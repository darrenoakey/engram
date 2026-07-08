# engram

A self-modifying local inference engine for **Ornith-1.0-9B** on Apple Silicon (MLX).

Engram is an OpenAI-compatible inference service that **changes its own weights as a
consequence of inference**. Every turn it generates is recorded; tool outcomes and user
feedback become a scalar reward; a guarded online gradient update reinforces the decisions
that succeeded and punishes the ones that failed — into a low-rank plastic overlay that is
periodically consolidated into the base weights themselves. Learned state survives restarts.

This is not retrieval or a prompt-memory trick: the model's parameters move. No shipping
system does weight-level online learning during serving with persistence and consolidation —
engram's contribution is doing it *safely*, behind layered drift guards borrowed from the
test-time-training, continual-learning, and online-RL literatures (see `DESIGN.md §0`).

## Quick start

```
./run serve                 # start the service on 127.0.0.1:8500 (loads the 9B)
./run status                # brain status: updates, overlay norm, checkpoints, drift
./run proof --rounds 6      # demonstrate learning end-to-end against a running server
./run check                 # quality gate: ruff + full test suite (real 0.8B model)
```

**Chat with it:** open **http://127.0.0.1:8500/** — a plain chatbot that keeps the
conversation and quietly individuates as you talk. The header pill shows
`N noticed · M learned`; click it for the memory drawer (what it learned, a
**Consolidate** button, and a **Prove recall** button).

**Make it consolidate what it noticed into learned facts (run a "dream"):**
```
curl -sX POST http://127.0.0.1:8500/v1/brain/dream -H "Authorization: Bearer $(./run token)"
```

Point any OpenAI client at `http://127.0.0.1:8500/v1`. To teach it, send a reward:

```
curl -s http://127.0.0.1:8500/v1/feedback \
  -H "Authorization: Bearer $(./run token)" \
  -d '{"trace_id":"<from the chat response>","reward":-1.0}'
```

Tool results flowing back in as `role:"tool"` messages are scored and learned from
automatically — no client change needed.

## How the learning loop works

engram has **two learners on one plastic overlay** (full detail in **LEARNING.md**):

**A — behaviour (the reward loop):** reward good/bad *decisions*, training on the model's own output.
```
chat ─▶ generate (eval mode) ─▶ Trace{tokens, per-token logprobs, spans}
feedback / tool outcome ─▶ scalar reward ─▶ update queue (GPU yields to inference)
   worker: snapshot ▶ teacher-forced grads on overlay only ▶ top-k mask ▶ norm clip
           ▶ AdamW step ▶ delta cap ▶ KL gate (breach ⇒ restore snapshot) ▶ journal
   canary probe after every negative update ▶ 2 breaches ⇒ rollback to last-good checkpoint
consolidate ▶ merge overlay into bf16 master ▶ requantize serving base ▶ KL-gated swap
```

**B — knowledge (ambient individuation):** absorb *the user* from ordinary chat, training on the user's words.
```
each turn ─▶ peak-surprise gate ─▶ log experience + gentle overlay absorb (felt by day)
dream (POST /v1/brain/dream) ─▶ corroborate durable facts ─▶ self-edit into Q→A knowledge
           ─▶ strong absorb ─▶ health-gate (cold-recall probe + entropy/sycophancy sentinels)
           ─▶ commit (learned) or revert (nothing durable) ─ atomic, provenance-logged
```

- **Positive reward**: reward-scaled cross-entropy — push up the tokens the model chose.
- **Negative reward**: bounded, self-saturating unlikelihood — push them down without the
  "squeezing" collapse that naive negative gradients cause.
- Every update is anchored by a KL term to the frozen base, gradient-masked to its top 30%,
  norm-clipped, delta-capped, and rejected outright if it moves the distribution too far.

## Documentation

- **LEARNING.md** — ⭐ start here: the complete reference for **how engram learns** — both
  learners, the guarded update step, the loss math, the config, how to operate it, the code
  map, the empirical findings, and where to take it next. Written for a fresh maintainer.
- **DESIGN.md** — the behaviour (reward) loop in depth: architecture, update math, research
  pedigree, per-module contracts.
- **INDIVIDUATION.md** — the knowledge (ambient individuation) loop in depth, with the live
  results and open problems.
- **STATUS.md** — current build state.
- **AGENTS.md** — operating manual and the non-obvious facts that bite.

## Requirements

Apple Silicon (developed on M1 Max, 64 GB), Python 3.12+. `./run` bootstraps an isolated venv.
Model weights live under `/Volumes/Gumby/models/` (see STATUS.md). No environment variables;
config is `local/config.toml`, the API token lives in the macOS keychain.
