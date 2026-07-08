# engram — status & handoff

Living snapshot of where the build is, so a fresh session (or person) can pick it up cold.
Canonical design is **DESIGN.md**; day-to-day gotchas are **AGENTS.md**. This file is *state*.

Last updated: 2026-07-08.

## What engram is (one paragraph)

A local inference service that hosts **Ornith-1.0-9B** (Qwen3.5 hybrid, MLX) on an M1 Max and
**changes its own weights as a result of inference**. Every generated turn is recorded as a
trace; tool outcomes and user feedback produce a scalar reward; a background worker applies a
guarded online gradient update to a low-rank plastic overlay — reinforcing good decisions,
punishing failures — and periodically consolidates the overlay into the base weights. Learned
state persists across restarts. Full research pedigree and the exact update math are in DESIGN.md.

## Build status: CODE COMPLETE, LOCALLY GREEN, NOT YET DEPLOYED

All five code partitions are implemented, reviewed, committed, and passing on the real 0.8B
test model. **115 tests pass, ruff clean, warnings-as-errors.** ~3,400 LOC + ~2,100 test LOC.

| Area | Files | Tests | State |
|---|---|---|---|
| `common/` (config, store, identity) | 3 | 12 | ✅ committed `64eff9e` |
| `engine/` (model_host, generation, tool_parser, trace) | 4 | 34 | ✅ committed `53515bd` |
| `plasticity/` (adapter, losses, updater, guards, journal, checkpoints, replay, consolidate) | 8 | 31 | ✅ committed `ef7065d` |
| `evaluation/canary` (drift probe) | 2 | 12 | ✅ committed `5aa19a8` |
| `server/` (openai_api, feedback_api, brain_api, work_queue, app, client) | 6 | 26 | ✅ committed `9c77872` |
| `evaluation/proof` (E2E demonstration) | 1 | 3 (in eval count) | ✅ committed `9fbdf45` |

### Remaining work (the "continue from here" list)
1. **Deploy under `auto`** as service `engram` (`~/bin/engram serve`). First boot loads the 9B
   4-bit base and captures the 60-prompt canary baseline against the *original* base (one-time,
   a few minutes). Verify `GET /v1/brain` responds and a chat completion works.
2. **Live proof on the 9B**: `engram proof --rounds 6`. Expect reinforcement logprob UP,
   punishment DOWN, stability not-paused, RESULT PASS. Wider margins than the 0.8B (the 9B's
   per-prompt reasoning is diverse, so the shared-boilerplate contention that forced
   punishment-first ordering is a 0.8B pathology — the ordering stays correct regardless).
3. **Prove persistence live**: `engram proof` → `auto restart engram` → re-probe the same
   continuations; learned overlay must reload from checkpoint (values within ~1e-3).
4. **Push to GitHub** (repo does not exist remotely yet — `/publish` or `gh repo create`).
5. Optional: exercise a real `/v1/brain/consolidate` on the 9B (heavy: dequantize-merge-requant
   the 18.8GB master; canary-gated with auto-revert), then confirm the swapped generation serves.

## How to run it

```
cd ~/src/engram
./run check                 # full gate: ruff + 115 tests (~2 min, loads 0.8B repeatedly)
./run serve                 # start service on 127.0.0.1:8500 (loads the 9B)
./run status                # GET /v1/brain, pretty-printed
./run proof --rounds 6      # end-to-end learning demonstration against a running server
./run token                 # print the API bearer token
```

Auth token for `/v1/feedback` and `/v1/brain/*` POSTs is minted on first boot into the macOS
keychain (service `engram`, account `api-token`); `./run token` prints it. There are NO env vars.

### API surface (OpenAI-compatible + engram control plane)
- `POST /v1/chat/completions` — streaming/non-streaming; `reasoning_content` split out; qwen3_xml
  tool calls surfaced as OpenAI `tool_calls`; response carries `engram.trace_id` (+ header
  `X-Engram-Trace`). Incoming `role:"tool"` messages are auto-scored and enqueued as updates.
- `GET /v1/models`
- `POST /v1/feedback {trace_id, reward∈[-1,1], source?, note?}` — Bearer auth; enqueues an update.
- `GET /v1/brain` — model generation, update stats, queue depth, pause flag, overlay norm,
  recent checkpoints, uptime. `GET /v1/brain/journal?limit=`.
- `POST /v1/brain/{checkpoint,rollback,probe,consolidate}` — Bearer auth. `probe {prompt,
  continuation}` returns the continuation's logprob under the live overlay (the learning ruler).

## Models on disk (already downloaded)
- `/Volumes/Gumby/models/ornith-9b-4bit` — serving base (5.95 GB)
- `/Volumes/Gumby/models/ornith-9b-bf16` — consolidation master (18.8 GB)
- `/Volumes/Gumby/models/qwen3.5-0.8b-4bit` — pytest model (0.65 GB, same qwen3_5 architecture)
- Consolidation generations will be written under `/Volumes/Gumby/models/engram-generations/`.

## Config & data
- Defaults live in code (`common/config.py`); override via `local/config.toml` (gitignored).
  The council-ratified plasticity numbers (λ_neg 0.5, β_kl 0.05, LRs, top-k 0.3, KL budgets)
  are deliberate — change only with evidence.
- Runtime state under `local/data/` (gitignored): `journal.jsonl`, `traces/`, `checkpoints/`,
  `canary/`, `replay.json`, `current_base.json` (consolidation pointer).

## Key decisions & where they came from
- **Signal-gated updates, never blind per-turn self-training** — avoids the self-distillation
  overconfidence spiral (council; SEAL forgot after ~8 ungated edits).
- **Reward-scaled CE (positive) + bounded unlikelihood `-log(1-p)` (negative), λ_neg=0.5** — the
  only loss family that works at batch=1 online; low λ_neg avoids the negative-gradient
  "squeezing" pathology (Ren & Sutherland 2024).
- **KL-anchor + top-k mask + norm caps + replay + canary/rollback** — layered drift guards;
  ROME/MEMIT collapse was norm blowup, so caps are non-negotiable.
- **4-bit serving + bf16 master for consolidation** — requantization erases sub-grid deltas.
- **DeltaNet projections frozen in v1** — the `in_proj_*` naming/gate-packing is the bug surface;
  overlay targets middle-stack MLP + full-attention q/k/v/o only.

The 8-agent research corpus and 5-seat council record that produced these are summarized in
DESIGN.md §0. Review fixes made during integration: `worker_error` journaling so a poisoned job
can't silently kill the learning loop; consolidation is canary-KL-gated with auto-revert that
restores the pre-dream overlay; "0 failed" test output no longer auto-scores as a tool failure.
