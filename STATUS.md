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

## Build status: DEPLOYED AND PROVEN ON THE LIVE 9B

All five code partitions are implemented, reviewed, committed, and passing on the real 0.8B
test model. **124 tests pass, ruff clean, warnings-as-errors.** ~3,500 LOC + ~2,600 test LOC.
Published at https://github.com/darrenoakey/engram. Running under `auto` as `engram` on the 9B.

**Proven end-to-end on the live Ornith-1.0-9B:**
- Reinforcement raises a rewarded continuation's logprob; punishment lowers it (multiple
  rounds=4/5 proof passes).
- The 60-prompt canary stays within KL budget (mean_kl ~0.0003) with zero match failures and
  no spurious rollbacks.
- Learned overlay survives a graceful restart bit-exactly (probe delta 0.00000, overlay norm
  preserved) via the shutdown checkpoint.

Five bugs were found and fixed by running end-to-end on the real 9B (none reproducible on the
0.8B): proof phase-coupling, runaway generation length, silently-dropped feedback on
reasoning-only turns, spurious canary rollbacks, and learning lost on restart. See the git log.

| Area | Files | Tests | State |
|---|---|---|---|
| `common/` (config, store, identity) | 3 | 12 | ✅ committed `64eff9e` |
| `engine/` (model_host, generation, tool_parser, trace) | 4 | 34 | ✅ committed `53515bd` |
| `plasticity/` (adapter, losses, updater, guards, journal, checkpoints, replay, consolidate) | 8 | 31 | ✅ committed `ef7065d` |
| `evaluation/canary` (drift probe) | 2 | 12 | ✅ committed `5aa19a8` |
| `server/` (openai_api, feedback_api, brain_api, work_queue, app, client) | 6 | 26 | ✅ committed `9c77872` |
| `evaluation/proof` (E2E demonstration) | 1 | 3 (in eval count) | ✅ committed `9fbdf45` |

### Ambient self-individuation: SHIPPED and PROVEN on the 9B (INDIVIDUATION.md)
The *knowledge* half of "interaction changes the model" is built and live. From ordinary
unlabelled use — no commands, no explicit teaching — engram absorbs its user into the weights:
a peak-surprise gate picks turns worth learning (training on the USER's tokens, never its own →
collapse-proof), a volatile overlay absorbs by day, and a guarded `/v1/brain/dream` corroborates
durable facts, self-edits them into assistant-knowledge, absorbs them, and health-gates the night
(commit or revert). **Proven live:** told "I'm allergic to shellfish" / "My name is Darren" once
in chat, a fresh empty-context session recalls *shellfish* and *Darren*; a one-off role-play is
dropped, not persisted. New `src/individuation/` package + wiring; ~150 tests green. The honest
frontier (§14): the 9B's "I'm stateless" reasoning fights absorbed facts — v1 crosses it for clear
personal facts with stronger consolidation; robustly overriding it for arbitrary knowledge is the
targeted-editing stage (v2). Live tuning in gitignored `local/config.toml`.

### Original next steps (still open)

### Remaining / optional next steps
- **Consolidation on the 9B**: exercise a real `/v1/brain/consolidate` (heavy: dequantize-merge-
  requant the 18.8GB master; canary-gated with auto-revert), then confirm the swapped generation
  serves. The path is tested on the 0.8B; it has not yet been run against the 9B master.
- **Punishment margin**: the deliberately-gentle punishment (λ_neg=0.5) is near the noise floor
  at rounds≤3 on the 9B; use rounds≥4 for a clean proof. Not a bug — a conservative-by-design
  negative update. Raising λ_neg would need fresh squeezing-pathology evidence first.
- **Real agent traffic**: point a tool-using client at the OpenAI endpoint so tool outcomes
  auto-score and drive learning over real work (the reason auto tool-scoring exists).

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
