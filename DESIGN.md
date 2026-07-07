# engram — a self-modifying local inference engine

Engram hosts **Ornith-1.0-9B** (Qwen3.5 hybrid architecture) on Apple Silicon via MLX and
**changes its own weights as a consequence of inference**: decisions the model makes are
reinforced when they succeed, and punished when they fail — driven by tool outcomes and
user feedback. Learned state persists across restarts and is periodically consolidated
into the base weights themselves.

Nobody ships this today. Research prototypes (MIT SEAL, In-Place TTT, dynamic evaluation)
either discard their adaptations, binary-gate them, or reset at document boundaries.
Engram's contribution is the persistent, guarded, consolidated loop.

## 0. Design pedigree (research-backed decisions)

Every load-bearing choice below traces to verified sources (see the council + research
records in the repo history). The short version:

| Decision | Why |
|---|---|
| Signal-gated updates, never blind per-turn self-training | Self-distillation spiral; SEAL saw forgetting after ~8 ungated edits |
| Reward-scaled CE (positive) + bounded unlikelihood `-log(1-p)` (negative) | Only family that works at batch=1 online; unlikelihood self-saturates as p→0 |
| λ_neg = 0.5, never 2.0 | "Squeezing" pathology: negative gradients displace mass onto unrelated tokens (Ren & Sutherland 2024; Razin 2024) |
| KL(live‖base) anchor β=0.05 in every update | TRL default; bounds distribution drift per update |
| ≥1 positive replay span mixed into every update batch | 1% replay measurably cuts forgetting; mixed-sign batches counter squeezing |
| Top-k gradient masking (30%) + global norm clip + per-update delta cap | MoFO/FGGM show masked updates drift less; ROME/MEMIT collapse was norm blowup |
| LoRA rank 8 on middle-stack MLP + full-attention q/k/v/o; DeltaNet projections frozen in v1 | O-LoRA: rank 2 ≈ rank 16; DeltaNet in_proj_* naming/packing is the bug surface |
| 4-bit serving base + bf16 master for consolidation | Requantization snaps sub-grid deltas to zero; master keeps full fidelity |
| Canary probes + checkpoint ring + auto-rollback | 50–200 prompts, 2-of-N breach rule; every update snapshots first |
| Fresh AdamW per boot (no optimizer-state persistence) | Simplifies persistence; single-step online updates barely use momentum |
| Authenticated /v1/feedback | An open feedback endpoint is a weight-poisoning API |

Hardware envelope (measured on this M1 Max 64GB, mlx-lm 0.31.3, 4-bit base, grad-checkpointed
adapter-only steps): T256 ≈ 11.6s / 12.3GB, T512 ≈ 26.1s / 18.2GB, T1024 ≈ 64s / 30GB, T2048 OOM.
Therefore: **teacher-forced update spans are capped at 256 tokens by default (512 max)**, and
multi-span batches use sequential gradient accumulation, never concatenation.

Architecture facts that must not be violated:
- `model.train()` routes Gated-DeltaNet through the differentiable ops scan; `model.eval()`
  uses the fast Metal kernel (no VJP). Training steps REQUIRE train mode.
- 24 of 32 layers are DeltaNet: projections are `in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj`.
  8 layers (`layer_types[i] == "full_attention"`) have `q_proj/k_proj/v_proj/o_proj`, and
  `q_proj` output width is DOUBLE (gate-packed, `attn_output_gate: true`). Never assume shapes
  by name; read them from the module.
- Per-token logprobs come free from `mlx_lm` `generate_step`/`stream_generate` (full log-softmax).
- Generation KV caches are built by the non-differentiable kernel — update steps re-run the
  span teacher-forced in train mode; never backprop through generation caches.
- Chat template keeps `<think>` blocks in history; tool calls are qwen3_xml XML
  (`<tool_call>\n<function=NAME>\n<parameter=P>\nvalue\n</parameter>\n</function>\n</tool_call>`).
- Sampling defaults (thinking mode): temp 0.6, top_p 0.95, top_k 20.

## 1. System shape

```
client ──POST /v1/chat/completions──▶ server ──▶ ModelHost.generate (eval mode, GPU lock)
                                        │              │
                                        │              ▼ Trace (tokens, logprobs, spans) → trace store
                                        │
client ──POST /v1/feedback {trace_id,reward}──▶ update queue (background worker, GPU lock between requests)
   tool results auto-scored ────────────┘        │
                                                  ▼ Updater: snapshot → grads (train mode) → guards → step → journal
                                                  │
                              canary probe ◀──────┤ (after every negative update; every N updates)
                              breach → rollback to last-good checkpoint
                                                  │
POST /v1/brain/consolidate ──▶ dream job: merge overlay Δ into bf16 master → requantize →
                               KL-gate vs live → atomic swap serving base → reset overlay
```

The journal (append-only JSONL) is the system of record. Weights are derived artifacts.

## 2. Repository layout (python-dev skill standards apply in full)

```
run                      # executable facade (venv bootstrap → src/main.py)
requirements.txt
README.md  DESIGN.md
local/                   # gitignored: config.toml, data/ (journal, traces, checkpoints, canary)
output/testing/          # gitignored test artifacts
src/
  main.py                # subcommands: serve | status | consolidate | rollback | probe
  common/   config.py store.py identity.py
  engine/   model_host.py generation.py tool_parser.py trace.py
  plasticity/ adapter.py losses.py updater.py guards.py journal.py checkpoints.py replay.py consolidate.py
  server/   app.py openai_api.py feedback_api.py brain_api.py work_queue.py
  evaluation/ canary.py canary_prompts.py proof.py
```

Every `x.py` has `x_test.py` beside it. All tests are REAL — they load the actual
0.8B test model (`/Volumes/Gumby/models/qwen3.5-0.8b-4bit`, same qwen3_5 architecture)
and run actual generation/updates. The 9B is exercised by the proof harness and the live
service, not by pytest. Tests share one Metal GPU: pytest runs serially (no -n).

## 3. Configuration (`src/common/config.py`)

Frozen dataclass `EngramConfig`, loaded by `load_config()` from `local/config.toml`
over defaults. NO environment variables. Secrets (feedback auth token) via keyring
(`common/identity.py`: service "engram", account "api-token"; create-on-first-boot with
`secrets.token_urlsafe`, readable via `engram status`).

Defaults:
```toml
[model]
serve_path   = "/Volumes/Gumby/models/ornith-9b-4bit"     # current serving base (generation dirs live next to it)
master_path  = "/Volumes/Gumby/models/ornith-9b-bf16"     # bf16 master, consolidation target
test_path    = "/Volumes/Gumby/models/qwen3.5-0.8b-4bit"  # used by pytest only
base_generations_dir = "/Volumes/Gumby/models/engram-generations"

[server]
host = "127.0.0.1"
port = 8500

[sampling]
temperature = 0.6
top_p = 0.95
top_k = 20
max_tokens = 4096

[plasticity]
enabled = true
self_reinforce = "gated"       # off | gated | always   (gated: only turns with >=1 non-negative signal)
lora_rank = 8
lora_alpha = 16
lora_scope = "mid_mlp_full_attn"  # MLP gate/up/down on middle-stack layers + q/k/v/o on full-attention layers; DeltaNet frozen
mid_layers = [8, 28]           # inclusive band of the 32 layers that receives adapters
lr_reinforce = 1e-6
lr_reward = 5e-6
lambda_neg = 0.5               # punishment scale (bounded unlikelihood); asymmetry via triggering, not magnitude
beta_kl = 0.05                 # KL(live||base) anchor weight inside every update loss
max_span_tokens = 256          # teacher-forced span cap (hard max 512)
replay_spans = 1               # positive replay spans mixed into every update (grad accumulation)
topk_grad_fraction = 0.3       # only the top 30% of overlay grads (by magnitude, per tensor) update
grad_clip_norm = 1.0
delta_frobenius_cap = 0.05     # per-tensor per-update delta cap (rescale if exceeded)
adapter_norm_ceiling = 5.0     # total overlay norm ceiling; breach pauses plasticity
update_kl_budget = 0.5         # mean nats over span; breach → undo this step (restore snapshot)
include_think_tokens = false   # credit answer + tool_call spans only

[guards]
canary_every = 20              # also after EVERY negative update
canary_kl_budget = 0.15        # mean nats vs stored base distribution on canary set
canary_breaches_to_rollback = 2
checkpoint_every = 10
checkpoint_ring = 20

[feedback]
auto_tool_scoring = true
tool_success_reward = 0.3
tool_failure_reward = -0.5
```

## 4. Module contracts

### common/store.py
Atomic persistence helpers (write temp file + `os.rename`), JSON/JSONL/bytes; data-dir
layout accessors (`traces_dir()`, `journal_path()`, `checkpoints_dir()`, `canary_dir()`).
Every writer in the codebase goes through this module.

### engine/model_host.py — `class ModelHost`
- `__init__(config, model_path)` → `mlx_lm.load()`, attach plastic overlay
  (`plasticity.adapter.attach_overlay`), `model.eval()`.
- `generate(messages, tools, sampling, on_token) -> Trace` — applies chat template
  (`enable_thinking` on), streams via `mlx_lm.stream_generate`, records per-token ids +
  logprobs, parses spans (think / answer / tool_call via `tool_parser`), returns a `Trace`.
- `span_logprobs(token_ids, span, adapters_enabled: bool) -> mx.array` — teacher-forced
  forward in eval mode over a stored trace prefix+span; used for probes and KL baselines.
- `gpu_lock` — a single `threading.Lock` serializing ALL Metal work (generation, updates,
  probes). Exposed for the work queue.
- Owns train/eval mode transitions; callers never touch `model.train()` directly.

### engine/trace.py — `@dataclass Trace`
`trace_id` (uuid), `created_at`, `token_ids` (full sequence), `gen_start` (index),
`logprobs` (chosen-token, generation only), `spans` (list of `Span(kind, start, end)`,
kind ∈ {think, answer, tool_call}), `tool_call_ids` (map tool_call_id → span index),
`sampling`, `feedback` (list of applied rewards). Persisted as gzipped JSON via store;
`save/load/list_recent`.

### engine/tool_parser.py
qwen3_xml parse + render: `parse_tool_calls(text) -> list[ToolCall]` (name, arguments
dict, char offsets), `openai_tool_calls(...)` conversion, and `score_tool_result(content)
-> float` heuristic (error/traceback/exit-code patterns → failure reward; else success).

### plasticity/adapter.py — the plastic overlay
`class PlasticLinear(nn.Module)` wrapping any Linear/QuantizedLinear:
`y = base(x) + (alpha/rank) * (x @ A) @ B` when `overlay.enabled` (module-level flag),
A ~ N(0, 0.02) frozen-shape, B zero-init. Shapes read from the wrapped module
(`base.weight` may be quantized: derive in/out dims the way mlx-lm's LoRA does).
`attach_overlay(model, config) -> Overlay`:
- selects targets: for layer index i in `mid_layers` band — MLP `gate_proj/up_proj/down_proj`
  always; `q_proj/k_proj/v_proj/o_proj` when `layer_types[i] == "full_attention"`.
- `Overlay` API: `trainable_parameters()`, `enabled` toggle (context manager
  `disabled()` for base-logit forwards), `snapshot()/restore(snap)` (in-memory),
  `save(path)/load(path)` (safetensors), `tensor_norms()`, `total_norm()`, `reset()`,
  `merge_deltas() -> dict[str, mx.array]` (alpha-scaled A@B per wrapped weight path,
  for consolidation).

### plasticity/losses.py
All losses masked to the credit span, per-token weights `w_t`:
- `positive_loss(logits, targets, w, reward)` = `-reward * Σ w_t·logp(y_t) / Σ w_t`
- `negative_loss(logits, targets, w, reward)` = `-|reward| * λ_neg * Σ w_t·log1mexp(logp(y_t)) / Σ w_t`
  where `log1mexp` is the numerically-stable `log(1 - e^x)` for x<0 (branch at log 0.5).
  Self-saturating: gradient → 0 as p(y_t) → 0.
- `kl_anchor(live_logits, base_logits)` = mean over span of full-vocab
  `Σ p_live·(logp_live − logp_base)` (computed in fp32).
- Total per update: `class_loss + beta_kl * kl_anchor`.

### plasticity/updater.py — `class Updater`
Single entry: `apply(trace, reward, source, kind)` where kind ∈ {reinforce, reward}.
Steps (all under gpu_lock, train mode, grad_checkpoint enabled):
1. Select credit span(s): tool_call + answer spans, newest first, ≤ `max_span_tokens`.
2. `overlay.snapshot()`.
3. For primary span + `replay_spans` positive spans from replay buffer: teacher-forced
   `nn.value_and_grad` over overlay params only; accumulate grads (sequential, never
   concatenated batches). Base logits for the KL term via `overlay.disabled()` forward
   of the same span (eval mode, no grad).
4. Guard pipeline (guards.py): top-k mask per tensor → global norm clip → optimizer step
   (fresh AdamW per boot, lr by kind) → per-tensor delta cap (rescale) →
   post-step span KL check: if mean KL > `update_kl_budget`, `overlay.restore(snapshot)`
   and journal a `rejected_update`.
5. Journal `update` event: trace_id, kind, reward, span sizes, loss values, grad norm,
   delta norms, span KL, wall-clock ms.
6. Trigger canary probe (async job) if kind==reward and reward<0, or every `canary_every`.

### plasticity/guards.py
`topk_mask(grads, fraction)`, `clip_global(grads, max_norm)`, `cap_delta(before, after, cap)`,
`class CanaryGuard`: runs `evaluation.canary.probe`, tracks breach count in a window,
`should_rollback() -> bool` (≥2 breaches), and `adapter ceiling` check → sets
`plasticity_paused` flag surfaced in /v1/brain.

### plasticity/journal.py
Append-only JSONL at `local/data/journal.jsonl` via store atomic append (open a+, single
process). Event types: update, rejected_update, rollback, checkpoint, consolidate,
canary. `tail(n)`, `stats()` (counts by type, last canary, cumulative reward).

### plasticity/checkpoints.py
`save(overlay, meta) -> checkpoint_id` (safetensors + json in ring of `checkpoint_ring`),
`restore(overlay, checkpoint_id=None)` (default: last good — most recent whose meta shows
clean canary), `list()`. Auto-checkpoint every `checkpoint_every` updates (updater calls in).

### plasticity/replay.py
Replay buffer of positive spans: every trace with reward > 0 contributes its credit span
(token ids); ring buffer (json via store, cap ~200 spans). `sample(k)`; seeded from canary
prompts' reference continuations when empty (cold start).

### plasticity/consolidate.py
`dream(host, overlay, config) -> report`:
1. Pause serving (work queue drains; brain reports `consolidating`).
2. `deltas = overlay.merge_deltas()` — keyed by weight path.
3. Stream bf16 master shards; for each targeted weight: `w += delta` (fp32 math, cast back);
   write new master generation (atomic dir swap `master-gen-N`).
4. Requantize targeted layers to a new serving base generation dir (mlx-lm convert-style,
   4-bit affine gs64, reusing the untouched quantized tensors for non-targeted weights).
5. KL-gate: load new base + EMPTY overlay lazily; canary probe vs the pre-swap live model;
   if mean KL beyond `canary_kl_budget` × 2 or breaches → abort (keep old base, keep overlay).
6. Swap: update `serve_path` current-generation symlink, reload host, `overlay.reset()`,
   checkpoint, journal `consolidate` with report. Keep previous generation for rollback (ring of 2).

### server/ (FastAPI + uvicorn)
- `openai_api.py`: POST `/v1/chat/completions` — OpenAI-compatible (messages, tools,
  stream SSE + non-stream), reasoning split into `reasoning_content`, qwen3_xml →
  `tool_calls`. Response carries `"engram": {"trace_id": ...}` in body (and
  `X-Engram-Trace` header). On each request, scan incoming `role:"tool"` messages: match
  `tool_call_id` to stored traces; if unscored and `auto_tool_scoring`, score via
  `tool_parser.score_tool_result` and enqueue feedback (this is how tool outcomes train
  the model without any client change). GET `/v1/models`.
- `feedback_api.py`: POST `/v1/feedback` `{trace_id, reward∈[-1,1], source?, note?}` —
  Bearer token (keyring) required. Enqueues update job. 404 unknown trace; 409 if
  plasticity paused.
- `brain_api.py`: GET `/v1/brain` (model generation, update counts by type, cumulative
  reward, adapter norms, last canary, queue depth, paused flag, checkpoints, uptime);
  GET `/v1/brain/journal?limit=`; POST `/v1/brain/rollback {checkpoint_id?}`;
  POST `/v1/brain/consolidate`; POST `/v1/brain/checkpoint`; POST `/v1/brain/probe
  {prompt, continuation}` → logprob of continuation under live model (the proof harness's
  measuring stick). All POSTs authenticated.
- `work_queue.py`: one background worker thread; `queue.Queue` of update jobs; generation
  has priority via gpu_lock fairness — worker only claims the lock when no request is
  waiting (server tracks in-flight/waiting count); worker uses `Event.wait(timeout)`
  loops, never busy-spin.
- `app.py`: wiring, lifespan (load model, canary baseline on first boot, restore adapter
  from latest checkpoint), setproctitle("engram").

### evaluation/
- `canary_prompts.py`: 60 fixed prompts (coding, tool-call formatting, general knowledge,
  instruction following) + 12 with exact-match expected answers.
- `canary.py`: `baseline(host)` — store per-prompt base logprob stats (first boot, base
  model, overlay disabled); `probe(host)` — mean per-token KL vs stored baseline on the
  first 64 generated-token positions teacher-forced over stored reference continuations +
  exact-match checks; returns `CanaryReport(mean_kl, per_prompt, match_failures)`;
  journaled.
- `proof.py`: the end-to-end demonstration against a LIVE server (used on 0.8B in tests,
  9B in deployment): (a) reinforcement shifts a target continuation's logprob up
  measurably; (b) punishment drops an elicited bad behavior's logprob AND an alternative
  rises; (c) canary KL stays within budget throughout; (d) service restart → probes
  return identical values (persistence); prints a scoreboard.

## 5. What v1 explicitly does NOT do
- No DeltaNet-projection adapters (v2 after stability evidence).
- No dream-refit from journal (consolidation merges the current guarded overlay; refit is v1.1).
- No multi-session fast-weight state persistence (RW-TTT territory).
- No MTP head usage (unconsumed by mlx-lm for this arch).
- No multi-user batched serving; one request at a time, queue the rest.

## 6. Proof-of-life criteria (the definition of done)
1. Full pytest suite green (0.8B real-model tests), ruff clean, zero warnings.
2. Service runs under `auto` as `engram`, survives restart with learned state intact.
3. Proof harness against the live 9B: reinforcement shift, punishment suppression,
   canary within budget, persistence across restart — all demonstrated with numbers.
