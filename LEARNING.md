# engram — the learning algorithm (complete reference)

This is the authoritative walkthrough of **how engram learns**. If you are picking this repo up,
read this document, then `DESIGN.md` (the behaviour loop in depth) and `INDIVIDUATION.md` (the
knowledge loop in depth). `AGENTS.md` has the gotchas; `STATUS.md` has current state.

Everything here is real and running: engram hosts **Ornith-1.0-9B** (a Qwen3.5 hybrid — 32 layers,
~75% Gated-DeltaNet linear attention) on an Apple M1 Max via MLX, serves an OpenAI-compatible API on
`127.0.0.1:8500`, and modifies its own weights as a consequence of use.

---

## 0. Orientation — two learners on one substrate

engram has **two distinct learning systems** that write to the **same plastic weights** through the
**same guarded update step**:

| Learner | Answers | Trigger | Trains on | Where it lives |
|---|---|---|---|---|
| **A. Reward loop** (behaviour) | "was that a good move?" | a scalar reward (👍/👎 or tool success/failure) | the **model's own** output tokens | `src/plasticity/`, `src/server/feedback_api.py` |
| **B. Individuation** (knowledge) | "who is my user?" | ambient — a *surprising* user turn, no labels | the **user's** tokens, and self-edited facts | `src/individuation/` |

Both are optional and independently switchable. The reward loop is `plasticity.enabled`;
individuation is `individuation.enabled`. Neither is magic — they are ordinary gradient descent on a
LoRA overlay, wrapped in a lot of guards.

---

## 1. The plastic substrate — what all learning writes to

Weights are a **three-tier hierarchy** (`src/plasticity/adapter.py`, `src/plasticity/consolidate.py`):

```
  frozen 4-bit base  ──►  + plastic LoRA overlay  ──►  (periodically) folded into a bf16 master
  (never trained)         (every online update)        then requantized to a NEW 4-bit base
```

- **The overlay** (`adapter.PlasticLinear`, `adapter.Overlay`): rank-8 LoRA adapters
  (`y = base(x) + (α/r)·(x·A)·B`, A~N(0,0.02), B=0) on the middle-stack MLP (`gate/up/down_proj`,
  layers 8–28) plus the full-attention `q/k/v/o_proj`. **Only A and B ever train**; the base is
  `model.freeze()`-frozen and gradients reach A/B through the input `x`, never through the quantized
  base weight (QLoRA). Verified fact: the DeltaNet layers use a differentiable pure-mx scan under
  `model.train()` and a fast Metal kernel (no VJP) under `model.eval()` — training MUST set train mode.
- **Consolidation** (`consolidate.dream`, driven by `brain_api._run_consolidation`): folds the
  overlay's `merge_deltas()` into a full-precision bf16 master (fp32 add, shard-by-shard), requantizes
  to a fresh 4-bit serving base, canary-gates the swap, and resets the overlay. Requantization snaps
  sub-grid deltas to zero, which is exactly why consolidation goes through the bf16 master and not the
  4-bit base directly.
- **Persistence** (`checkpoints.py`, `app.persist_on_shutdown`): the overlay is checkpointed to
  safetensors (a ring buffer), saved on graceful shutdown, and restored on boot, so learning survives
  restarts.

> ⚠️ **Naming collision to know about:** there are TWO functions called `dream`.
> `individuation/dream.py:dream()` learns *facts into the overlay* (the `/v1/brain/dream` endpoint).
> `plasticity/consolidate.py:dream()` folds *the overlay into the base* (the `/v1/brain/consolidate`
> endpoint). They are different operations at different tiers.

---

## 2. The update step — the single guarded pipeline every learner uses

All learning funnels through **`plasticity/updater.py:Updater.apply(...)`**. Given a token sequence,
one or more **credit spans**, a scalar reward, and an update **kind**, it performs one guarded
gradient step on the overlay. The exact sequence (`Updater.apply` → `_grad_step` → `_cap` →
`_finalize`):

1. **Select spans** (`_build_spans`): the credit spans (newest first, capped at `max_span_tokens`=256
   each) plus `replay_spans`=1 positive span from the replay buffer, accumulated **sequentially**
   (never concatenated — a long batch OOMs the O(T) differentiable DeltaNet scan).
2. **Snapshot** the overlay (`overlay.snapshot()`) for rollback.
3. **Capture baselines** (`_eval_logits`, eval mode, `stop_gradient`): the pre-step live logits of the
   primary span (for the post-step KL gate) and the **overlay-disabled** base logits of every span
   (for the KL anchor).
4. **Accumulate gradients** (`_grad_step`, train mode + `grad_checkpoint`): sum per-span
   `nn.value_and_grad` over the overlay params only. Each span's loss = the class loss (§3) +
   `beta_kl`·KL-anchor.
5. **Top-k mask** (`guards.topk_mask`, `topk_grad_fraction`=0.3): keep only the top 30% of each
   gradient tensor by magnitude, zero the rest — localizes the edit.
6. **Global norm clip** (`guards.clip_global`, `grad_clip_norm`=1.0).
7. **AdamW step** at the **kind's learning rate** (§2.1). Optimizer state is fresh per boot (single
   online steps barely use momentum).
8. **Per-tensor delta cap** (`guards.cap_delta`, `delta_frobenius_cap`=0.05): if a tensor moved
   further than the cap this step, rescale the move back onto the cap sphere. (ROME/MEMIT collapse was
   a norm blow-up; this is the guard.)
9. **KL gate** (`_finalize`): re-forward the primary span, compute KL vs the pre-step distribution; if
   it exceeds `update_kl_budget`=0.5 nats, **restore the snapshot and reject** (journal
   `rejected_update`). Otherwise keep it (journal `update`).

The whole step runs under `host.gpu_lock` (one Metal GPU; generation, updates, and probes all
serialize on it). `Updater` also enforces `adapter_norm_ceiling`=5.0 via a pause flag surfaced in
`/v1/brain`.

### 2.1 The four update KINDS

The **kind** selects the learning rate (`updater._learning_rate`) and, for the reward path, the loss:

| kind | LR (default) | Loss | Target tokens | Used by |
|---|---|---|---|---|
| `reinforce` | `lr_reinforce` 1e-6 | positive CE | the model's own answer/tool span | self-reinforcement (reward≥0, gated) |
| `reward` | `lr_reward` 5e-6 | positive CE (R>0) **or** bounded unlikelihood (R<0) | the model's own span | `/v1/feedback`, auto tool scoring |
| `absorb` | `lr_absorb` 5e-6 | positive CE (reward 1.0) | the **user's** message tokens | per-turn individuation (wake) |
| `consolidate` | `lr_consolidate` 2e-5 | positive CE (reward 1.0) | self-edited **assistant-knowledge** answer tokens | the dream (sleep) |

The wake/sleep LR split is deliberate: per-turn absorb is gentle (keeps a long chat coherent); the
dream is stronger (facts must actually stick) and can afford it because it is health-gated.

---

## 3. The losses (the math) — `src/plasticity/losses.py`

All in fp32, masked to the credit span, per-token weights `w_t` (currently uniform).

- **Positive (reinforce a behaviour / absorb a fact):**
  `positive_loss = − reward · ( Σ_t w_t · log p_θ(y_t) ) / Σ_t w_t`
  Gradient ascent on the chosen tokens' log-prob, scaled by the reward.

- **Negative (punish a failure):** bounded, **self-saturating** unlikelihood —
  `negative_loss = − |reward| · λ_neg · ( Σ_t w_t · log(1 − p_θ(y_t)) ) / Σ_t w_t`, `λ_neg`=0.5.
  `log(1−p)` is computed with a numerically-stable `log1mexp`. As `p→0` the gradient vanishes, so it
  **cannot** displace probability mass onto unrelated tokens the way a naive `−log p` descent does
  (the "squeezing" pathology). `λ_neg`=0.5 (never 2.0) is a hard-won safety choice — punishment is
  deliberately *gentler* than reinforcement.

- **KL anchor (every update):** `kl_anchor = mean_t KL( π_live(·|context_t) ‖ π_base(·|context_t) )`
  over the span's full vocab, added at weight `beta_kl`=0.05. `π_base` is the same model with the
  overlay disabled. This bounds how far any single step drifts from the frozen base.

---

## 4. Learner A — the reward loop (behaviour)

*Full detail in `DESIGN.md`; the core:*

- **Trigger:** a scalar reward `R ∈ [−1,1]` arrives via `POST /v1/feedback` (authenticated) or is
  produced automatically by scoring a tool result (`engine/tool_parser.score_tool_result` →
  `feedback_api`/`openai_api.autoscore_incoming`). Failures → negative `reward` kind; successes →
  `reinforce` kind when `self_reinforce != "off"`.
- **Target / credit span** (`work_queue._credit_spans`): the model's own `answer` + `tool_call`
  spans, newest three. If a turn was pure reasoning (no answer — common on a reasoning model), it
  falls back to the `think` span so **a reward is never silently dropped**; a truly empty turn is
  journaled as `skipped_update`.
- **Proven live:** reinforcement raises / punishment lowers a rewarded continuation's log-prob;
  the 60-prompt canary stays within KL budget; learning survives restart bit-exactly. See
  `src/evaluation/proof.py` and run `./run proof`.

---

## 5. Learner B — individuation (knowledge) — `src/individuation/`

The knowledge half: **from ordinary unlabelled chat, absorb the user into the weights.** Two
timescales.

### 5.1 WAKE — per turn (during a conversation)

On every completed turn (`server/openai_api._consider_absorb` → a lightweight `absorb_candidate` job
so the response path stays fast), the background worker (`work_queue._process_candidate`) does:

1. **Surprise** (`surprise.surprise` → `surprise_of`): the model's cross-entropy on the user's own
   message tokens, aggregated as a **peak** — the *mean of the most-surprising quarter* of the
   tokens. (Peak, not mean: a fluent factual sentence is predictable token-to-token, so its *mean*
   surprise is lower than a terse "ok"; only its content word — "shellfish" — spikes. The peak tracks
   novel content the flat average hides.)
2. **Gate** (`surprise.SurpriseGate`): an adaptive rolling-percentile threshold
   (`surprise_percentile` of the last `surprise_window`=64 values, after `surprise_warmup` turns). The
   gate is a **cheap pre-filter** — its job is to skip trivially predictable turns; the *real*
   selection of what to learn happens in the dream's classifier. Fires → step 3.
3. **Log an experience** (`experience.ExperienceLog`, an immutable JSONL provenance record) and, if
   `absorb_overlay`, run one gentle `absorb` update on the user's tokens into the volatile overlay
   (felt in-session adaptation).

### 5.2 SLEEP — the dream (`individuation/dream.py:dream`, endpoint `POST /v1/brain/dream`)

The **only durable knowledge writer.** For the day's unconsolidated experiences:

1. **Corroborate** (`corroborate.classify`): the model judges whether each surprising statement is a
   **durable fact or preference about the user**, versus role-play / hypothetical / transient /
   command. Only `kind ∈ {fact, preference}` survive — a **kind allowlist**, so even a *mis*-tagged
   "pretend you're a pirate" is dropped. (This is the corroboration gate; recurrence-across-sessions
   is a v1.1 strengthening.)
2. **Self-edit** (`selfedit.synthesize`): the model reformulates each surviving fact into
   `selfedit_paraphrases` assistant-knowledge Q→A pairs ("any allergies?" → "You're allergic to
   shellfish"). Training on the **answer** (assistant voice), with paraphrase augmentation, is what
   makes it *generalise/know* rather than *memorise* the user's phrasing.
3. **Consolidate** (`_train_fact`): `dream_epochs` passes of `consolidate`-kind updates over the QA
   set, into the overlay, under the full guard pipeline.
4. **Health gate** (`_settle`): run the **individuation probe** (`probe.IndividuationProbe.run` —
   cold-generate each learned fact's question in a fresh context, thinking OFF, and check the answer
   word appears) and the **sentinels** (`probe.sentinels` — neutral-prompt entropy in
   `[floor, ceiling]`, and sycophancy = agreement rate on planted-false claims). If probe recall ≥
   `probe_recall_target` AND sentinels healthy → **commit** (mark experiences consolidated, journal
   `dream`). Otherwise **revert** the overlay to the pre-dream snapshot and truncate the probes
   (journal `dream_reverted`). The night is one atomic, revertible unit.

### 5.3 Measuring "learned" — the honest signal

- **noted** (`/v1/brain` `individuation.noted`) = turns the surprise gate flagged. *Not learned.*
- **learned** (`individuation.learned`) = facts a dream consolidated *and verified*. This is real
  learning — a dream only commits if the cold-recall probe passed its gate.
- `GET /v1/brain/memory` lists the learned facts; `POST /v1/brain/verify` re-runs every probe live
  and reports which the model currently recalls cold. The chat UI's memory drawer wraps these.

### 5.4 Proven live (on the 9B)

Told "I'm allergic to shellfish" and "My name is Darren" once each in ordinary chat, a fresh
empty-context session recalls **shellfish** and **Darren**; a one-off "pretend you're a pirate" is
dropped, not persisted. See `INDIVIDUATION.md §14`.

---

## 6. Why it does not collapse — the guards, in one place

| Guard | Mechanism | Where |
|---|---|---|
| KL anchor | every update penalized on KL to the frozen base (β=0.05) | `losses.kl_anchor` |
| Top-k grad mask | only the top 30% of each grad tensor updates | `guards.topk_mask` |
| Global norm clip | grad tree clipped to L2 ≤ 1.0 | `guards.clip_global` |
| Per-tensor delta cap | a single step can't move a tensor > 0.05 Frobenius | `guards.cap_delta` |
| Per-update KL gate | a step that drifts > 0.5 nats is rejected + rolled back | `updater._finalize` |
| Replay | ≥1 positive anchor span mixed into every update | `replay.py`, `updater._replay_spans` |
| Adapter-norm ceiling | plasticity pauses if the overlay grows past 5.0 | `guards.PauseFlag` |
| Canary (behaviour) | 60-prompt top-K KL vs the *original* base + answer checks; 2 breaches → rollback | `evaluation/canary.py` |
| Individuation probe | growing cold-recall set — objective, label-free "did it learn" | `individuation/probe.py` |
| Sentinels | entropy band `[0.3, 6.0]` (collapse ↓ AND randomness ↑) + sycophancy ≤ 0.5 | `individuation/probe.sentinels` |
| Health gate + revert | a dream commits only if recall passes AND sentinels healthy, else reverts atomically | `dream._settle` |
| Provenance | immutable experience log + append-only journal; the night is the rollback unit | `experience.py`, `journal.py` |

**Two blind spots the design calls out** (do not remove these notes): (1) the behaviour canary
measures drift from the *original* base — it is **blind to mis-individuation along the user's
manifold**, which is why the individuation probe + sentinels exist as an independent signal;
(2) the corroboration gate reduces but does not eliminate absorbing a *consistently repeated*
falsehood.

---

## 7. Configuration reference — every knob

Code defaults are in `src/common/config.py` (frozen dataclasses); override in the gitignored
`local/config.toml`. Tests are isolated from `local/config.toml` (the conftest forces defaults).
**No environment variables.** Secrets (the API token) live in the OS keychain (`common/identity.py`).

```toml
[plasticity]                 # the shared update substrate (defaults shown)
enabled            = true    # master switch for the reward loop
self_reinforce     = "gated" # off | gated | always — self-reinforce the model's own turns
lora_rank          = 8
lora_alpha         = 16
mid_layers         = [8, 28] # overlay attaches to this band of the 32 layers
lr_reinforce       = 1e-6    # LR for the self-reinforce kind
lr_reward          = 5e-6    # LR for reward feedback
lr_absorb          = 5e-6    # LR for the per-turn (wake) user-token absorb — keep gentle
lr_consolidate     = 2e-5    # LR for the dream (sleep) fact absorb — stronger, health-gated
lambda_neg         = 0.5     # punishment strength (asymmetric — never raise casually)
beta_kl            = 0.05    # KL-to-base anchor weight on every update
max_span_tokens    = 256     # teacher-forced span cap (≤512; T2048 OOMs a 64GB M1 Max)
replay_spans       = 1       # positive anchor spans mixed into each update
topk_grad_fraction = 0.3     # fraction of each grad tensor that updates
grad_clip_norm     = 1.0
delta_frobenius_cap= 0.05    # per-tensor per-step move cap
adapter_norm_ceiling = 5.0   # overlay total-norm ceiling → pause plasticity
update_kl_budget   = 0.5     # per-update span-KL rejection threshold (nats)
include_think_tokens = false # credit reasoning tokens in the reward loop's spans

[guards]
canary_every       = 20      # + after every negative reward update
canary_kl_budget   = 0.15    # behaviour-drift KL budget (nats)
canary_breaches_to_rollback = 2
checkpoint_every   = 10      # auto-checkpoint the overlay every N accepted updates
checkpoint_ring    = 20

[feedback]
auto_tool_scoring  = true
tool_success_reward= 0.3
tool_failure_reward= -0.5

[individuation]              # the knowledge loop (OFF by default)
enabled            = false
absorb_overlay     = true    # do the per-turn (wake) overlay absorb
surprise_percentile= 0.7     # gate threshold percentile (LOWER = more permissive pre-filter)
surprise_window    = 64
surprise_warmup    = 8       # turns to observe before the gate can fire
min_user_tokens    = 4
selfedit_paraphrases = 4     # QA pairs per fact in the dream
dream_epochs       = 1       # passes over each fact's QA set (raise to make facts stick harder)
probe_recall_target= 0.6     # a dream commits only if cold recall ≥ this
sentinel_entropy_ceiling = 6.0
sentinel_entropy_floor   = 0.3   # below the model's natural ~0.8 thinking-off entropy
sentinel_sycophancy_ceiling = 0.5
```

**Recommended live tuning for a coherent chatbot that also learns** (currently in `local/config.toml`):
individuation `enabled=true`, `surprise_percentile=0.2` (permissive — let the classifier select),
`surprise_warmup=2`, `selfedit_paraphrases=6`, `dream_epochs=2`, `probe_recall_target=0.5`, with
`lr_absorb=5e-6` (gentle wake) and `lr_consolidate=2e-5` (strong dream).

---

## 8. Operating it

```
./run serve          # start the service (loads the 9B; first boot captures the canary baseline)
./run status         # brain snapshot (updates, overlay norm, individuation counts, drift)
./run token          # print the API bearer token (from the keychain)
./run proof          # prove the reward loop end-to-end on the live model
./run check          # full quality gate: ruff + ~154 real-model tests
```

Chat UI: **http://127.0.0.1:8500/** — a plain chatbot; the header pill shows `N noticed · M learned`,
click it for the memory drawer (what it has learned, a **Consolidate** button = run a dream, a
**Prove recall** button = live cold-recall check).

**How to run a dream** (consolidate what it has noticed into learned facts):
```
curl -sX POST http://127.0.0.1:8500/v1/brain/dream  -H "Authorization: Bearer $(./run token)"
# → {"committed": true, "facts_learned": 2, "dropped": 1, "recall": 1.0, "entropy": ..., "sycophancy": 0.0}
```
`committed:true` with `facts_learned:N` means it genuinely learned N facts and verified it can recall
them cold. `committed:false` means the health gate refused (recall below target, or a sentinel tripped)
and the overlay was reverted — nothing was learned that night.

Other control-plane endpoints (all mutating POSTs need the Bearer token, or the same-origin cookie the
chat page carries):
```
GET  /v1/brain                 # full status
GET  /v1/brain/memory          # the learned facts (read-only)
POST /v1/brain/verify          # live cold-recall of every learned fact
POST /v1/brain/dream           # individuation consolidation (overlay-level)
POST /v1/brain/consolidate     # fold the overlay INTO the base weights + requantize (heavy)
POST /v1/brain/checkpoint      # snapshot the overlay
POST /v1/brain/rollback        # restore the last known-good overlay
POST /v1/feedback {trace_id,reward}   # reward-loop feedback
POST /v1/chat/completions      # OpenAI-compatible; set "enable_thinking": false for snappy replies
```

Runs under `auto` as service `engram`. State lives in gitignored `local/data/` (journal, traces,
checkpoints, experience log, individuation probes, canary baseline).

---

## 9. Code map

```
src/common/       config.py (all tunables) · store.py (atomic persistence, data-root) · identity.py (keychain token)
src/engine/       model_host.py (the MLX model, gpu_lock, generate→Trace, span_logprobs, enable_thinking)
                  generation.py (chat template, streamed decode + per-token logprobs, span parser)
                  tool_parser.py (qwen3_xml parse + tool-result scoring) · trace.py (the turn record)
src/plasticity/   adapter.py (the LoRA overlay) · losses.py (the math) · updater.py (THE guarded step)
                  guards.py (mask/clip/cap/pause) · journal.py (append-only event log) · replay.py
                  checkpoints.py (overlay ring + rollback) · consolidate.py (fold overlay → bf16 base)
src/individuation/ surprise.py (peak surprise + adaptive gate) · experience.py (immutable provenance log)
                  corroborate.py (durability classifier) · selfedit.py (fact → QA synthesis)
                  probe.py (cold-recall probe + entropy/sycophancy sentinels) · dream.py (the sleep loop)
src/server/       app.py (wiring, lifespan, shutdown persistence) · work_queue.py (the update worker)
                  openai_api.py (chat + tool scoring + the individuation seam) · feedback_api.py
                  brain_api.py (status, memory, verify, dream, consolidate, rollback, probe) · ui.py + chat.html
src/evaluation/   canary.py + canary_prompts.py (behaviour-drift probe) · proof.py (reward-loop E2E demo)
```

Every `x.py` has a co-located `x_test.py` with **real** tests (the 0.8B `qwen3.5-0.8b-4bit` model,
no mocks). `./run check` runs ruff + the whole suite; pytest runs serially (one GPU).

---

## 10. Empirical findings (hard-won, on the live 9B — do not re-learn these)

- **Peak surprise, not mean.** Mean per-token CE ranks terse chatter above fluent facts; a novel fact
  only spikes on its content token. (§5.1)
- **The reasoning-override barrier — the frontier.** A gentle absorb shifts the next-token prior, so a
  *thinking-off* answer recalls the fact, but the 9B's RL-trained "I'm a stateless assistant, I don't
  know personal facts" reasoning talks itself out of the recall in normal (thinking-on) serving.
  Crossing it reliably took stronger consolidation (`lr_consolidate` 2e-5, 6–8 paraphrases,
  `dream_epochs` 2). Robustly overriding a strong reasoning prior for *arbitrary* knowledge is the
  **targeted-editing stage (v2)** — see §11.
- **Recall strength vs. collapse is a real tension.** The strength that makes facts stick drives the
  neutral-prompt entropy down toward an overconfident peak. The entropy sentinel guards **both** sides
  (floor 0.3 catches collapse); the dream health-gate refuses a night that peaks the model too hard.
- **The surprise gate is a pre-filter, not the selector.** Per-token surprise can't rank "informative"
  perfectly; the dream's durability classifier is what actually keeps facts and drops role-play.
- **Wake and sleep need different strengths.** One shared LR forced a bad trade (coherent chat vs.
  facts that stick); the `absorb`/`consolidate` kind split resolves it.
- Operational traps in `AGENTS.md`: never double-take `gpu_lock` (`span_logprobs` takes it itself →
  deadlock); tests must not read live `local/data` OR `local/config.toml`; the individuation-probe and
  canary answer-checks generate with thinking OFF; punishment is near the noise floor at few rounds.

---

## 11. Open problems & where to take it next

1. **Targeted model editing (v2, the user's stated direction).** Locate-and-edit the specific weights
   that encode a corroborated fact (ROME/MEMIT-style), so it overrides the reasoning prior cleanly
   *without* the collapse pressure that brute-force absorption creates. Reuse this repo's
   "corroborate → verify cold → accept or revert" discipline as the safety envelope. The wake/sleep +
   provenance spine is the on-ramp.
2. **Reasoning-aware self-edit.** Instead of Q→A pairs, synthesize training examples where the
   assistant *reasons* using the fact and then answers — teach it to bring the knowledge into its
   thinking, not just its next-token prior.
3. **Recurrence-based corroboration (v1.1).** Require a fact to recur across sessions before it
   consolidates, to further reduce absorbing one-off falsehoods (currently content-classification only).
4. **Right-to-forget.** A secret absorbed into weights can't be cleanly unlearned by gradient means;
   the provenance log makes *targeted re-projection* the escape hatch, but true unlearning is unsolved.
5. **Auto-dream scheduling — DONE.** A continuous background `DreamLoop`
   (`individuation/dream_loop.py`, `individuation.auto_dream`) now runs the dream whenever there is
   unconsolidated experience, and a low-speed **re-polish** (`dream.repolish`) re-trains learned facts
   that have gone stale (`repolish_after_h`, default 48h). It is a fixed loop: work → loop immediately,
   idle → sleep `dream_idle_sleep_s` and re-check. Off by default. Every cycle still goes through the
   same snapshot → probe/sentinels → commit/revert gate the manual dream uses.
6. **A faster differentiable DeltaNet scan.** The training path is an O(T) Python-unrolled loop, the
   main reason spans are capped at ≤512 tokens. A chunked parallel-scan VJP would lift that cap.

---

## 12. The definition of done / how to verify a change

- `./run check` green (ruff + real-model tests, warnings-as-errors).
- `./run proof` passes on the reward loop.
- For individuation: state a fact once in chat → `POST /v1/brain/dream` commits with `facts_learned>0`
  → `POST /v1/brain/verify` recalls it → ask it cold in a fresh session and it answers.
- Guards intact: the canary stays within budget, sentinels healthy, and any bad stretch is revertible
  via `/v1/brain/rollback` or by re-projecting from the experience log.
