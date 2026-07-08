# engram — ambient continual self-individuation (design)

> Status: **design, not yet built.** This document is the plan a council of five reviewers
> converged on. `DESIGN.md` is the canonical description of what already exists and runs; this
> extends it. Nothing here ships until it is reviewed and the acceptance criteria (§12) are met.

## 1. The goal

Today engram *reinforces its own behaviour*: on a reward signal, it strengthens or punishes the
tokens **it** produced. That is the behaviour half of "interaction changes the model." This
document specifies the **knowledge half**:

> From ordinary, unlabelled use — **nothing explicit, ever** — engram should progressively become
> an individual tailored to its user: it learns their facts, behaviours, and preferences **into
> its weights**, and *properly learns* them (integrates them into its world-model, recallable cold
> in a fresh context) rather than *remembering* them (looking them up). This is not RAG — nothing
> is retrieved and injected at answer time; the weights themselves change.

The user's framing, verbatim: *"we use the model — and it adjusts to us. never anything explicit …
as we use it it becomes an individual tailored to us — it learns our behaviours our facts, our
preferences etc — properly learns them rather than remembering them."*

This is the stability–plasticity dilemma at its hardest: continual, lifelong, label-free
absorption of one person into a model that must stay generally competent and never collapse.

## 2. The load-bearing decision: train on the USER's tokens, never the model's own

Three of five council seats reached this independently, and it dissolves the two scariest problems
at once:

- **It solves the no-labels problem.** The user's words *are* ground truth. Their facts,
  preferences, and idiom live in their own text. Teacher-forcing the user's tokens is
  self-supervised — no reward, no human label, no synthesizer required for the signal itself.
  This is classic *dynamic evaluation* (Krause 2018/19; DeepMind 2024), pointed at the human half
  of the conversation.
- **It makes model collapse structurally impossible.** Model collapse is degeneration from
  training on your own outputs. If engram **never** trains on its own generations for knowledge,
  it cannot eat its own tail. (The existing reward loop still touches assistant tokens — but that
  is bounded, reward-gated *behaviour* shaping, not open-loop knowledge absorption.)

**Rule:** the knowledge-absorption path trains only on `role:"user"` message tokens (and, at
sleep, on self-edited *assistant-knowledge* reformulations of them — §5). It never trains on the
assistant's raw generated tokens.

## 3. Architecture: two tiers, two timescales

A memory hierarchy, mirroring episodic→semantic consolidation in brains:

```
                        WAKE (during use)                     SLEEP (nightly / on idle)
  user turn ──▶ surprise gate ──▶ experience log ─────────────▶ dream job
                     │                (immutable, source of record)   │
                     └─▶ volatile OVERLAY micro-update            corroborate → self-edit →
                         (fast, felt, recoverable)                consolidate into BASE
                                                                  (guarded, atomic, revertible)
```

| Tier | Store | Timescale | Writes | Risk posture |
|---|---|---|---|---|
| **Episodic** | plastic LoRA overlay (exists) | per surprising turn, during use | user-token micro-update | volatile, recoverable — cheap to be wrong |
| **Semantic** | bf16 master → 4-bit serving base (exists) | nightly / idle, batched | corroborated, self-edited, gated | permanent — only written under full guards |

**Why the split (not raw per-turn base edits):** the guards that keep online learning safe
(KL-anchor, replay, canary) are *distributional* — on a batch of one turn they are noise. Batches
only exist at sleep, so that is the only place a durable write is defensible. And one write per
night gives a bisectable, revertible history instead of thousands of un-versioned micro-edits.

**Why still write the overlay during the day:** the user explicitly wants to *feel* the model
adapt within use ("as we use it it adjusts to us"). The overlay is volatile and recoverable, so
these writes carry the felt immediacy without risking the base. This is the reconciliation the
council landed on (Architect's split, adopting the Contrarian's surprise gate).

## 4. WAKE — the per-turn loop (during use)

For each completed turn, on the user's most recent message `U = (u_1 … u_m)` in context `C`:

**4.1 Surprise gate.** Compute the model's surprise as the mean cross-entropy it assigns to the
user's actual message:

```
surprise(U | C) = − (1/m) Σ_t log p_θ(u_t | C, u_<t)
```

This forward pass is nearly free (it overlaps the normal prefill of the next turn). Gate:

- `surprise ≤ τ` → the model already predicts this user → **do nothing** (skips the ~95% of
  turns that carry no individuating signal, per the Contrarian).
- `surprise > τ` → this turn carries individuating residual → proceed.

`τ` is **adaptive**, a rolling high percentile (e.g. 70th) of recent per-turn surprise, so it
self-calibrates per user and never depends on an absolute threshold. Surprise is the implicit
label the user will never give explicitly.

**4.2 Log it.** Append the turn to the **experience log** (immutable, append-only, the source of
record): `{context digest, user tokens, surprise, timestamp, serving-generation id, learner
version, seed}`. This is the provenance spine (§9). Extends the existing journal/trace stores.

**4.3 Absorb into the overlay (optional, config-gated).** An async `absorb` micro-update
(~12 s, off the response critical path) teacher-forces the **user's tokens** through the existing
guarded pipeline — KL-anchor to the frozen base, top-k mask, norm clip, delta cap, per-update KL
gate, replay mix — into the **overlay only**. This is a new update `kind` alongside `reinforce`
and `reward`; the updater already accepts `(token_ids, credit_spans, kind)`, so the credit span
is simply aimed at the user's message instead of the assistant's.

The overlay write shifts the model's *world-model* toward the user (felt adaptation, better
in-session priors). It is **not** relied on for cold factual recall — that is the sleep job's
work (§5), because raw next-token absorption tends to make the model *mimic* the user's voice more
than *know* the fact (the "predict vs know" gap, §11).

## 5. SLEEP — the nightly dream (the only durable writer)

Runs on idle or a nightly schedule. The base is never written anywhere else.

1. **Select.** Pull the day's high-surprise experiences from the log.
2. **Corroborate (the truth gate).** Keep a candidate only if it *recurs* or stays *consistent*
   across sessions; drop one-off role-play, sarcasm, hypotheticals, and test-probes. A statement
   restated or relied upon on separate days earns consolidation; "pretend you're a pirate" said
   once does not. Contradiction filter: a new claim inconsistent with a more-corroborated one is
   held, not written. (The Skeptic's defence against mode/truth confusion.)
3. **Self-edit (learn, not memorise).** For each survivor, the model reformulates the user's
   statement into **assistant-knowledge** form with paraphrase augmentation — e.g. the user saying
   "ugh, shellfish again" across contexts becomes `{Q: "any dietary constraints?" → A: "You're
   allergic to shellfish."}` plus paraphrases. Training targets the **assistant-knowledge answer
   tokens**, which is what turns prediction-of-the-user into knowledge-about-the-user and yields
   cold recall. This is the SEAL-style self-edit the user already chose; augmentation is what
   makes it generalise rather than memorise a single phrasing.
4. **Consolidate.** SFT the overlay on the distilled, augmented, corroborated set — a real batch,
   where KL-anchor + replay-of-anchors + norm caps finally have statistics to act on — until the
   day's held-out individuation probes (§8) pass **and** the degeneration sentinels stay green.
   Then fold the overlay into the bf16 master, requantize to a new serving base, reset the
   overlay. Exactly the consolidation path that already exists and is canary-gated with revert.
5. **Commit atomically.** The night is one revertible unit. Keep the pre-consolidation master and
   the experience log; consolidation never destroys provenance (§9).

## 6. The `absorb` update, precisely

- **Target tokens:** wake — the user's message tokens; sleep — the assistant-knowledge answer
  tokens of the self-edited pairs. Never the assistant's raw generation.
- **Loss:** teacher-forced cross-entropy on the target span, `+ β·KL(π_overlay ‖ π_base)` on the
  same span (existing `losses.kl_anchor`), identical shape to the current positive path with
  reward fixed to `+1` (there is no reward here — absorption is unconditional given the gate).
- **Guards:** unchanged and reused verbatim — top-k grad mask, global norm clip, per-tensor delta
  cap, per-update KL gate (reject + restore snapshot), replay mix, adapter-norm-ceiling pause.
- **Learning rate:** conservative (start at the `lr_reward` order, ≤5e-6); the felt adaptation is
  cumulative across a session, not a single big step.

## 7. Handling the modes that look identical to facts

With no labels, a fact, a joke, a hypothetical, a quote, and a role-play instruction are the same
token stream. Defences, in layers (none individually sufficient — see §11):

- **User-tokens-only** already neutralises the worst case: "pretend you're a pirate" trains on the
  *user's instruction*, not the model's pirate voice, so the persona is never absorbed.
- **Surprise-gating** skips coherent, already-predictable hypotheticals (low surprise).
- **Corroboration/contradiction** at sleep drops the one-off and the inconsistent.
- **Volatile-overlay-by-day** means a single bad turn lives only in the recoverable tier until it
  proves itself worthy of the base overnight.

## 8. Measurement — fixing the canary's blind spot

Every seat that owns measurement flagged the same thing: **the existing canary measures drift from
the ORIGINAL base, but mis-individuation drifts *along the user's manifold* — exactly where a
base-anchored canary is blind.** "It passes while the model rots." So the base canary stays (it
guards general competence) but is *necessary, not sufficient*. Add:

- **Individuation probe (the success signal, label-free).** Each consolidated fact becomes a
  held-out **cold-context** probe: fresh session, empty context, retrieval off — does the model
  recall it? Recall rate over the *growing* set of the user's own corroborated facts is an
  objective, unlabelled success curve — the user supplied the labels by talking. A rising recall
  curve is the definition of "it's learning me."
- **Degeneration sentinels (trend-gated health).** Mean next-token entropy on a fixed neutral
  prompt set (entropy-collapse detector), a sycophancy probe (agreement rate on planted-false
  statements — catches collapse *toward* the user), and a self-contradiction rate. Gate on
  **trends** (drift slope, entropy creep), not per-update noise.
- **Consolidate to base only after N sustained-green nights**, so slow rot is caught before it
  hardens into requantized weights.

## 9. Provenance and rollback

The Maintainer's spine, in its achievable form (not bit-exact replay, which he himself called "a
lie I may not be able to keep" across MLX kernels and requantization):

- The **experience log is the source of record.** Every ambient edit traces to the logged
  interaction(s) that caused it, with seed and learner version.
- **The night is the rollback unit.** "Undo last Tuesday" = restore the last green pre-consolidation
  master and re-project forward from the log, filtering that window — not a hunt for a bad commit
  among thousands of micro-edits.
- **Consolidation never destroys provenance:** retain the pre-consolidation bf16 master and the log
  so requantization is never a one-way door.
- **A checksum sentinel** periodically compares the live serving weights against what the log +
  checkpoints imply; divergence is a loud alarm, not a silent lie.

## 10. What is reused vs genuinely new

**Reused, already built and proven live (~80%):** the plastic overlay + attachment; the entire
guarded update pipeline (`plasticity/updater`, `losses`, `guards`); replay buffer; checkpoints +
ring + shutdown persistence; consolidation into bf16 master + requantize + canary-gate + revert;
the journal; the `/v1/brain` surface; the base canary.

**New:**
1. **Surprise gate** — a per-turn CE-on-user-tokens computation + adaptive threshold.
2. **`absorb` update kind** — user-token / self-edit-answer target through the existing pipeline.
3. **Experience log** — immutable, provenance-tagged (extends the trace store).
4. **Dream job** — corroborate → self-edit → batch-consolidate → atomic nightly commit.
5. **Corroboration/contradiction gate** — recurrence + consistency over the buffer/history.
6. **Individuation probe + degeneration sentinels** — the label-free health/success signals.

## 11. Honest open problems (the council would not let these pass)

- **Truth/mode confusion is reduced, not solved.** A *consistently repeated* falsehood or running
  joke will still consolidate. Corroboration raises the bar; it is not a lie detector.
- **The generalisation gate is self-referential** — the model judges whether its own self-edits
  generalised. The independent individuation probe is what keeps this from rotting invisibly, which
  makes that probe set load-bearing and worth building carefully.
- **Predict-vs-know / persona bleed** — if the self-edit reformulation is weak, user-token
  absorption bleeds the user's voice into the assistant instead of producing clean knowledge. The
  self-edit quality is the pivot; it needs its own eval.
- **Right-to-forget** — a secret typed once and absorbed into weights cannot be cleanly unlearned
  by gradient means; the provenance log makes *targeted re-projection* the escape hatch, but true
  weight-level unlearning is unsolved (and is part of why the base write is gated hard).
- **"No explicit signal ever" is a real constraint, not a preference** — surprise + corroboration
  are *implicit* signals doing the work labels normally do; they will occasionally admit garbage
  and occasionally drop rare truths. This is inherent to the goal, not a bug to fully engineer out.

## 12. Acceptance criteria (definition of done for v1)

Per the user's choice, v1 is **volatile overlay by day + guarded nightly base**. It is done when,
on the live 9B, all of the following hold with real numbers:

1. **Cold recall from ordinary chat.** State a fact once in a normal conversation (no command, no
   `/teach`). In a later, fresh session — empty context, retrieval off — the model uses it
   (the shellfish demo). Over a held-out set of once-stated facts, a **rising cold-recall curve**.
2. **Surprise gate works** — predictable turns are skipped; only high-surprise turns learn
   (measurable in the journal).
3. **No collapse / no rot** — base canary within budget AND degeneration sentinels flat across a
   multi-day run; a planted one-off ("pretend you're a pirate") does **not** persist.
4. **Nightly atomicity + rollback** — a night can be reverted, restoring the prior self exactly.
5. **Full gate green** — real-model tests for the surprise gate, `absorb` update, corroboration,
   self-edit, dream job, and probes; ruff clean; warnings-as-errors.

## 13. Phasing

- **v1 (this doc):** surprise gate + `absorb` into the overlay (felt, by day) + experience log +
  nightly dream (corroborate → self-edit → consolidate) + individuation probe + sentinels +
  nightly rollback. Overlay persists across restart (already works).
- **v1.1:** richer corroboration (embedding-clustered recurrence), sycophancy/entropy sentinels
  tuned on real multi-week data.
- **v2:** targeted model editing (the user's stated later direction) — locate-and-edit the specific
  weights for a corroborated fact, using this doc's "corroborate → verify cold → accept or revert"
  discipline as the safety envelope. The wake/sleep + provenance spine is exactly the on-ramp.
