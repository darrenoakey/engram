# engram — agent operating manual

Self-modifying MLX inference engine for Ornith-1.0-9B. **DESIGN.md is canonical** — read it
before touching anything; every module contract, tunable, and safety rail is specified there
with its research pedigree.

## Commands
- `./run check` — full quality gate (ruff + entire pytest suite). Must be green before commit.
- `./run serve` — start the service (OpenAI-compatible, :8500).
- Dev loop: `.venv/bin/python -m pytest src/<module> -q` (only the files you're changing).

## Non-obvious facts that bite
- Tests load the REAL 0.8B model at `/Volumes/Gumby/models/qwen3.5-0.8b-4bit` (same qwen3_5
  architecture as the 9B). Never introduce test doubles — forbidden words include mock/fake/
  stub/dummy/sleep/todo (use `threading.Event().wait()` for delays).
- `model.train()` vs `model.eval()` is load-bearing: train mode routes Gated-DeltaNet through
  the differentiable scan; eval uses a Metal kernel with NO gradient support. Generation must
  be eval; update steps must be train; always restore eval in a finally block.
- 24 of 32 layers are DeltaNet (`in_proj_qkv/z/b/a`, `out_proj` — no q/k/v names); full-attention
  `q_proj` is double-width (gate-packed). Never select adapter targets by name pattern alone.
- Teacher-forced update spans are hard-capped (256 default / 512 max) — T2048 OOMs a 64GB M1 Max.
- Requantization to 4-bit erases sub-grid weight deltas: consolidation ALWAYS merges into the
  bf16 master first, never directly into the quantized serving base.
- One Metal GPU: pytest runs serially; generation/updates/probes all serialize on the host gpu_lock.
- No env vars ever (config: `local/config.toml`; secrets: OS keychain via `common/identity.py`).
- Learning persists across a GRACEFUL restart via a shutdown checkpoint (app lifespan); a hard
  crash loses at most `checkpoint_every` updates. Restart restores the last canary-clean overlay.
- Ornith is a reasoning model: short generations stay inside `<think>` with no answer span. The
  worker credits the reasoning span as a fallback (never drops feedback) and the canary answer
  check generates with thinking OFF so a direct reply fits its token budget. `enable_thinking`
  threads through `generate`; only turn it off for direct-answer probes, not normal serving.
- Tests must never write to live `local/data` — the session-autouse conftest redirects the whole
  store root to `output/testing`. A leaked canary subset would skip the real boot baseline.
