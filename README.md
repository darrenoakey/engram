# engram

A self-modifying local inference engine for Ornith-1.0-9B on Apple Silicon (MLX).
Weights change as a consequence of inference: decisions are reinforced when they
succeed and punished when they fail, driven by tool outcomes and user feedback.
Learned state persists across restarts and is periodically consolidated into the
base weights.

See DESIGN.md for the full architecture, research pedigree, and module contracts.

## Usage

```
./run serve          # start the service (OpenAI-compatible, port 8500)
./run status         # brain status
./run check          # full quality gate: ruff + pytest
```
