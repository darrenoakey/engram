# =============================================================================
#  adapter — the plastic overlay (LoRA-style deltas over a frozen base)
#  why: engram changes its own weights online; a low-rank overlay is the only
#  thing that trains, so the base stays frozen and updates stay bounded and
#  cheap. Deltas add in only when the overlay is enabled (base-logit forwards
#  toggle it off). Never imports src/engine — callers pass the raw mlx model.
# =============================================================================
from __future__ import annotations

import contextlib

import mlx.core as mx
import mlx.nn as nn


# ##################################################################
# switch
# a shared mutable on/off flag so a single toggle turns every attached
# adapter's delta on or off at once (base-logit forwards need it off)
class Switch:
    def __init__(self) -> None:
        self.on = True


# ##################################################################
# derive dims
# read (in_dims, out_dims) from a wrapped linear the way mlx-lm LoRA does:
# a quantized weight packs its input dim by 32//bits, so unpack it
def derive_dims(base: nn.Module) -> tuple[int, int]:
    out_dims, in_dims = base.weight.shape
    if isinstance(base, nn.QuantizedLinear):
        in_dims = in_dims * 32 // base.bits
    return in_dims, out_dims


# ##################################################################
# plastic linear
# wraps any Linear/QuantizedLinear: y = base(x) + (alpha/rank)*(x @ A) @ B.
# A ~ N(0,0.02), B zero-init (cold start adds nothing); bf16 params. Gradients
# reach A/B through the input x, never through the frozen quantized weight.
class PlasticLinear(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: int, switch: Switch) -> None:
        super().__init__()
        self.base = base
        in_dims, out_dims = derive_dims(base)
        self.scale = alpha / rank
        self.a = (mx.random.normal((in_dims, rank)) * 0.02).astype(mx.bfloat16)
        self.b = mx.zeros((rank, out_dims)).astype(mx.bfloat16)
        self.switch = switch

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        if not self.switch.on:
            return y
        z = (x @ self.a) @ self.b
        return y + (self.scale * z).astype(x.dtype)

    # ##################################################################
    # delta weight
    # the effective weight change, oriented (out, in) to match the base weight
    # layout: W_delta = scale * (A @ B).T computed in fp32 for consolidation
    def delta_weight(self) -> mx.array:
        a32 = self.a.astype(mx.float32)
        b32 = self.b.astype(mx.float32)
        return self.scale * (b32.T @ a32.T)


# ##################################################################
# band
# scale DESIGN's mid-layer band [8,28] (defined against 32 layers) onto the
# actual layer count so the 0.8B's 24 layers get a proportional band
def band(mid_layers: tuple, n_layers: int) -> tuple[int, int]:
    lo = round(mid_layers[0] / 32 * n_layers)
    hi = round(mid_layers[1] / 32 * n_layers)
    return lo, hi


# ##################################################################
# target names
# which submodules of one decoder layer receive an adapter: MLP always,
# attention q/k/v/o only on full-attention layers (DeltaNet in_proj_* frozen)
def target_names(layer) -> list[str]:
    names = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    if not getattr(layer, "is_linear", True):
        names += ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    return names


# ##################################################################
# resolve
# walk a dotted submodule path (e.g. "mlp.gate_proj") to the parent module
# and the final attribute name so we can swap the linear in place
def resolve(layer, dotted: str):
    parent = layer
    parts = dotted.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


# ##################################################################
# overlay
# the set of attached adapters plus the guarded lifecycle around them:
# enable/snapshot/restore/save/load/norms/reset/merge. Holds only mlx arrays
# and module references; no engine coupling.
class Overlay:
    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config
        self.switch = Switch()
        self.adapters: list[tuple[str, PlasticLinear]] = []

    # ##################################################################
    # enabled
    # module-level flag controlling whether every adapter adds its delta
    @property
    def enabled(self) -> bool:
        return self.switch.on

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.switch.on = bool(value)

    # ##################################################################
    # disabled
    # context manager for base-logit forwards: deltas off inside, restored after
    @contextlib.contextmanager
    def disabled(self):
        previous = self.switch.on
        self.switch.on = False
        try:
            yield
        finally:
            self.switch.on = previous

    # ##################################################################
    # trainable parameters
    # flat path->array of exactly the adapter A/B tensors (base is frozen)
    def trainable_parameters(self) -> dict:
        params: dict = {}
        for weight_path, module in self.adapters:
            base = weight_path[: -len(".weight")]
            params[base + ".a"] = module.a
            params[base + ".b"] = module.b
        return params

    # ##################################################################
    # snapshot
    # materialise a bit-exact copy of every adapter tensor for rollback
    def snapshot(self) -> dict:
        params = self.trainable_parameters()
        mx.eval(list(params.values()))
        return {key: mx.array(value) for key, value in params.items()}

    # ##################################################################
    # restore
    # reassign adapter tensors from a snapshot (arrays are immutable so this
    # is exact); optimizer state is intentionally not restored
    def restore(self, snap: dict) -> None:
        for weight_path, module in self.adapters:
            base = weight_path[: -len(".weight")]
            module.a = snap[base + ".a"]
            module.b = snap[base + ".b"]

    # ##################################################################
    # save / load
    # persist and reload adapter tensors as safetensors (path keys match
    # trainable_parameters); enables restart-with-learned-state
    def save(self, path: str) -> None:
        mx.save_safetensors(path, self.trainable_parameters())

    def load(self, path: str) -> None:
        loaded = mx.load(path)
        self.restore(loaded)

    # ##################################################################
    # tensor norms
    # per-wrapped-weight frobenius norm of the effective delta — the real
    # measure of how far a weight has been moved
    def tensor_norms(self) -> dict:
        norms: dict = {}
        for weight_path, module in self.adapters:
            norms[weight_path] = float(mx.linalg.norm(module.delta_weight()))
        return norms

    # ##################################################################
    # total norm
    # single scalar overlay magnitude (L2 over per-weight delta norms) used by
    # the adapter-norm-ceiling pause guard
    def total_norm(self) -> float:
        total = 0.0
        for value in self.tensor_norms().values():
            total += value * value
        return total ** 0.5

    # ##################################################################
    # reset
    # return every adapter to cold start (A ~ N(0,0.02), B zero) after a
    # consolidation folds the current deltas into the base weights
    def reset(self) -> None:
        for _weight_path, module in self.adapters:
            in_dims, out_dims = module.a.shape[0], module.b.shape[1]
            module.a = (mx.random.normal((in_dims, module.a.shape[1])) * 0.02).astype(mx.bfloat16)
            module.b = mx.zeros((module.a.shape[1], out_dims)).astype(mx.bfloat16)

    # ##################################################################
    # merge deltas
    # alpha-scaled weight deltas keyed by base weight path, oriented to the
    # base layout — the payload consolidation adds into the bf16 master
    def merge_deltas(self) -> dict:
        return {weight_path: module.delta_weight() for weight_path, module in self.adapters}


# ##################################################################
# attach overlay
# find the decoder layers by the known qwen3_5 path, wrap the targeted linears
# on the mid-layer band, freeze the base, and leave only adapters trainable
def attach_overlay(model, plasticity_config) -> Overlay:
    overlay = Overlay(model, plasticity_config)
    layers = model.layers
    lo, hi = band(plasticity_config.mid_layers, len(layers))
    for i in range(lo, hi + 1):
        _attach_layer(overlay, layers[i], i, plasticity_config)
    model.freeze()
    for _weight_path, module in overlay.adapters:
        module.unfreeze(keys=["a", "b"], recurse=False)
    return overlay


# ##################################################################
# attach layer
# wrap each target submodule of one layer in a PlasticLinear and record its
# base weight path (matches the safetensors key) for merge/save
def _attach_layer(overlay: Overlay, layer, index: int, config) -> None:
    for dotted in target_names(layer):
        parent, attr = resolve(layer, dotted)
        base = getattr(parent, attr)
        plastic = PlasticLinear(base, config.lora_rank, config.lora_alpha, overlay.switch)
        setattr(parent, attr, plastic)
        weight_path = f"language_model.model.layers.{index}.{dotted}.weight"
        overlay.adapters.append((weight_path, plastic))
