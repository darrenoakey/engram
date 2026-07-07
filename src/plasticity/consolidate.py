# =============================================================================
#  consolidate — fold the overlay into the bf16 master, then requantize
#  why: deltas that live only in the low-rank overlay eventually saturate; a
#  4-bit serving base would snap sub-grid deltas to zero, so consolidation
#  merges them into the full-precision master first and requantizes from there.
#  Streams shard-by-shard in fp32 so a large master never loads whole.
# =============================================================================
from __future__ import annotations

import shutil
from pathlib import Path

import mlx.core as mx
from mlx_lm.convert import convert


# ##################################################################
# selected deltas
# the alpha-scaled weight deltas to merge, restricted to targeted paths when
# the caller names them (otherwise every adapter)
def selected_deltas(overlay, targeted_paths) -> dict:
    deltas = overlay.merge_deltas()
    if not targeted_paths:
        return deltas
    keep = set(targeted_paths)
    return {key: value for key, value in deltas.items() if key in keep}


# ##################################################################
# merge shards
# add each delta into its master shard in fp32, cast back to the shard's dtype,
# and write the shard to out_dir; returns the weight keys actually merged
def merge_shards(master: Path, out: Path, deltas: dict) -> list:
    merged: list = []
    for shard in sorted(master.glob("*.safetensors")):
        tensors = mx.load(str(shard))
        for key, delta in deltas.items():
            if key in tensors:
                original_dtype = tensors[key].dtype
                tensors[key] = (tensors[key].astype(mx.float32) + delta).astype(original_dtype)
                merged.append(key)
        mx.save_safetensors(str(out / shard.name), tensors)
    return merged


# ##################################################################
# copy aux
# copy every non-weight file (config.json, tokenizer files, index) verbatim so
# out_dir is a complete, loadable model directory
def copy_aux(master: Path, out: Path) -> None:
    for item in master.iterdir():
        if item.is_file() and item.suffix != ".safetensors":
            shutil.copy2(item, out / item.name)


# ##################################################################
# dream
# consolidation: merge the overlay deltas into the bf16 master shard-by-shard,
# producing a new full-precision master generation at out_dir
def dream(model_dir_master: str, overlay, out_dir: str, targeted_paths) -> dict:
    master = Path(model_dir_master)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    deltas = selected_deltas(overlay, targeted_paths)
    merged = merge_shards(master, out, deltas)
    missing = sorted(set(deltas) - set(merged))
    if missing:
        raise ValueError(f"{len(missing)} deltas matched no master shard key (naming drift?): {missing[:3]}")
    copy_aux(master, out)
    return {"out_dir": str(out), "merged": merged, "shards": len(list(master.glob("*.safetensors")))}


# ##################################################################
# quantize targets
# requantize a merged bf16 master into a 4-bit affine gs64 serving base using
# mlx-lm's own convert path (the same quantizer that produced the base)
def quantize_targets(out_dir: str, quant_dir: str, bits: int = 4, group_size: int = 64) -> str:
    quant = Path(quant_dir)
    if quant.exists():
        shutil.rmtree(quant)
    convert(out_dir, str(quant), quantize=True, q_group_size=group_size, q_bits=bits, q_mode="affine")
    return str(quant)
