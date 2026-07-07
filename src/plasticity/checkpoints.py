# =============================================================================
#  checkpoints — versioned overlay snapshots with a canary-clean ring
#  why: every update snapshots first, but across many updates we also keep a
#  ring of durable checkpoints so a canary breach can roll back to the last
#  known-good overlay. Stored as safetensors + json meta through the store.
# =============================================================================
from __future__ import annotations

import time
import uuid
from pathlib import Path

import mlx.core as mx

from common import store


# ##################################################################
# checkpoints
# manages the on-disk ring of overlay checkpoints; the directory is injectable
# so tests write to output/testing rather than the live data dir
class Checkpoints:
    def __init__(self, directory: Path | None = None, ring: int = 20) -> None:
        self.directory = directory if directory is not None else store.checkpoints_dir()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ring = ring

    # ##################################################################
    # save
    # write the overlay's adapter tensors plus json meta under a fresh id,
    # then prune the oldest beyond the ring
    def save(self, overlay, updates_at_save: int, canary_clean: bool) -> str:
        checkpoint_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        mx.save_safetensors(str(self._weights(checkpoint_id)), overlay.trainable_parameters())
        meta = {
            "id": checkpoint_id,
            "created_at": time.time(),
            "updates_at_save": updates_at_save,
            "canary_clean": canary_clean,
        }
        store.atomic_write_json(self._meta(checkpoint_id), meta)
        self._prune()
        return checkpoint_id

    # ##################################################################
    # restore
    # load a checkpoint into the overlay; with no id, pick the most recent
    # checkpoint whose canary was clean (the last known-good overlay)
    def restore(self, overlay, checkpoint_id: str | None = None) -> str:
        if checkpoint_id is None:
            checkpoint_id = self._last_good()
        if checkpoint_id is None:
            raise RuntimeError("no canary-clean checkpoint to restore")
        overlay.load(str(self._weights(checkpoint_id)))
        return checkpoint_id

    # ##################################################################
    # list
    # all checkpoint metas, newest first
    def list(self) -> list[dict]:
        metas = [store.read_json(p) for p in self.directory.glob("*.json")]
        return sorted(metas, key=lambda m: m["created_at"], reverse=True)

    def _weights(self, checkpoint_id: str) -> Path:
        return self.directory / f"{checkpoint_id}.safetensors"

    def _meta(self, checkpoint_id: str) -> Path:
        return self.directory / f"{checkpoint_id}.json"

    # ##################################################################
    # last good
    # id of the newest canary-clean checkpoint, or None
    def _last_good(self) -> str | None:
        for meta in self.list():
            if meta.get("canary_clean"):
                return meta["id"]
        return None

    # ##################################################################
    # prune
    # delete weights+meta for every checkpoint beyond the newest `ring`
    def _prune(self) -> None:
        for meta in self.list()[self.ring:]:
            self._weights(meta["id"]).unlink(missing_ok=True)
            self._meta(meta["id"]).unlink(missing_ok=True)
