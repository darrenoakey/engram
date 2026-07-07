# =============================================================================
#  store — atomic persistence and data-directory layout
#  why: a daemon that reads its own mid-write files silently degrades to empty
#  state; every writer in engram goes through temp-file + os.replace here
# =============================================================================
from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path

from common.config import repo_root


def data_root() -> Path:
    root = repo_root() / "local" / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _subdir(name: str) -> Path:
    path = data_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def traces_dir() -> Path:
    return _subdir("traces")


def checkpoints_dir() -> Path:
    return _subdir("checkpoints")


def canary_dir() -> Path:
    return _subdir("canary")


def journal_path() -> Path:
    return data_root() / "journal.jsonl"


def replay_path() -> Path:
    return data_root() / "replay.json"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False).encode())


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json_gz(path: Path, obj) -> None:
    atomic_write_bytes(path, gzip.compress(json.dumps(obj, ensure_ascii=False).encode()))


def read_json_gz(path: Path):
    return json.loads(gzip.decompress(path.read_bytes()))


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path, limit: int | None = None) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows if limit is None else rows[-limit:]
