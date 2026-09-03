"""Immutable upstream and model lockfile support."""

from __future__ import annotations

import copy
import hashlib
import os
import re
import sysconfig
import tomllib
from pathlib import Path
from typing import Any

from .profiles import sha256_json


class LockError(ValueError):
    """Raised when a lockfile is missing or malformed."""


_LOCK_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def discover_locks_dir(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_root = os.environ.get("H3_FLASH_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "locks")
    candidates.extend(
        [
            Path.cwd() / "locks",
            Path(__file__).resolve().parents[2] / "locks",
            Path(sysconfig.get_path("data")) / "h3_flash" / "locks",
        ]
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.toml")):
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise LockError(f"lock directory not found; checked: {rendered}")


class LockRepository:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = discover_locks_dir(directory)
        self._cache: dict[str, dict[str, Any]] = {}

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(path.stem for path in self.directory.glob("*.toml")))

    def load(self, name: str) -> dict[str, Any]:
        if not _LOCK_NAME.fullmatch(name):
            raise LockError(f"invalid lock name: {name!r}")
        if name not in self._cache:
            path = self.directory / f"{name}.toml"
            if not path.is_file():
                raise LockError(f"unknown lock: {name}")
            with path.open("rb") as handle:
                data = tomllib.load(handle)
            self._validate(name, data)
            self._cache[name] = data
        return copy.deepcopy(self._cache[name])

    def digest(self, name: str) -> str:
        return sha256_json(self.load(name))

    def reference(self, name: str) -> dict[str, Any]:
        data = self.load(name)
        reference = {
            "name": name,
            "sha256": self.digest(name),
            "schema_version": data["schema_version"],
        }
        for field in (
            "status",
            "profile_family",
            "base_model_lock",
            "repo",
            "revision",
            "total_weight_bytes",
        ):
            if field in data:
                reference[field] = data[field]
        return reference

    @staticmethod
    def _validate(name: str, data: dict[str, Any]) -> None:
        if data.get("schema_version") != 1:
            raise LockError(f"{name}: schema_version must be 1")
        files = data.get("files")
        downloads = data.get("downloads", [])
        if not isinstance(downloads, list):
            raise LockError(f"{name}: downloads must be an array")
        for index, entry in enumerate(downloads):
            required = ("repo", "revision", "local_prefix", "include")
            if not isinstance(entry, dict) or any(
                field not in entry for field in required
            ):
                raise LockError(f"{name}: downloads[{index}] lacks a required field")
            prefix = Path(entry["local_prefix"])
            if prefix.is_absolute() or ".." in prefix.parts:
                raise LockError(f"{name}: unsafe download local_prefix")
            if not isinstance(entry["include"], list) or not entry["include"]:
                raise LockError(f"{name}: downloads[{index}].include must be non-empty")
        if files is None:
            return
        if not isinstance(files, list) or not files:
            raise LockError(f"{name}: files must be a non-empty array")
        total = 0
        for index, entry in enumerate(files):
            if not isinstance(entry, dict):
                raise LockError(f"{name}: files[{index}] must be a table")
            required = ("path", "local_path", "bytes", "sha256")
            if any(field not in entry for field in required):
                raise LockError(f"{name}: files[{index}] lacks a required field")
            local = Path(entry["local_path"])
            if local.is_absolute() or ".." in local.parts:
                raise LockError(f"{name}: unsafe local_path {entry['local_path']!r}")
            if not isinstance(entry["bytes"], int) or entry["bytes"] < 1:
                raise LockError(f"{name}: invalid byte count for {entry['path']}")
            digest = entry["sha256"]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise LockError(f"{name}: invalid SHA-256 for {entry['path']}")
            total += entry["bytes"]
        declared = data.get("total_weight_bytes")
        if declared is not None and total != declared:
            raise LockError(f"{name}: total_weight_bytes={declared}, computed={total}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(32 << 20):
            digest.update(block)
    return digest.hexdigest()
