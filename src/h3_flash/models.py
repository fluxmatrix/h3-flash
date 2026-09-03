"""Plan, download, and verify model artifacts described by lockfiles."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .locks import LockRepository, sha256_file


class ModelError(RuntimeError):
    """Raised when locked model artifacts cannot be prepared safely."""


_VERIFICATION_DIRECTORY = ".h3-flash-verification"


@dataclass(frozen=True, slots=True)
class DownloadGroup:
    repo: str
    revision: str
    local_dir: Path
    files: tuple[str, ...] = ()
    include: tuple[str, ...] = ()


def download_groups(
    locks: LockRepository, lock_name: str, weights_root: Path
) -> tuple[DownloadGroup, ...]:
    model_lock = locks.load(lock_name)
    downloads = model_lock.get("downloads", [])
    if downloads:
        return tuple(
            DownloadGroup(
                entry["repo"],
                entry["revision"],
                weights_root / entry["local_prefix"],
                include=tuple(entry["include"]),
            )
            for entry in downloads
        )
    grouped: dict[tuple[str, str, Path], list[str]] = defaultdict(list)
    default_repo = model_lock.get("repo")
    default_revision = model_lock.get("revision")
    for entry in model_lock.get("files", []):
        repo = entry.get("repo", default_repo)
        revision = entry.get("revision", default_revision)
        if not isinstance(repo, str) or not isinstance(revision, str):
            raise ModelError(f"{entry['path']}: repo and revision are required")
        remote = Path(entry["path"])
        local = Path(entry["local_path"])
        remote_parts = remote.parts
        if (
            len(local.parts) < len(remote_parts)
            or local.parts[-len(remote_parts) :] != remote_parts
        ):
            raise ModelError(
                f"{entry['path']}: local_path must end with the exact remote path"
            )
        prefix_parts = local.parts[: -len(remote_parts)]
        local_dir = weights_root.joinpath(*prefix_parts)
        grouped[(repo, revision, local_dir)].append(entry["path"])
    return tuple(
        DownloadGroup(repo, revision, local_dir, tuple(sorted(files)))
        for (repo, revision, local_dir), files in sorted(
            grouped.items(), key=lambda item: (item[0][0], str(item[0][2]))
        )
    )


def download_commands(
    locks: LockRepository,
    lock_name: str,
    weights_root: Path,
    *,
    hf_bin: str = "hf",
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        (
            hf_bin,
            "download",
            group.repo,
            *group.files,
            *(
                argument
                for pattern in group.include
                for argument in ("--include", pattern)
            ),
            "--revision",
            group.revision,
            "--local-dir",
            str(group.local_dir),
        )
        for group in download_groups(locks, lock_name, weights_root)
    )


def format_download_plan(commands: tuple[tuple[str, ...], ...]) -> str:
    return "\n".join(shlex.join(command) for command in commands)


def download_models(
    locks: LockRepository,
    lock_name: str,
    weights_root: Path,
    *,
    hf_bin: str = "hf",
) -> None:
    if hf_bin == "hf":
        prefix = (sys.executable, "-m", "huggingface_hub.cli.hf")
        command_name = "hf"
    else:
        resolved = shutil.which(hf_bin) if "/" not in hf_bin else hf_bin
        if not resolved or not Path(resolved).is_file():
            raise ModelError(f"Hugging Face CLI not found: {hf_bin}")
        prefix = (str(resolved),)
        command_name = str(resolved)
    for command in download_commands(
        locks, lock_name, weights_root, hf_bin=command_name
    ):
        subprocess.run(prefix + command[1:], check=True)


def verify_models(
    locks: LockRepository,
    lock_name: str,
    weights_root: Path,
    *,
    hash_files: bool = False,
) -> dict[str, Any]:
    model_lock = locks.load(lock_name)
    records = []
    for entry in model_lock.get("files", []):
        path = weights_root / entry["local_path"]
        record: dict[str, Any] = {
            "local_path": entry["local_path"],
            "expected_bytes": entry["bytes"],
            "expected_sha256": entry["sha256"],
        }
        if not path.is_file():
            record.update(status="missing", actual_bytes=None, actual_sha256=None)
        else:
            stat = path.stat()
            size = stat.st_size
            if size != entry["bytes"]:
                record.update(
                    status="wrong_size",
                    actual_bytes=size,
                    actual_sha256=None,
                    actual_mtime_ns=stat.st_mtime_ns,
                )
            elif hash_files:
                digest = sha256_file(path)
                record.update(
                    status="ok" if digest == entry["sha256"] else "wrong_hash",
                    actual_bytes=size,
                    actual_sha256=digest,
                    actual_mtime_ns=stat.st_mtime_ns,
                )
            else:
                record.update(
                    status="ok",
                    actual_bytes=size,
                    actual_sha256=None,
                    actual_mtime_ns=stat.st_mtime_ns,
                )
        records.append(record)
    counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("ok", "missing", "wrong_size", "wrong_hash")
    }
    return {
        "schema_version": 1,
        "lock": locks.reference(lock_name),
        "weights_root": str(weights_root.resolve()),
        "hash_files": hash_files,
        "counts": counts,
        "files": records,
    }


def verification_ok(report: dict[str, Any]) -> bool:
    return not any(
        report["counts"][status] for status in ("missing", "wrong_size", "wrong_hash")
    )


def verification_marker_path(weights_root: Path, lock_name: str) -> Path:
    return weights_root / _VERIFICATION_DIRECTORY / f"{lock_name}.json"


def write_verification_marker(report: dict[str, Any]) -> Path:
    """Persist proof that every locked file passed a full SHA-256 check."""

    if not report.get("hash_files") or not verification_ok(report):
        raise ModelError("a verification marker requires a successful full hash check")
    weights_root = Path(report["weights_root"])
    marker_path = verification_marker_path(weights_root, report["lock"]["name"])
    marker = {
        "schema_version": 1,
        "kind": "locked_model_sha256_verification",
        "lock": report["lock"],
        "files": [
            {
                "local_path": record["local_path"],
                "bytes": record["actual_bytes"],
                "sha256": record["actual_sha256"],
                "mtime_ns": record["actual_mtime_ns"],
            }
            for record in report["files"]
        ],
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{marker_path.name}.", dir=marker_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(marker, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return marker_path


def require_verification_marker(
    locks: LockRepository, lock_name: str, weights_root: Path
) -> dict[str, Any]:
    """Reject weights changed since their last full locked-hash verification."""

    marker_path = verification_marker_path(weights_root, lock_name)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelError(
            f"missing valid verification marker for {lock_name}; run "
            f"`h3-flash model verify --lock {lock_name} --weights-root "
            f"{weights_root} --hash --write-marker`"
        ) from error
    if (
        marker.get("schema_version") != 1
        or marker.get("kind") != "locked_model_sha256_verification"
        or marker.get("lock") != locks.reference(lock_name)
    ):
        raise ModelError(f"stale verification marker: {marker_path}")
    expected = locks.load(lock_name).get("files", [])
    records = marker.get("files")
    if not isinstance(records, list) or len(records) != len(expected):
        raise ModelError(f"incomplete verification marker: {marker_path}")
    by_path = {record.get("local_path"): record for record in records}
    for entry in expected:
        record = by_path.get(entry["local_path"])
        path = weights_root / entry["local_path"]
        if not isinstance(record, dict) or not path.is_file():
            raise ModelError(f"verified model file is missing: {entry['local_path']}")
        stat = path.stat()
        if (
            record.get("bytes") != entry["bytes"]
            or record.get("sha256") != entry["sha256"]
            or stat.st_size != entry["bytes"]
            or stat.st_mtime_ns != record.get("mtime_ns")
        ):
            raise ModelError(
                f"verified model file changed: {entry['local_path']}; rerun full verification"
            )
    return marker


def require_profile_model_verification(
    locks: LockRepository, profile: dict[str, Any], model_root: Path
) -> None:
    """Validate every locked input and any derived Turbo model before loading."""

    lock_name = profile["provenance"]["model_lock"]
    model_lock = locks.load(lock_name)
    weights_root = model_root.expanduser().resolve().parent
    require_verification_marker(locks, lock_name, weights_root)
    base_model_lock = model_lock.get("base_model_lock")
    if isinstance(base_model_lock, str):
        require_verification_marker(locks, base_model_lock, weights_root)
    if profile.get("model", {}).get("turbo_lora"):
        from .turbo import (
            TurboPreparationError,
            require_turbo_verification_marker,
        )

        expected_lora = model_lock["files"][0]["sha256"]
        try:
            require_turbo_verification_marker(
                model_root,
                model_lock=lock_name,
                expected_lora_sha256=expected_lora,
            )
        except TurboPreparationError as error:
            raise ModelError(str(error)) from error
