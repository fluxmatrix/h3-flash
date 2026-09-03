from pathlib import Path

import pytest

from h3_flash.locks import LockRepository
from h3_flash.models import (
    ModelError,
    download_commands,
    require_verification_marker,
    verification_ok,
    verify_models,
    write_verification_marker,
)

ROOT = Path(__file__).resolve().parents[1]


def test_official_download_includes_configs_but_not_ref_transformer(
    tmp_path: Path,
) -> None:
    locks = LockRepository(ROOT / "locks")
    (command,) = download_commands(locks, "models.official", tmp_path, hf_bin="hf")

    assert "modular_model_index.json" in command
    assert "transformer/*" in command
    assert "transformer_ref/*" not in command


def test_verify_reports_missing_without_hashing(tmp_path: Path) -> None:
    locks = LockRepository(ROOT / "locks")
    report = verify_models(locks, "models.fast-turbo4-bf16-inputs", tmp_path)

    assert report["counts"]["missing"] == 1
    assert report["counts"]["ok"] == 0
    assert not verification_ok(report)


def test_full_hash_marker_rejects_a_changed_file(tmp_path: Path) -> None:
    import hashlib
    import os

    locks_root = tmp_path / "locks"
    weights_root = tmp_path / "weights"
    locks_root.mkdir()
    weights_root.mkdir()
    payload = b"locked model bytes"
    digest = hashlib.sha256(payload).hexdigest()
    (weights_root / "model.bin").write_bytes(payload)
    (locks_root / "models.test.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                "[[files]]",
                'path = "model.bin"',
                'local_path = "model.bin"',
                f"bytes = {len(payload)}",
                f'sha256 = "{digest}"',
            )
        ),
        encoding="utf-8",
    )
    locks = LockRepository(locks_root)
    report = verify_models(locks, "models.test", weights_root, hash_files=True)
    marker = write_verification_marker(report)
    assert marker.is_file()
    require_verification_marker(locks, "models.test", weights_root)

    original = (weights_root / "model.bin").stat()
    os.utime(
        weights_root / "model.bin",
        ns=(original.st_atime_ns, original.st_mtime_ns + 1),
    )
    with pytest.raises(ModelError, match="changed"):
        require_verification_marker(locks, "models.test", weights_root)
