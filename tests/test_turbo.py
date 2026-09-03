from __future__ import annotations

import json
from pathlib import Path

import pytest

from h3_flash.locks import sha256_file
from h3_flash.turbo import (
    TurboPreparationError,
    require_turbo_verification_marker,
    verify_turbo_model,
    write_turbo_verification_marker,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    source = tmp_path / "official"
    destination = tmp_path / "flash"
    source_transformer = source / "transformer"
    derived_transformer = destination / "transformer"
    source_transformer.mkdir(parents=True)
    derived_transformer.mkdir(parents=True)
    index = {"weight_map": {"block.weight": "model-1.safetensors"}}
    source_index = source_transformer / "diffusion_pytorch_model.safetensors.index.json"
    derived_index = (
        derived_transformer / "diffusion_pytorch_model.safetensors.index.json"
    )
    source_index.write_text(json.dumps(index), encoding="utf-8")
    derived_index.write_text(json.dumps(index), encoding="utf-8")
    shard = derived_transformer / "model-1.safetensors"
    shard.write_bytes(b"derived-shard")
    lora = tmp_path / "turbo.safetensors"
    lora.write_bytes(b"locked-lora")
    lora_sha = sha256_file(lora)
    manifest = {
        "schema_version": 1,
        "derivation": "official_diffusers_plus_lightx2v_turbo_lora",
        "source_transformer_index_sha256": sha256_file(source_index),
        "lora": {
            "bytes": lora.stat().st_size,
            "sha256": lora_sha,
        },
        "bake": {
            "output_shards": [
                {
                    "path": "transformer/model-1.safetensors",
                    "bytes": shard.stat().st_size,
                    "sha256": sha256_file(shard),
                }
            ]
        },
    }
    (destination / "turbo-bake.json").write_text(json.dumps(manifest), encoding="utf-8")
    return destination, source, lora, lora_sha


def test_verify_turbo_model_checks_all_recorded_inputs_and_shards(
    tmp_path: Path,
) -> None:
    destination, source, lora, lora_sha = _fixture(tmp_path)
    report = verify_turbo_model(
        destination,
        source,
        lora,
        expected_lora_sha256=lora_sha,
    )
    assert report["status"] == "ok"
    assert report["verified_shards"] == 1
    assert report["hash_shards"] is True

    marker = write_turbo_verification_marker(report, model_lock="models.turbo")
    assert marker.is_file()
    require_turbo_verification_marker(
        destination,
        model_lock="models.turbo",
        expected_lora_sha256=lora_sha,
    )


def test_verify_turbo_model_rejects_same_size_corruption(tmp_path: Path) -> None:
    destination, source, lora, lora_sha = _fixture(tmp_path)
    (destination / "transformer/model-1.safetensors").write_bytes(b"corrupt-shard")
    with pytest.raises(TurboPreparationError, match="SHA-256 mismatch"):
        verify_turbo_model(
            destination,
            source,
            lora,
            expected_lora_sha256=lora_sha,
        )


def test_verify_turbo_model_rejects_unlocked_lora(tmp_path: Path) -> None:
    destination, source, lora, _ = _fixture(tmp_path)
    with pytest.raises(TurboPreparationError, match="immutable model lock"):
        verify_turbo_model(
            destination,
            source,
            lora,
            expected_lora_sha256="0" * 64,
            hash_shards=False,
        )
