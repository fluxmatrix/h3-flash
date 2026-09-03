"""Prepare an auditable Diffusers model derived with the LightX2V Turbo LoRA."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .locks import sha256_file


class TurboPreparationError(RuntimeError):
    """Raised when a Turbo-derived model cannot be prepared safely."""


_TURBO_VERIFICATION_MARKER = ".h3-flash-verified.json"


_LORA_SUFFIXES = (
    (".lora_A.default.weight", "A"),
    (".lora_B.default.weight", "B"),
    (".lora_A.weight", "A"),
    (".lora_B.weight", "B"),
)


def _lora_pairs(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for key, tensor in state.items():
        for suffix, side in _LORA_SUFFIXES:
            if key.endswith(suffix):
                pairs.setdefault(key[: -len(suffix)], {})[side] = tensor
                break
        else:
            raise TurboPreparationError(f"unexpected Turbo LoRA key: {key}")
    incomplete = [stem for stem, pair in pairs.items() if set(pair) != {"A", "B"}]
    if incomplete:
        raise TurboPreparationError(f"incomplete Turbo LoRA pairs: {incomplete[:4]}")
    return pairs


def _copy_model_tree(source: Path, destination: Path) -> None:
    """Copy mutable transformer shards and hard-link immutable components."""

    destination.mkdir()
    for child in source.iterdir():
        target = destination / child.name
        if child.name == "transformer":
            shutil.copytree(child, target, copy_function=shutil.copy2)
        elif child.is_dir():
            shutil.copytree(child, target, copy_function=os.link, symlinks=True)
        elif child.is_symlink():
            target.symlink_to(os.readlink(child))
        else:
            os.link(child, target)


def _bake_component(
    component: Path,
    lora_path: Path,
    *,
    strength: float,
    device: str,
    compute_dtype_name: str,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    index_path = component / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.is_file():
        raise TurboPreparationError(f"missing transformer index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    with safe_open(lora_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    alpha = float(metadata.get("alpha", 0))
    if alpha <= 0:
        raise TurboPreparationError("Turbo LoRA metadata requires positive alpha")
    lora = load_file(lora_path)
    pairs = _lora_pairs(lora)
    by_shard: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for stem, pair in pairs.items():
        target = f"{stem}.weight"
        shard = weight_map.get(target)
        if shard is None:
            raise TurboPreparationError(
                f"LoRA target absent from transformer: {target}"
            )
        by_shard.setdefault(shard, []).append((target, pair))

    compute_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[compute_dtype_name]
    compute_device = torch.device(device)
    output_shards = []
    for shard_name, targets in sorted(by_shard.items()):
        shard_path = component / shard_name
        state = load_file(shard_path)
        for target, pair in targets:
            down = pair["A"]
            up = pair["B"]
            rank = int(down.shape[0])
            scale = strength * alpha / rank
            update = (
                up.to(device=compute_device, dtype=compute_dtype)
                @ down.to(device=compute_device, dtype=compute_dtype)
            ).mul_(scale)
            base = state[target]
            if update.shape != base.shape:
                raise TurboPreparationError(
                    f"shape mismatch for {target}: update={tuple(update.shape)} "
                    f"base={tuple(base.shape)}"
                )
            state[target] = (
                base.to(device=compute_device, dtype=compute_dtype) + update
            ).to(device="cpu", dtype=base.dtype)
            del update
        temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
        save_file(state, temporary, metadata={"format": "pt"})
        os.replace(temporary, shard_path)
        output_shards.append(
            {
                "path": str(Path("transformer") / shard_name),
                "bytes": shard_path.stat().st_size,
                "sha256": sha256_file(shard_path),
                "targets": len(targets),
            }
        )
        print(f"[h3-flash] baked {len(targets)} targets into {shard_name}", flush=True)

    return {
        "alpha": alpha,
        "strength": strength,
        "pairs": len(pairs),
        "target_shards": len(by_shard),
        "compute_device": str(compute_device),
        "compute_dtype": compute_dtype_name,
        "output_shards": output_shards,
    }


def prepare_turbo_model(
    source_model_root: Path,
    destination_model_root: Path,
    lora_path: Path,
    *,
    strength: float = 1.0,
    device: str = "cpu",
    compute_dtype: str = "float32",
) -> dict[str, Any]:
    """Create a new model root without mutating the locked official source."""

    source = source_model_root.expanduser().resolve(strict=True)
    destination = destination_model_root.expanduser().resolve()
    lora = lora_path.expanduser().resolve(strict=True)
    if strength < 0:
        raise TurboPreparationError("strength must be non-negative")
    if compute_dtype not in {"float32", "bfloat16"}:
        raise TurboPreparationError(f"unsupported compute dtype: {compute_dtype}")
    if not (source / "modular_model_index.json").is_file():
        raise TurboPreparationError(f"not a Diffusers H3 model root: {source}")
    if destination.exists():
        raise TurboPreparationError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise TurboPreparationError(
            f"temporary destination already exists: {temporary}"
        )

    try:
        _copy_model_tree(source, temporary)
        bake = _bake_component(
            temporary / "transformer",
            lora,
            strength=strength,
            device=device,
            compute_dtype_name=compute_dtype,
        )
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "derivation": "official_diffusers_plus_lightx2v_turbo_lora",
            "source_model_root": str(source),
            "source_transformer_index_sha256": sha256_file(
                source / "transformer/diffusion_pytorch_model.safetensors.index.json"
            ),
            "lora": {
                "path": str(lora),
                "bytes": lora.stat().st_size,
                "sha256": sha256_file(lora),
            },
            "bake": bake,
        }
        manifest_path = temporary / "turbo-bake.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        print(f"[h3-flash] prepared Turbo model: {destination}", flush=True)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_turbo_model(
    destination_model_root: Path,
    source_model_root: Path,
    lora_path: Path,
    *,
    expected_lora_sha256: str | None = None,
    hash_shards: bool = True,
) -> dict[str, Any]:
    """Verify a prepared Turbo model against its source, LoRA, and bake manifest."""

    destination = destination_model_root.expanduser().resolve(strict=True)
    source = source_model_root.expanduser().resolve(strict=True)
    lora = lora_path.expanduser().resolve(strict=True)
    manifest_path = destination / "turbo-bake.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TurboPreparationError(
            f"cannot read Turbo bake manifest: {manifest_path}"
        ) from error
    if manifest.get("schema_version") != 1:
        raise TurboPreparationError("Turbo bake manifest schema_version must be 1")
    if manifest.get("derivation") != "official_diffusers_plus_lightx2v_turbo_lora":
        raise TurboPreparationError("unexpected Turbo model derivation")

    source_index = source / "transformer/diffusion_pytorch_model.safetensors.index.json"
    actual_source_index_sha = sha256_file(source_index)
    if actual_source_index_sha != manifest.get("source_transformer_index_sha256"):
        raise TurboPreparationError(
            "official Transformer index differs from bake input"
        )

    lora_record = manifest.get("lora")
    if not isinstance(lora_record, dict):
        raise TurboPreparationError("Turbo bake manifest has no LoRA record")
    actual_lora_sha = sha256_file(lora)
    if lora.stat().st_size != lora_record.get("bytes"):
        raise TurboPreparationError("Turbo LoRA byte size differs from bake input")
    if actual_lora_sha != lora_record.get("sha256"):
        raise TurboPreparationError("Turbo LoRA SHA-256 differs from bake input")
    if expected_lora_sha256 is not None and actual_lora_sha != expected_lora_sha256:
        raise TurboPreparationError(
            "Turbo LoRA SHA-256 differs from the immutable model lock"
        )

    derived_index_path = (
        destination / "transformer/diffusion_pytorch_model.safetensors.index.json"
    )
    try:
        derived_index = json.loads(derived_index_path.read_text(encoding="utf-8"))
        indexed_shards = {
            str(Path("transformer") / name)
            for name in derived_index["weight_map"].values()
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise TurboPreparationError("invalid derived Transformer index") from error

    bake = manifest.get("bake")
    output_shards = bake.get("output_shards") if isinstance(bake, dict) else None
    if not isinstance(output_shards, list) or not output_shards:
        raise TurboPreparationError("Turbo bake manifest has no output shard records")
    recorded_shards = {record.get("path") for record in output_shards}
    if recorded_shards != indexed_shards:
        raise TurboPreparationError(
            "Turbo bake manifest shard set differs from the derived Transformer index"
        )

    verified_bytes = 0
    verified = []
    for record in output_shards:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise TurboPreparationError(f"unsafe Turbo shard path: {relative}")
        shard = destination / relative
        if not shard.is_file():
            raise TurboPreparationError(f"missing Turbo shard: {relative}")
        actual_bytes = shard.stat().st_size
        if actual_bytes != record.get("bytes"):
            raise TurboPreparationError(f"Turbo shard byte mismatch: {relative}")
        actual_sha = sha256_file(shard) if hash_shards else None
        if hash_shards and actual_sha != record.get("sha256"):
            raise TurboPreparationError(f"Turbo shard SHA-256 mismatch: {relative}")
        verified_bytes += actual_bytes
        verified.append(
            {
                "path": str(relative),
                "bytes": actual_bytes,
                "sha256": actual_sha,
                "mtime_ns": shard.stat().st_mtime_ns,
            }
        )
    return {
        "schema_version": 1,
        "status": "ok",
        "destination_model_root": str(destination),
        "source_model_root": str(source),
        "lora_sha256": actual_lora_sha,
        "hash_shards": hash_shards,
        "verified_shards": len(verified),
        "verified_bytes": verified_bytes,
        "shards": verified,
    }


def write_turbo_verification_marker(report: dict[str, Any], *, model_lock: str) -> Path:
    """Persist a cheap runtime guard after all derived shards were hashed."""

    if report.get("status") != "ok" or not report.get("hash_shards"):
        raise TurboPreparationError(
            "a Turbo verification marker requires a successful full shard hash"
        )
    destination = Path(report["destination_model_root"])
    bake_manifest = destination / "turbo-bake.json"
    marker = {
        "schema_version": 1,
        "kind": "turbo_derived_sha256_verification",
        "model_lock": model_lock,
        "lora_sha256": report["lora_sha256"],
        "bake_manifest_sha256": sha256_file(bake_manifest),
        "shards": report["shards"],
    }
    marker_path = destination / _TURBO_VERIFICATION_MARKER
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{marker_path.name}.", dir=destination
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


def require_turbo_verification_marker(
    destination_model_root: Path,
    *,
    model_lock: str,
    expected_lora_sha256: str,
) -> dict[str, Any]:
    """Reject a derived model changed since its full verification."""

    destination = destination_model_root.expanduser().resolve()
    marker_path = destination / _TURBO_VERIFICATION_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TurboPreparationError(
            "missing valid derived-model verification marker; rerun "
            "scripts/verify_turbo_model.py without --size-only"
        ) from error
    bake_manifest = destination / "turbo-bake.json"
    if (
        marker.get("schema_version") != 1
        or marker.get("kind") != "turbo_derived_sha256_verification"
        or marker.get("model_lock") != model_lock
        or marker.get("lora_sha256") != expected_lora_sha256
        or not bake_manifest.is_file()
        or marker.get("bake_manifest_sha256") != sha256_file(bake_manifest)
    ):
        raise TurboPreparationError(f"stale Turbo verification marker: {marker_path}")
    shards = marker.get("shards")
    if not isinstance(shards, list) or not shards:
        raise TurboPreparationError(
            f"incomplete Turbo verification marker: {marker_path}"
        )
    for record in shards:
        relative = Path(record.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise TurboPreparationError("unsafe path in Turbo verification marker")
        shard = destination / relative
        if not shard.is_file():
            raise TurboPreparationError(f"verified Turbo shard is missing: {relative}")
        stat = shard.stat()
        if stat.st_size != record.get("bytes") or stat.st_mtime_ns != record.get(
            "mtime_ns"
        ):
            raise TurboPreparationError(
                f"verified Turbo shard changed: {relative}; rerun full verification"
            )
    return marker
