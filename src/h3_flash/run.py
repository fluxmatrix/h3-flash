"""Reproducible single-request runners and artifact records."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from .backends.official_diffusers import OfficialDiffusersBackend
from .locks import LockRepository, sha256_file
from .manifest import GenerationRequest, build_manifest
from .models import ModelError, require_profile_model_verification
from .profiles import ProfileRepository
from .runtime.media import HostTransferPool, encode_video_ffmpeg


class RunError(RuntimeError):
    """Raised when a run request violates the public execution contract."""


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _shape(value: Any) -> list[int] | None:
    return list(value.shape) if hasattr(value, "shape") else None


def _to_cpu(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu()
    return value


def _video_to_uint8(video: Any) -> Any:
    """Convert Diffusers' FCHW [0, 1] tensor contract to FHWC uint8."""

    import torch

    if isinstance(video, torch.Tensor) and torch.is_floating_point(video):
        video = video.clamp(0, 1).mul(255).round().to(torch.uint8)
    if (
        isinstance(video, torch.Tensor)
        and video.ndim == 4
        and video.shape[1] in {1, 3, 4}
    ):
        video = video.permute(0, 2, 3, 1).contiguous()
    return video


def _prepare_video_on_device(video: Any) -> Any:
    """Apply the reference encoder's value/layout conversion before D2H."""

    if hasattr(video, "ndim") and video.ndim == 5:
        video = video[0]
    return _video_to_uint8(video)


def run_official_diffusers(
    *,
    request: GenerationRequest,
    model_root: Path,
    output_dir: Path,
    artifact_format: str,
    profiles: ProfileRepository,
    locks: LockRepository,
    generation_device: str = "cuda:0",
    text_device: str = "cuda:0",
    backend: OfficialDiffusersBackend | None = None,
    profile_name: str = "official",
    host_transfer_pool: HostTransferPool | None = None,
) -> dict[str, Any]:
    """Run the official model or one supported single-factor lossless ablation."""

    if artifact_format not in {"latent", "mp4"}:
        raise RunError("artifact_format must be 'latent' or 'mp4'")
    request.validate()
    if profile_name not in {
        "official",
        "lossless-pinned-d2h",
        "lossless-dense-fa4",
        "lossless-1xb200",
    }:
        raise RunError(f"unsupported official-model profile: {profile_name}")
    profile = profiles.resolve(profile_name)
    try:
        require_profile_model_verification(locks, profile, model_root)
    except ModelError as error:
        raise RunError(str(error)) from error
    output_dir = output_dir.expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise RunError(f"output directory already exists: {output_dir}") from error
    process_started = perf_counter()
    owns_backend = backend is None
    attention_backend = profile["attention"]["backend"]
    optimizations = profile.get("optimizations", {})
    fused_qkv = optimizations.get("fused_qkv") is True
    invariant_caches = optimizations.get("invariant_caches") is True
    transformer_fusions = optimizations.get("transformer_fusions") is True
    packed_ulysses = optimizations.get("packed_ulysses") is True
    rank_local_inputs = optimizations.get("rank_local_inputs") is True
    compact_output_gather = optimizations.get("compact_output_gather") is True
    if backend is None:
        fa4_site = os.environ.get("H3_FA4_SITE_PACKAGES")
        backend = OfficialDiffusersBackend(
            model_root,
            generation_device=generation_device,
            text_device=text_device,
            attention_backend=attention_backend,
            fa4_site_packages=Path(fa4_site) if fa4_site else None,
            fused_qkv=fused_qkv,
            invariant_caches=invariant_caches,
            transformer_fusions=transformer_fusions,
            packed_ulysses=packed_ulysses,
            rank_local_inputs=rank_local_inputs,
            compact_output_gather=compact_output_gather,
            vae_compile_mode=profile.get("vae", {}).get("compile_mode"),
        )
    if backend.attention_backend != attention_backend:
        raise RunError(
            "backend/profile attention mismatch: "
            f"backend={backend.attention_backend}, profile={attention_backend}"
        )
    if (
        backend.fused_qkv != fused_qkv
        or backend.invariant_caches != invariant_caches
        or backend.transformer_fusions != transformer_fusions
        or backend.packed_ulysses != packed_ulysses
        or backend.rank_local_inputs != rank_local_inputs
        or backend.compact_output_gather != compact_output_gather
    ):
        raise RunError("backend/profile optimization mismatch")
    runtime_source = backend.source_provenance()
    if (
        backend.generation_device != generation_device
        or backend.text_device != text_device
    ):
        raise RunError(
            "backend/device override mismatch: "
            f"backend=({backend.generation_device},{backend.text_device}), "
            f"request=({generation_device},{text_device})"
        )
    expected_diffusers_commit = locks.load("upstreams")["git"]["diffusers_h3"]["commit"]
    actual_diffusers_commit = runtime_source["diffusers"]["git_commit"]
    if actual_diffusers_commit != expected_diffusers_commit:
        raise RunError(
            "official Diffusers source mismatch: "
            f"expected {expected_diffusers_commit}, got {actual_diffusers_commit!r}"
        )
    manifest = build_manifest(
        profiles,
        profile_name,
        request,
        backend="official-diffusers",
        locks=locks,
        runtime_source=runtime_source,
        execution={
            "artifact_format": artifact_format,
            "generation_device": generation_device,
            "text_device": text_device,
        },
    )
    _write_json_atomic(output_dir / "manifest.json", manifest)
    startup_preparation = None
    if owns_backend and backend.cache_runtime is not None:
        startup_preparation = backend.prepare_fixed_schedule(
            request,
            api_num_inference_steps=profile["sampling"]["api_num_inference_steps"],
        )
    generation = backend.generate(
        request,
        api_num_inference_steps=profile["sampling"]["api_num_inference_steps"],
        output_type="latent" if artifact_format == "latent" else "pt",
    )

    transfer_started = perf_counter()
    use_pinned = profile.get("optimizations", {}).get("persistent_pinned_d2h") is True
    gpu_video_reorder = (
        artifact_format == "mp4"
        and profile.get("optimizations", {}).get("rank0_gpu_video_reorder") is True
    )
    original_video_shape = _shape(generation.state.get("videos"))
    pool_reused = host_transfer_pool is not None and host_transfer_pool.buffer_count > 0
    if use_pinned:
        if host_transfer_pool is None:
            host_transfer_pool = HostTransferPool()
        device_video = generation.state.get("videos")
        if gpu_video_reorder:
            device_video = _prepare_video_on_device(device_video)
        host_values = host_transfer_pool.copy(
            {
                "video": device_video,
                "audio": generation.state.get("audio"),
            }
        )
        video = host_values["video"]
        audio = host_values["audio"]
    else:
        video = _to_cpu(generation.state.get("videos"))
        audio = _to_cpu(generation.state.get("audio"))
    device_to_host_seconds = perf_counter() - transfer_started

    artifact_started = perf_counter()
    if artifact_format == "latent":
        import torch

        artifact_path = output_dir / "latents.pt"
        torch.save(
            {
                "video": video,
                "audio": audio,
                "sampling_rate": generation.state.get("sampling_rate"),
            },
            artifact_path,
        )
    else:
        artifact_path = output_dir / "output.mp4"
        encoder_video = video if gpu_video_reorder else _video_to_uint8(video[0])
        output_config = profile.get("output", {})
        if output_config.get("encoder") == "ffmpeg_raw_pipe":
            encoder_runtime = encode_video_ffmpeg(
                encoder_video,
                fps=request.fps,
                output_path=artifact_path,
                audio=audio[0],
                audio_sample_rate=generation.state.get("sampling_rate"),
                preset=str(output_config.get("preset", "veryfast")),
                crf=int(output_config.get("crf", 20)),
                video_codec=str(output_config.get("video_codec", "libx264")),
                pixel_format=str(output_config.get("pixel_format", "yuv420p")),
                audio_codec=str(output_config.get("audio_codec", "aac")),
            )
        else:
            from diffusers.utils.export_utils import encode_video

            encode_video(
                encoder_video,
                fps=request.fps,
                output_path=str(artifact_path),
                audio=audio[0],
                audio_sample_rate=generation.state.get("sampling_rate"),
            )
            encoder_runtime = {"backend": "diffusers_pyav"}
    artifact_write_seconds = perf_counter() - artifact_started
    process_seconds = perf_counter() - process_started

    result = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": manifest["manifest_sha256"],
        "configuration_sha256": manifest["configuration_sha256"],
        "backend": "official-diffusers",
        "profile": profile_name,
        "request": asdict(request),
        "sampling": {
            "api_num_inference_steps": generation.api_num_inference_steps,
            "model_evaluations": generation.model_evaluations,
            "count_semantics": profile["sampling"]["count_semantics"],
        },
        "timing_seconds": {
            "model_load_charged_to_request": generation.load_seconds
            if owns_backend
            else 0.0,
            "session_model_load": generation.load_seconds,
            "generation": generation.generation_seconds,
            "text_encoding": generation.text_encoding_seconds,
            "diffusion_and_decode": generation.diffusion_and_decode_seconds,
            "device_to_host": device_to_host_seconds,
            "artifact_write": artifact_write_seconds,
            "request_process": process_seconds,
        },
        "startup_preparation": startup_preparation,
        "peak_gpu_memory_bytes": list(generation.peak_gpu_memory_bytes),
        "outputs": {
            "format": artifact_format,
            "path": artifact_path.name,
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
            "video_shape": original_video_shape,
            "encoded_video_shape": _shape(video) if gpu_video_reorder else None,
            "audio_shape": _shape(audio),
            "sampling_rate": generation.state.get("sampling_rate"),
            "host_transfer": "persistent_pinned" if use_pinned else "pageable",
            "host_transfer_pool_reused": pool_reused,
            "video_reorder": "gpu_before_d2h" if gpu_video_reorder else "cpu_after_d2h",
            "encoder": encoder_runtime if artifact_format == "mp4" else None,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "generation_device": generation_device,
            "text_device": text_device,
            "source": runtime_source,
            "attention_audit": generation.attention_audit,
            "cache_audit": generation.cache_audit,
        },
    }
    _write_json_atomic(output_dir / "result.json", result)
    return result
