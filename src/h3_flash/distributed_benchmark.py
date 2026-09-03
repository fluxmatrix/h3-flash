"""Collective eight-GPU lossless benchmark and reproducible launcher."""

from __future__ import annotations

import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from .backends.official_diffusers import OfficialDiffusersBackend
from .evals import EvaluationSuite
from .locks import LockRepository, sha256_file
from .manifest import build_manifest
from .models import require_profile_model_verification
from .profiles import ProfileRepository
from .run import (
    _prepare_video_on_device,
    _to_cpu,
    _video_to_uint8,
    _write_json_atomic,
)
from .runtime.media import HostTransferPool, encode_video_ffmpeg


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "standard_deviation": statistics.pstdev(values),
    }


def _maximum_across_ranks(values: list[float], device: str) -> list[float]:
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return [float(value) for value in tensor.cpu().tolist()]


def _peaks_across_ranks(local_peak: int, device: str) -> list[int]:
    import torch
    import torch.distributed as dist

    value = torch.tensor([local_peak], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    return [int(item.item()) for item in gathered]


def run_distributed_suite_worker(
    *,
    suite_path: Path,
    model_root: Path,
    output_root: Path,
    profiles: ProfileRepository,
    locks: LockRepository,
    profile_name: str = "lossless",
) -> dict[str, Any] | None:
    """Run one collective suite; invoke only below ``torch.distributed.run``."""

    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 2:
        raise RuntimeError("distributed benchmark requires at least two ranks")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(device),
    )

    output_root = output_root.expanduser().resolve()
    suite = EvaluationSuite.load(suite_path)
    profile = profiles.resolve(profile_name)
    require_profile_model_verification(locks, profile, model_root)
    expected_world = int(profile.get("parallel", {}).get("world_size", 1))
    if expected_world != world_size:
        raise RuntimeError(
            f"profile world_size={expected_world} but launcher world_size={world_size}"
        )
    optimizations = profile.get("optimizations", {})
    backend = OfficialDiffusersBackend(
        model_root,
        generation_device=device,
        text_device=device,
        attention_backend=profile["attention"]["backend"],
        fused_qkv=optimizations.get("fused_qkv") is True,
        invariant_caches=optimizations.get("invariant_caches") is True,
        transformer_fusions=optimizations.get("transformer_fusions") is True,
        packed_ulysses=optimizations.get("packed_ulysses") is True,
        rank_local_inputs=optimizations.get("rank_local_inputs") is True,
        compact_output_gather=optimizations.get("compact_output_gather") is True,
        vae_compile_mode=profile.get("vae", {}).get("compile_mode"),
        ulysses_degree=world_size,
        vae_clip_parallel=profile.get("vae", {}).get("video_decoder")
        == "official_clip_parallel",
    )
    wall_started = perf_counter()
    started_at = datetime.now(UTC).isoformat()
    backend.load()
    dist.barrier()

    # Cache profiles populate their exact declared fixed-schedule table during
    # startup and release the no-longer-needed 26 GB of AdaLN projections.
    # Other profiles retain the official one-forward warmup protocol.
    warmup_started = perf_counter()
    if backend.cache_runtime is not None:
        schedule_precompute = backend.prepare_fixed_schedule(
            suite.request(suite.cases[0].case_id),
            api_num_inference_steps=profile["sampling"]["api_num_inference_steps"],
        )
    else:
        backend.generate(
            suite.request(suite.cases[0].case_id),
            api_num_inference_steps=2,
            output_type="latent",
        )
        schedule_precompute = None
    dist.barrier()
    warmup_seconds = perf_counter() - warmup_started

    # Latency reports are for a resident service. If the optional equivalent
    # VAE compile path is selected, compile and numerically gate its static
    # tile graph during startup rather than charging the first request.
    vae_compile_warmup = None
    if backend.vae_compile_mode is not None:
        vae_started = perf_counter()
        compiled = backend.generate(
            suite.request(suite.cases[0].case_id),
            api_num_inference_steps=profile["sampling"]["api_num_inference_steps"],
            output_type="pt",
        )
        dist.barrier()
        vae_compile_warmup = perf_counter() - vae_started
        del compiled

    runtime_source = backend.source_provenance() if rank == 0 else None
    if rank == 0:
        expected_commit = locks.load("upstreams")["git"]["diffusers_h3"]["commit"]
        actual_commit = runtime_source["diffusers"]["git_commit"]
        if actual_commit != expected_commit:
            raise RuntimeError(
                f"official Diffusers source mismatch: expected {expected_commit}, "
                f"got {actual_commit!r}"
            )
        if profile["model"].get("turbo_lora"):
            derived = runtime_source.get("derived_model")
            if not isinstance(derived, dict):
                raise RuntimeError(
                    "Turbo profile requires a model root prepared with "
                    "scripts/prepare_turbo_model.py"
                )
            model_lock = locks.load(profile["provenance"]["model_lock"])
            expected_lora_sha = model_lock["files"][0]["sha256"]
            actual_lora_sha = derived["manifest"]["lora"]["sha256"]
            if actual_lora_sha != expected_lora_sha:
                raise RuntimeError(
                    f"Turbo LoRA mismatch: expected {expected_lora_sha}, "
                    f"got {actual_lora_sha}"
                )

    use_pinned_d2h = optimizations.get("persistent_pinned_d2h") is True
    gpu_video_reorder = optimizations.get("rank0_gpu_video_reorder") is True
    pool = HostTransferPool() if rank == 0 and use_pinned_d2h else None
    records: list[dict[str, Any]] = []
    for case in suite.cases:
        request = suite.request(case.case_id)
        case_dir = output_root / "cases" / case.case_id
        manifest = None
        if rank == 0:
            manifest = build_manifest(
                profiles,
                profile_name,
                request,
                backend="official-diffusers-distributed",
                locks=locks,
                runtime_source=runtime_source,
                execution={
                    "artifact_format": "mp4",
                    "generation_device": "distributed_local_rank",
                    "text_device": "distributed_local_rank",
                    "world_size": world_size,
                },
            )
            _write_json_atomic(case_dir / "manifest.json", manifest)

        dist.barrier()
        request_started = perf_counter()
        generation = backend.generate(
            request,
            api_num_inference_steps=profile["sampling"]["api_num_inference_steps"],
            output_type="pt",
        )
        generation_max, text_max, diffusion_decode_max = _maximum_across_ranks(
            [
                generation.generation_seconds,
                generation.text_encoding_seconds,
                generation.diffusion_and_decode_seconds,
            ],
            device,
        )
        local_peak = max(generation.peak_gpu_memory_bytes)
        peaks = _peaks_across_ranks(local_peak, device)

        device_to_host_seconds = 0.0
        artifact_write_seconds = 0.0
        if rank == 0:
            assert manifest is not None
            pool_reused = pool is not None and pool.buffer_count > 0
            transfer_started = perf_counter()
            original_video = generation.state.get("videos")
            transfer_video = (
                _prepare_video_on_device(original_video)
                if gpu_video_reorder
                else original_video
            )
            if pool is not None:
                host = pool.copy(
                    {
                        "video": transfer_video,
                        "audio": generation.state.get("audio"),
                    }
                )
            else:
                host = {
                    "video": _to_cpu(transfer_video),
                    "audio": _to_cpu(generation.state.get("audio")),
                }
            device_to_host_seconds = perf_counter() - transfer_started
            video = host["video"]
            audio = host["audio"]
            encoder_video = video if gpu_video_reorder else _video_to_uint8(video[0])

            artifact_started = perf_counter()
            artifact_path = case_dir / "output.mp4"
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
            request_process_seconds = perf_counter() - request_started
        else:
            pool_reused = False
            artifact_path = None
            request_process_seconds = perf_counter() - request_started

        request_process_max = _maximum_across_ranks([request_process_seconds], device)[
            0
        ]
        if rank == 0:
            assert artifact_path is not None and manifest is not None
            result = {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "manifest_sha256": manifest["manifest_sha256"],
                "configuration_sha256": manifest["configuration_sha256"],
                "backend": "official-diffusers-distributed",
                "profile": profile_name,
                "request": asdict(request),
                "sampling": {
                    "api_num_inference_steps": generation.api_num_inference_steps,
                    "model_evaluations": generation.model_evaluations,
                    "count_semantics": profile["sampling"]["count_semantics"],
                },
                "timing_seconds": {
                    "model_load_charged_to_request": 0.0,
                    "session_model_load": backend.load_seconds,
                    "generation": generation_max,
                    "text_encoding": text_max,
                    "diffusion_and_decode": diffusion_decode_max,
                    "device_to_host": device_to_host_seconds,
                    "artifact_write": artifact_write_seconds,
                    "request_process": request_process_max,
                },
                "peak_gpu_memory_bytes": peaks,
                "outputs": {
                    "format": "mp4",
                    "path": artifact_path.name,
                    "bytes": artifact_path.stat().st_size,
                    "sha256": sha256_file(artifact_path),
                    "video_shape": list(original_video.shape),
                    "encoded_video_shape": list(encoder_video.shape),
                    "audio_shape": list(audio.shape),
                    "sampling_rate": generation.state.get("sampling_rate"),
                    "host_transfer": (
                        "persistent_pinned" if use_pinned_d2h else "tensor_cpu"
                    ),
                    "host_transfer_pool_reused": pool_reused,
                    "video_reorder": (
                        "gpu_before_d2h" if gpu_video_reorder else "cpu_after_d2h"
                    ),
                    "encoder": encoder_runtime,
                },
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "world_size": world_size,
                    "source": runtime_source,
                    "attention_audit": generation.attention_audit,
                    "cache_audit": generation.cache_audit,
                },
            }
            _write_json_atomic(case_dir / "result.json", result)
            records.append(
                {
                    "case_id": case.case_id,
                    "status": "ok",
                    "result": str(case_dir / "result.json"),
                    "generation_seconds": generation_max,
                    "request_process_seconds": request_process_max,
                }
            )
            print(
                f"[h3-flash] {case.case_id}: generation={generation_max:.3f}s "
                f"e2e={request_process_max:.3f}s",
                flush=True,
            )
        del generation
        dist.barrier()

    summary = None
    if rank == 0:
        generation_values = [record["generation_seconds"] for record in records]
        process_values = [record["request_process_seconds"] for record in records]
        suite_wall = perf_counter() - wall_started
        summary = {
            "schema_version": 1,
            "suite_id": suite.suite_id,
            "profile": profile_name,
            "backend": "official-diffusers-distributed",
            "allocation": {
                "concurrent_requests": 1,
                "gpu_per_request": world_size,
                "total_physical_gpus": world_size,
                "topology": profile["execution"]["topology"],
                "interpretation": "per-request latency uses all ranks collectively",
            },
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "counts": {"ok": len(records), "error": 0, "total": len(records)},
            "suite_wall_seconds": suite_wall,
            "suite_throughput_videos_per_minute": len(records) * 60 / suite_wall,
            "timing_seconds": {
                "model_load": backend.load_seconds,
                "startup_preparation": {
                    "seconds": warmup_seconds,
                    "mode": (
                        "fixed_schedule_precompute_and_freeze"
                        if schedule_precompute is not None
                        else "one_model_evaluation_warmup"
                    ),
                },
                "fixed_schedule_precompute": schedule_precompute,
                "vae_compile_warmup": vae_compile_warmup,
            },
            "latency_seconds": {
                "generation": _distribution(generation_values),
                "request_process": _distribution(process_values),
            },
            "records": records,
        }
        _write_json_atomic(output_root / "summary.json", summary)
    dist.barrier()
    dist.destroy_process_group()
    return summary


def launch_distributed_suite(
    *,
    suite_path: Path,
    model_root: Path,
    output_root: Path,
    physical_gpus: tuple[str, ...],
    profiles_dir: Path | None = None,
    locks_dir: Path | None = None,
    profile_name: str = "lossless",
) -> tuple[dict[str, Any], int]:
    """Launch a collective suite and persist the exact torchrun command."""

    if len(physical_gpus) < 2 or len(set(physical_gpus)) != len(physical_gpus):
        raise ValueError("physical_gpus must be a unique list of at least two GPUs")
    output_root = output_root.expanduser().resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(f"output root already exists: {output_root}") from error
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(len(physical_gpus)),
        "--module",
        "h3_flash.cli",
    ]
    if profiles_dir is not None:
        command += ["--profiles-dir", str(profiles_dir.resolve())]
    if locks_dir is not None:
        command += ["--locks-dir", str(locks_dir.resolve())]
    command += [
        "benchmark",
        "distributed-worker",
        "--suite-file",
        str(suite_path.resolve()),
        "--model-root",
        str(model_root.resolve()),
        "--output-root",
        str(output_root),
        "--profile",
        profile_name,
    ]
    environment = os.environ.copy()
    overrides = {
        "CUDA_VISIBLE_DEVICES": ",".join(physical_gpus),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if os.environ.get("H3_FLASH_FFMPEG_BIN"):
        overrides["H3_FLASH_FFMPEG_BIN"] = os.environ["H3_FLASH_FFMPEG_BIN"]
    environment.update(overrides)
    log_path = output_root / "distributed.log"
    launch = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "launcher_pid": os.getpid(),
        "launcher_argv": sys.argv,
        "python_executable": sys.executable,
        "argv": command,
        "shell_rendering": shlex.join(command),
        "environment": overrides,
        "suite_path": str(suite_path.resolve()),
        "model_root": str(model_root.resolve()),
        "output_root": str(output_root),
        "profile": profile_name,
        "log": str(log_path),
        "status": "running",
    }
    _write_json_atomic(output_root / "launch.json", launch)
    started = perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    launch.update(
        status="complete" if completed.returncode == 0 else "error",
        finished_at=datetime.now(UTC).isoformat(),
        suite_wall_seconds=perf_counter() - started,
        exit_code=completed.returncode,
    )
    _write_json_atomic(output_root / "launch.json", launch)
    if completed.returncode == 0:
        import json

        summary = json.loads((output_root / "summary.json").read_text())
    else:
        summary = {
            "schema_version": 1,
            "status": "worker_error",
            "exit_code": completed.returncode,
        }
        _write_json_atomic(output_root / "summary.json", summary)
    return summary, completed.returncode
