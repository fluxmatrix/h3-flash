"""Multi-process benchmark workers with one GPU assigned to each request."""

from __future__ import annotations

import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from .backends.official_diffusers import OfficialDiffusersBackend
from .evals import EvaluationSuite
from .locks import LockRepository
from .manifest import GenerationRequest
from .models import require_profile_model_verification
from .profiles import ProfileRepository
from .run import _write_json_atomic, run_official_diffusers
from .runtime.media import HostTransferPool


def partition_case_ids(
    suite: EvaluationSuite, worker_index: int, worker_count: int
) -> tuple[str, ...]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError("worker_index must be in [0, worker_count)")
    return tuple(case.case_id for case in suite.cases[worker_index::worker_count])


def active_worker_gpus(
    suite: EvaluationSuite, physical_gpus: tuple[str, ...]
) -> tuple[str, ...]:
    """Use no more one-GPU workers than the suite has independent cases."""

    if not physical_gpus or len(set(physical_gpus)) != len(physical_gpus):
        raise ValueError("physical_gpus must be a non-empty unique list")
    return physical_gpus[: min(len(physical_gpus), len(suite.cases))]


def process_exit_status(exit_codes: list[int]) -> int:
    """Return a shell-safe non-zero status if any child failed or was signalled."""

    for code in exit_codes:
        if code != 0:
            return code if code > 0 else min(255, 128 + abs(code))
    return 0


def suite_summary_is_complete(
    summary_path: Path,
    suite_path: Path,
    profile_name: str,
) -> bool:
    """Validate the minimum contract required for benchmark resume."""

    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        suite = EvaluationSuite.load(suite_path)
    except (OSError, ValueError, TypeError):
        return False
    counts = summary.get("counts")
    expected = len(suite.cases)
    return bool(
        summary.get("schema_version") == 1
        and summary.get("suite_id") == suite.suite_id
        and summary.get("profile") == profile_name
        and isinstance(counts, dict)
        and counts.get("ok") == expected
        and counts.get("error") == 0
        and counts.get("total", expected) == expected
        and isinstance(summary.get("latency_seconds"), dict)
    )


def run_official_worker(
    *,
    suite_path: Path,
    model_root: Path,
    output_root: Path,
    worker_index: int,
    worker_count: int,
    profiles: ProfileRepository,
    locks: LockRepository,
    device: str = "cuda:0",
    profile_name: str = "official",
    fa4_site_packages: Path | None = None,
) -> dict[str, Any]:
    """Run one deterministic shard; callers isolate its physical GPU."""

    suite = EvaluationSuite.load(suite_path)
    case_ids = partition_case_ids(suite, worker_index, worker_count)
    if not case_ids:
        raise ValueError(
            f"worker {worker_index} has no cases; use at most {len(suite.cases)} workers"
        )
    output_root = output_root.expanduser().resolve()
    worker_path = output_root / "workers" / f"worker-{worker_index:02d}.json"
    started_at = datetime.now(UTC).isoformat()
    wall_started = perf_counter()

    profile = profiles.resolve(profile_name)
    require_profile_model_verification(locks, profile, model_root)
    optimizations = profile.get("optimizations", {})
    backend = OfficialDiffusersBackend(
        model_root,
        generation_device=device,
        text_device=device,
        attention_backend=profile["attention"]["backend"],
        fa4_site_packages=fa4_site_packages,
        fused_qkv=optimizations.get("fused_qkv") is True,
        invariant_caches=optimizations.get("invariant_caches") is True,
        transformer_fusions=optimizations.get("transformer_fusions") is True,
        packed_ulysses=optimizations.get("packed_ulysses") is True,
        rank_local_inputs=optimizations.get("rank_local_inputs") is True,
        compact_output_gather=optimizations.get("compact_output_gather") is True,
        vae_compile_mode=profile.get("vae", {}).get("compile_mode"),
    )
    backend.load()

    # One target-shape model evaluation warms kernels and allocator state. It
    # is not included in any request latency or quality artifact.
    warmup_case = suite.case(case_ids[0])
    warmup_request = GenerationRequest(
        prompt=warmup_case.prompt,
        seed=warmup_case.seed,
        width=suite.width,
        height=suite.height,
        num_frames=suite.num_frames,
        fps=suite.fps,
    )
    warmup_started = perf_counter()
    if backend.cache_runtime is not None:
        warmup = backend.prepare_fixed_schedule(warmup_request)
    else:
        warmup = backend.generate(
            warmup_request,
            api_num_inference_steps=2,
            output_type="latent",
        )
    warmup_seconds = perf_counter() - warmup_started
    del warmup

    records: list[dict[str, Any]] = []
    host_transfer_pool = (
        HostTransferPool()
        if profile.get("optimizations", {}).get("persistent_pinned_d2h") is True
        else None
    )
    for case_id in case_ids:
        case_dir = output_root / "cases" / case_id
        try:
            result = run_official_diffusers(
                request=suite.request(case_id),
                model_root=model_root,
                output_dir=case_dir,
                artifact_format="mp4",
                profiles=profiles,
                locks=locks,
                generation_device=device,
                text_device=device,
                backend=backend,
                profile_name=profile_name,
                host_transfer_pool=host_transfer_pool,
            )
            records.append(
                {
                    "case_id": case_id,
                    "status": "ok",
                    "result": str(case_dir / "result.json"),
                    "generation_seconds": result["timing_seconds"]["generation"],
                    "request_process_seconds": result["timing_seconds"][
                        "request_process"
                    ],
                }
            )
        except Exception as error:  # noqa: BLE001 - isolate independent cases.
            failure = {
                "schema_version": 1,
                "case_id": case_id,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            _write_json_atomic(case_dir / "error.json", failure)
            records.append(failure)

    summary = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "profile": profile_name,
        "backend": "official-diffusers",
        "allocation": {
            "suite_worker_count": worker_count,
            "worker_index": worker_index,
            "gpu_per_request": 1,
            "visible_device": device,
        },
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "timing_seconds": {
            "worker_wall": perf_counter() - wall_started,
            "model_load": backend.load_seconds,
            "one_forward_warmup": warmup_seconds,
        },
        "case_ids": list(case_ids),
        "counts": {
            "ok": sum(record["status"] == "ok" for record in records),
            "error": sum(record["status"] == "error" for record in records),
        },
        "records": records,
    }
    _write_json_atomic(worker_path, summary)
    return summary


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


def summarize_official_suite(
    output_root: Path, worker_count: int, suite_wall: float, profile_name: str
) -> dict[str, Any]:
    workers = []
    for index in range(worker_count):
        path = output_root / "workers" / f"worker-{index:02d}.json"
        workers.append(json.loads(path.read_text(encoding="utf-8")))
    records = [record for worker in workers for record in worker["records"]]
    successful = [record for record in records if record["status"] == "ok"]
    generation = [record["generation_seconds"] for record in successful]
    request_process = [record["request_process_seconds"] for record in successful]
    summary = {
        "schema_version": 1,
        "suite_id": workers[0]["suite_id"],
        "profile": profile_name,
        "backend": "official-diffusers",
        "allocation": {
            "concurrent_workers": worker_count,
            "gpu_per_request": 1,
            "total_physical_gpus": worker_count,
            "interpretation": "parallel suite throughput; per-request latency is not divided by worker count",
        },
        "counts": {
            "ok": len(successful),
            "error": len(records) - len(successful),
            "total": len(records),
        },
        "suite_wall_seconds": suite_wall,
        "suite_throughput_videos_per_minute": len(successful) * 60 / suite_wall,
        "total_request_gpu_seconds": sum(request_process),
        "latency_seconds": {
            "generation": _distribution(generation) if generation else None,
            "request_process": _distribution(request_process)
            if request_process
            else None,
        },
        "workers": [
            str(output_root / "workers" / f"worker-{index:02d}.json")
            for index in range(worker_count)
        ],
    }
    _write_json_atomic(output_root / "summary.json", summary)
    return summary


def launch_official_suite(
    *,
    suite_path: Path,
    model_root: Path,
    output_root: Path,
    physical_gpus: tuple[str, ...],
    profiles_dir: Path | None = None,
    locks_dir: Path | None = None,
    profile_name: str = "official",
    fa4_site_packages: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Launch one isolated single-GPU worker per physical GPU and record it."""

    suite = EvaluationSuite.load(suite_path)
    requested_physical_gpus = physical_gpus
    physical_gpus = active_worker_gpus(suite, physical_gpus)
    output_root = output_root.expanduser().resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(f"output root already exists: {output_root}") from error
    logs_dir = output_root / "logs"
    logs_dir.mkdir()
    base = [sys.executable, "-m", "h3_flash.cli"]
    if profiles_dir is not None:
        base += ["--profiles-dir", str(profiles_dir.resolve())]
    if locks_dir is not None:
        base += ["--locks-dir", str(locks_dir.resolve())]

    started_at = datetime.now(UTC).isoformat()
    wall_started = perf_counter()
    specifications = []
    processes: list[tuple[subprocess.Popen, Any]] = []
    for worker_index, physical_gpu in enumerate(physical_gpus):
        command = base + [
            "benchmark",
            "official-worker",
            "--suite-file",
            str(suite_path.resolve()),
            "--model-root",
            str(model_root.resolve()),
            "--output-root",
            str(output_root),
            "--worker-index",
            str(worker_index),
            "--worker-count",
            str(len(physical_gpus)),
        ]
        if profile_name != "official":
            command += ["--profile", profile_name]
        if fa4_site_packages is not None:
            command += ["--fa4-site-packages", str(fa4_site_packages.resolve())]
        environment = os.environ.copy()
        overrides = {
            "CUDA_VISIBLE_DEVICES": physical_gpu,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        environment.update(overrides)
        log_path = logs_dir / f"worker-{worker_index:02d}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((process, log_handle))
        specifications.append(
            {
                "worker_index": worker_index,
                "physical_gpu": physical_gpu,
                "argv": command,
                "shell_rendering": shlex.join(command),
                "environment": overrides,
                "log": str(log_path),
                "pid": process.pid,
            }
        )

    launch_record = {
        "schema_version": 1,
        "started_at": started_at,
        "launcher_pid": os.getpid(),
        "launcher_argv": sys.argv,
        "launcher_shell_rendering": shlex.join(sys.argv),
        "python_executable": sys.executable,
        "inherited_environment": {
            name: os.environ[name]
            for name in ("H3_FLASH_ROOT", "PYTHONPATH")
            if name in os.environ
        },
        "suite_path": str(suite_path.resolve()),
        "profile": profile_name,
        "fa4_site_packages": (
            str(fa4_site_packages.resolve()) if fa4_site_packages is not None else None
        ),
        "model_root": str(model_root.resolve()),
        "output_root": str(output_root),
        "requested_physical_gpus": list(requested_physical_gpus),
        "active_physical_gpus": list(physical_gpus),
        "workers": specifications,
        "status": "running",
    }
    _write_json_atomic(output_root / "launch.json", launch_record)

    exit_codes = []
    for process, log_handle in processes:
        exit_codes.append(process.wait())
        log_handle.close()
    suite_wall = perf_counter() - wall_started
    launch_record.update(
        status="complete" if all(code == 0 for code in exit_codes) else "error",
        finished_at=datetime.now(UTC).isoformat(),
        suite_wall_seconds=suite_wall,
        exit_codes=exit_codes,
    )
    _write_json_atomic(output_root / "launch.json", launch_record)
    if all(code == 0 for code in exit_codes):
        summary = summarize_official_suite(
            output_root, len(physical_gpus), suite_wall, profile_name
        )
    else:
        summary = {
            "schema_version": 1,
            "status": "worker_error",
            "exit_codes": exit_codes,
            "suite_wall_seconds": suite_wall,
        }
        _write_json_atomic(output_root / "summary.json", summary)
    return summary, process_exit_status(exit_codes)
