#!/usr/bin/env python3
"""Decode-level integrity checks for all OFFICIAL/LOSSLESS/FLASH matrix MP4s."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from h3_flash.evals import EvaluationSuite
from tools.reporting.summarize_latency_matrix import (
    DURATIONS,
    ORIENTATIONS,
    RESOLUTIONS,
    VERSIONS,
    _suite_name,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(ffprobe: Path, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _check(task: dict[str, Any], ffprobe: Path) -> dict[str, Any]:
    result = json.loads(task["result_path"].read_text(encoding="utf-8"))
    artifact = task["result_path"].parent / result["outputs"]["path"]
    probe = _probe(ffprobe, artifact)
    video = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    audio = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "audio"
    )
    failures = []
    actual_frames = int(video.get("nb_read_frames", -1))
    expected_duration = task["num_frames"] / task["fps"]
    video_duration = float(video.get("duration", -1))
    audio_duration = float(audio.get("duration", -1))
    checks = {
        "video_codec_h264": video.get("codec_name") == "h264",
        "video_width": video.get("width") == task["width"],
        "video_height": video.get("height") == task["height"],
        "video_frames": actual_frames == task["num_frames"],
        "video_fps": video.get("avg_frame_rate") == f"{task['fps']}/1",
        "audio_codec_aac": audio.get("codec_name") == "aac",
        "audio_stereo": int(audio.get("channels", -1)) == 2,
        "audio_sample_rate": int(audio.get("sample_rate", -1)) == 32000,
        "audio_nonempty": audio_duration > 0,
        "audio_covers_video_tail": audio_duration + 0.05 >= expected_duration,
        "audio_tail_is_bounded": audio_duration <= expected_duration + 0.10,
        "recorded_bytes": artifact.stat().st_size == result["outputs"]["bytes"],
        "recorded_sha256": _sha256(artifact) == result["outputs"]["sha256"],
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "version": task["version"],
        "suite_id": task["suite_id"],
        "case_id": task["case_id"],
        "artifact": str(artifact.resolve()),
        "status": "ok" if not failures else "error",
        "failures": failures,
        "checks": checks,
        "video": {
            "codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "frames": actual_frames,
            "fps": video.get("avg_frame_rate"),
            "duration": video_duration,
        },
        "audio": {
            "codec": audio.get("codec_name"),
            "channels": audio.get("channels"),
            "sample_rate": audio.get("sample_rate"),
            "duration": audio_duration,
        },
    }


def _safe_check(task: dict[str, Any], ffprobe: Path) -> dict[str, Any]:
    try:
        return _check(task, ffprobe)
    except Exception as error:  # retain every failed artifact in the final report
        return {
            "version": task["version"],
            "suite_id": task["suite_id"],
            "case_id": task["case_id"],
            "artifact": str(task["result_path"].parent.resolve()),
            "status": "error",
            "failures": ["probe_or_validation_exception"],
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    for version in VERSIONS:
        parser.add_argument(f"--{version.lower()}-root", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument(
        "--suites-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/evals",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    roots = {version: getattr(args, f"{version.lower()}_root") for version in VERSIONS}
    tasks = []
    for resolution in RESOLUTIONS:
        for duration in DURATIONS:
            for orientation in ORIENTATIONS:
                suite_name = _suite_name(resolution, duration, orientation)
                suite = EvaluationSuite.load(args.suites_dir / f"{suite_name}.json")
                for version in VERSIONS:
                    for case in suite.cases:
                        tasks.append(
                            {
                                "version": version,
                                "suite_id": suite.suite_id,
                                "case_id": case.case_id,
                                "width": suite.width,
                                "height": suite.height,
                                "num_frames": suite.num_frames,
                                "fps": suite.fps,
                                "result_path": roots[version]
                                / suite_name
                                / "cases"
                                / case.case_id
                                / "result.json",
                            }
                        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda task: _safe_check(task, args.ffprobe), tasks))
    output = {
        "schema_version": 1,
        "scope": "container, stream, full-decode frame count, and recorded artifact integrity",
        "limitations": "does not score semantics, aesthetics, motion, sound content, or synchronization",
        "counts": {
            "ok": sum(record["status"] == "ok" for record in records),
            "error": sum(record["status"] == "error" for record in records),
            "total": len(records),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["counts"], sort_keys=True))
    if output["counts"]["error"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
