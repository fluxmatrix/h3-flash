#!/usr/bin/env python3
"""Compare non-semantic video/audio integrity and signal statistics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image

from h3_flash.evals import EvaluationSuite


def _artifact(root: Path, case_id: str) -> Path:
    case_root = root / "cases" / case_id
    result = json.loads((case_root / "result.json").read_text(encoding="utf-8"))
    return case_root / result["outputs"]["path"]


def _video(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        metadata = {
            "codec": stream.codec_context.name,
            "width": stream.width,
            "height": stream.height,
            "fps": float(stream.average_rate),
        }
        for frame in container.decode(video=0):
            image = frame.to_ndarray(format="rgb24")
            resized = Image.fromarray(image).resize((224, 128), Image.Resampling.BOX)
            frames.append(np.asarray(resized))
    values = np.stack(frames).astype(np.float32)
    gray = values[..., 0] * 0.299 + values[..., 1] * 0.587 + values[..., 2] * 0.114
    temporal = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    means = gray.mean(axis=(1, 2))
    metadata.update(
        decoded_frames=len(frames),
        brightness_mean=float(gray.mean()),
        brightness_std=float(gray.std()),
        near_black_frame_ratio=float((means < 5).mean()),
        clipped_pixel_ratio=float(((values <= 1) | (values >= 254)).mean()),
        sharpness_laplacian_variance=float(
            statistics.fmean(
                (
                    -4 * frame[1:-1, 1:-1]
                    + frame[:-2, 1:-1]
                    + frame[2:, 1:-1]
                    + frame[1:-1, :-2]
                    + frame[1:-1, 2:]
                ).var()
                for frame in gray
            )
        ),
        adjacent_motion_mae=float(temporal.mean()),
        near_duplicate_frame_ratio=float((temporal < 0.5).mean()),
        luminance_flicker=float(np.abs(np.diff(means)).mean()),
    )
    return metadata, values


def _audio(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    chunks = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        rate = int(stream.rate)
        for frame in container.decode(audio=0):
            chunk = frame.to_ndarray()
            if chunk.ndim == 1:
                chunk = chunk[None]
            if np.issubdtype(chunk.dtype, np.integer):
                limit = max(abs(np.iinfo(chunk.dtype).min), np.iinfo(chunk.dtype).max)
                chunk = chunk.astype(np.float32) / limit
            chunks.append(chunk.astype(np.float32))
    waveform = np.concatenate(chunks, axis=1)
    rms = float(np.sqrt(np.mean(waveform**2)))
    window = max(1, rate // 50)
    trimmed = waveform[:, : waveform.shape[1] // window * window]
    window_rms = np.sqrt(np.mean(trimmed.reshape(-1, window) ** 2, axis=1))
    metadata = {
        "codec": stream.codec_context.name,
        "sample_rate": rate,
        "channels": waveform.shape[0],
        "samples": waveform.shape[1],
        "duration_seconds": waveform.shape[1] / rate,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
        "peak": float(np.abs(waveform).max()),
        "clipped_sample_ratio": float((np.abs(waveform) >= 0.999).mean()),
        "silence_window_ratio": float((window_rms < 10 ** (-50 / 20)).mean()),
    }
    return metadata, waveform


def _paired_video(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    count = min(len(left), len(right))
    delta = left[:count] - right[:count]
    mse = float(np.mean(delta**2))
    return {
        "rgb_mae": float(np.abs(delta).mean()),
        "rgb_psnr_db": float("inf") if mse == 0 else 10 * math.log10(255**2 / mse),
    }


def _paired_audio(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    channels = min(left.shape[0], right.shape[0])
    samples = min(left.shape[1], right.shape[1])
    left_flat = left[:channels, :samples].reshape(-1).astype(np.float64)
    right_flat = right[:channels, :samples].reshape(-1).astype(np.float64)
    correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
    return {
        "sample_mae": float(np.abs(left_flat - right_flat).mean()),
        "correlation": correlation,
    }


def _aggregate(records: list[dict[str, Any]], lane: str, medium: str) -> dict[str, Any]:
    excluded = {"codec"}
    keys = [
        key
        for key, value in records[0][lane][medium].items()
        if key not in excluded and isinstance(value, (int, float))
    ]
    return {
        key: {
            "mean": statistics.fmean(record[lane][medium][key] for record in records),
            "median": statistics.median(
                record[lane][medium][key] for record in records
            ),
        }
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = EvaluationSuite.load(args.suite)
    records = []
    for case in suite.cases:
        case_id = case.case_id
        official_path = _artifact(args.official_root, case_id)
        candidate_path = _artifact(args.candidate_root, case_id)
        official_video, official_frames = _video(official_path)
        candidate_video, candidate_frames = _video(candidate_path)
        official_audio, official_waveform = _audio(official_path)
        candidate_audio, candidate_waveform = _audio(candidate_path)
        records.append(
            {
                "case_id": case_id,
                "official": {"video": official_video, "audio": official_audio},
                "candidate": {"video": candidate_video, "audio": candidate_audio},
                "paired_diagnostics": {
                    "video": _paired_video(official_frames, candidate_frames),
                    "audio": _paired_audio(official_waveform, candidate_waveform),
                },
            }
        )
        print(f"[av-analysis] {case_id}", flush=True)
    result = {
        "schema_version": 1,
        "cases": len(records),
        "scope": "technical integrity and non-semantic signal diagnostics",
        "limitations": [
            "These statistics do not measure prompt adherence or subjective quality.",
            "Paired pixel/audio similarity is diagnostic only because diffusion output is not process-repeat bit deterministic.",
            "Human blind review remains required for semantics, aesthetics, and audio-video synchronization.",
        ],
        "aggregate": {
            lane: {
                medium: _aggregate(records, lane, medium)
                for medium in ("video", "audio")
            }
            for lane in ("official", "candidate")
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
