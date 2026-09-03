#!/usr/bin/env python3
"""Merge OFFICIAL/LOSSLESS/FLASH latency suites into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from h3_flash.evals import EvaluationSuite


VERSIONS = ("OFFICIAL", "LOSSLESS", "FLASH")
RESOLUTIONS = (480, 768)
DURATIONS = (5, 10, 15)
ORIENTATIONS = ("landscape", "portrait")


def _suite_name(resolution: int, duration: int, orientation: str) -> str:
    suffix = "-portrait" if orientation == "portrait" else ""
    return f"h3-latency4-{resolution}p{suffix}-{duration}s-v1"


def _round_distribution(data: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 6) for key, value in data.items()}


def _peak_allocated_gib(suite_root: Path, case_ids: tuple[str, ...]) -> float | None:
    peaks = []
    for case_id in case_ids:
        result_path = suite_root / "cases" / case_id / "result.json"
        if not result_path.is_file():
            return None
        result = json.loads(result_path.read_text(encoding="utf-8"))
        peaks.extend(int(value) for value in result.get("peak_gpu_memory_bytes", ()))
    return round(max(peaks) / (1024**3), 6) if peaks else None


def collect(
    roots: dict[str, Path], suites_dir: Path, *, require_complete: bool
) -> dict[str, Any]:
    records = []
    missing = []
    for resolution in RESOLUTIONS:
        for duration in DURATIONS:
            for orientation in ORIENTATIONS:
                suite_name = _suite_name(resolution, duration, orientation)
                suite = EvaluationSuite.load(suites_dir / f"{suite_name}.json")
                for version in VERSIONS:
                    summary_path = roots[version] / suite_name / "summary.json"
                    if not summary_path.is_file():
                        missing.append(str(summary_path))
                        continue
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary.get("counts", {}).get("ok") != 4:
                        missing.append(f"{summary_path} (not 4/4 successful)")
                        continue
                    allocation = summary["allocation"]
                    records.append(
                        {
                            "version": version,
                            "suite_id": suite.suite_id,
                            "orientation": orientation,
                            "nominal_resolution": f"{resolution}p",
                            "width": suite.width,
                            "height": suite.height,
                            "nominal_duration_seconds": duration,
                            "num_frames": suite.num_frames,
                            "fps": suite.fps,
                            "encoded_duration_seconds": round(
                                suite.num_frames / suite.fps, 6
                            ),
                            "cases": 4,
                            "gpu_per_request": allocation["gpu_per_request"],
                            "peak_allocated_gib_per_rank": _peak_allocated_gib(
                                roots[version] / suite_name,
                                tuple(case.case_id for case in suite.cases),
                            ),
                            "profile": summary["profile"],
                            "generation_seconds": _round_distribution(
                                summary["latency_seconds"]["generation"]
                            ),
                            "e2e_seconds": _round_distribution(
                                summary["latency_seconds"]["request_process"]
                            ),
                            "suite_wall_seconds": round(
                                summary["suite_wall_seconds"], 6
                            ),
                            "summary_path": str(summary_path.resolve()),
                        }
                    )
    if require_complete and missing:
        raise SystemExit("latency matrix is incomplete:\n" + "\n".join(missing))
    return {
        "schema_version": 1,
        "protocol": {
            "versions": list(VERSIONS),
            "cases_per_cell": 4,
            "fps": 24,
            "frame_counts": {"5s": 124, "10s": 243, "15s": 345},
            "gpu_per_request": {"OFFICIAL": 1, "LOSSLESS": 8, "FLASH": 8},
            "matrix_concurrency": {
                "OFFICIAL": "8 independent one-GPU workers; landscape and portrait suites run concurrently",
                "LOSSLESS": "one SP8 request stream; cells run sequentially",
                "FLASH": "one SP8 request stream; cells run sequentially",
            },
            "artifact_writer": {
                "OFFICIAL": "reference Diffusers/PyAV",
                "LOSSLESS": "pinned FFmpeg raw pipe, full audio",
                "FLASH": "pinned FFmpeg raw pipe, full audio",
            },
            "timing": "resident request; model load and startup warm-up excluded",
        },
        "counts": {
            "complete_cells": len(records),
            "expected_cells": 36,
            "missing_cells": len(missing),
        },
        "roots": {key: str(value.resolve()) for key, value in roots.items()},
        "missing": missing,
        "records": records,
    }


def _seconds(record: dict[str, Any] | None, key: str) -> str:
    if record is None:
        return "pending"
    return f"{record[key]['median']:.3f}"


def _ratio(numerator: dict[str, Any] | None, denominator: dict[str, Any] | None) -> str:
    if numerator is None or denominator is None:
        return "pending"
    value = numerator["e2e_seconds"]["median"] / denominator["e2e_seconds"]["median"]
    return f"{value:.2f}x"


def render_markdown(result: dict[str, Any]) -> str:
    by_key = {
        (
            row["nominal_resolution"],
            row["nominal_duration_seconds"],
            row["orientation"],
            row["version"],
        ): row
        for row in result["records"]
    }
    lines = [
        "# OFFICIAL / LOSSLESS / FLASH latency matrix",
        "",
        "> Generated by `python -m tools.reporting.summarize_latency_matrix`; do not edit measured values manually.",
        "",
        "Each cell contains four fixed Broad40 prompts/seeds at 24 fps. OFFICIAL uses",
        "1×B200 per request; LOSSLESS and FLASH use 8×B200 per request. Values below",
        "are resident-request medians in seconds and exclude model load/startup warm-up.",
        "The 15-second protocol is 345 frames (14.375 encoded seconds), matching the",
        "existing H3 evaluation protocol rather than silently rounding above 15 seconds.",
        "OFFICIAL's eight workers only shorten suite collection time: each OFFICIAL",
        "request still uses one B200. LOSSLESS/FLASH latency is one SP8 request using",
        "all eight B200s, so the cross-lane ratios are latency ratios, not equal-cost",
        "hardware-normalized throughput comparisons.",
        "LOSSLESS and FLASH share the pinned full-audio raw-pipe FFmpeg path; OFFICIAL",
        "retains the reference Diffusers/PyAV artifact writer. This affects E2E but",
        "not the model-side generation timer and is intentionally disclosed.",
        "",
        "## Median generation / E2E",
        "",
        "| Orientation | Resolution | Nominal duration | Frames | OFFICIAL | LOSSLESS | FLASH |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for orientation in ORIENTATIONS:
        for resolution in RESOLUTIONS:
            for duration in DURATIONS:
                rows = {
                    version: by_key.get(
                        (f"{resolution}p", duration, orientation, version)
                    )
                    for version in VERSIONS
                }
                sample = next((row for row in rows.values() if row is not None), None)
                frames = (
                    sample["num_frames"]
                    if sample
                    else {5: 124, 10: 243, 15: 345}[duration]
                )
                values = [
                    f"{_seconds(rows[version], 'generation_seconds')} / "
                    f"{_seconds(rows[version], 'e2e_seconds')}"
                    for version in VERSIONS
                ]
                lines.append(
                    f"| {orientation} | {resolution}p | {duration}s | {frames} | "
                    + " | ".join(values)
                    + " |"
                )
    lines.extend(
        [
            "",
            "## E2E latency ratios",
            "",
            "These are raw resident-request latency ratios. OFFICIAL-to-SP8 columns",
            "change per-request hardware from one to eight B200s. FLASH/LOSSLESS",
            "keeps the same eight-B200 topology and the same output path.",
            "",
            "| Orientation | Resolution | Nominal duration | OFFICIAL / LOSSLESS | OFFICIAL / FLASH | LOSSLESS / FLASH |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for orientation in ORIENTATIONS:
        for resolution in RESOLUTIONS:
            for duration in DURATIONS:
                rows = {
                    version: by_key.get(
                        (f"{resolution}p", duration, orientation, version)
                    )
                    for version in VERSIONS
                }
                lines.append(
                    f"| {orientation} | {resolution}p | {duration}s | "
                    f"{_ratio(rows['OFFICIAL'], rows['LOSSLESS'])} | "
                    f"{_ratio(rows['OFFICIAL'], rows['FLASH'])} | "
                    f"{_ratio(rows['LOSSLESS'], rows['FLASH'])} |"
                )
    lines.extend(
        [
            "",
            "## Reproduction entry point",
            "",
            "```bash",
            "scripts/run_latency_matrix.sh OFFICIAL artifacts/matrix-official",
            "scripts/run_latency_matrix.sh LOSSLESS artifacts/matrix-lossless",
            "scripts/run_latency_matrix.sh FLASH artifacts/matrix-flash",
            "```",
            "",
            "The convenience wrapper resolves the quickstart's repository-local paths",
            "and invokes the fully explicit lower-level command:",
            "",
            "```bash",
            "scripts/benchmark_latency_matrix.sh \\",
            "  VERSION PYTHON DIFFUSERS_CHECKOUT MODEL_ROOT OUTPUT_ROOT \\",
            "  0,1,2,3,4,5,6,7 FFMPEG_BIN",
            "```",
            "",
            "Run it once for each of `OFFICIAL`, `LOSSLESS`, and `FLASH`. The wrapper",
            "records the fully expanded child command, GPU mapping, profile, source",
            "provenance, prompt/seed, per-stage timing, and artifact SHA-256 in each",
            "suite's `launch.json`, `summary.json`, `manifest.json`, and `result.json`.",
            "The exact evidence roots used for this generated table are:",
            "",
        ]
    )
    for version in VERSIONS:
        lines.append(f"- `{version}`: `{result['roots'][version]}`")
    lines.extend(
        [
            "",
            "## Full distributions",
            "",
            "| Version | Orientation | Shape | Nominal/encoded | GPUs/request | Peak allocated GiB/rank | Generation mean/median/P90 | E2E mean/median/P90 | E2E range |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    order = {version: index for index, version in enumerate(VERSIONS)}
    for row in sorted(
        result["records"],
        key=lambda item: (
            item["orientation"],
            int(item["nominal_resolution"][:-1]),
            item["nominal_duration_seconds"],
            order[item["version"]],
        ),
    ):
        generation = row["generation_seconds"]
        e2e = row["e2e_seconds"]
        peak = row["peak_allocated_gib_per_rank"]
        peak_text = f"{peak:.2f}" if peak is not None else "pending"
        lines.append(
            f"| {row['version']} | {row['orientation']} | "
            f"{row['width']}×{row['height']} | "
            f"{row['nominal_duration_seconds']}s/{row['encoded_duration_seconds']:.3f}s | "
            f"{row['gpu_per_request']} | "
            f"{peak_text} | "
            f"{generation['mean']:.3f}/{generation['median']:.3f}/{generation['p90']:.3f} | "
            f"{e2e['mean']:.3f}/{e2e['median']:.3f}/{e2e['p90']:.3f} | "
            f"{e2e['minimum']:.3f}–{e2e['maximum']:.3f} |"
        )
    if result["missing"]:
        lines.extend(
            [
                "",
                f"> Matrix still running: {len(result['missing'])} of 36 cells are pending.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    for version in VERSIONS:
        parser.add_argument(f"--{version.lower()}-root", type=Path, required=True)
    parser.add_argument(
        "--suites-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/evals",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    roots = {version: getattr(args, f"{version.lower()}_root") for version in VERSIONS}
    result = collect(roots, args.suites_dir, require_complete=args.require_complete)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
