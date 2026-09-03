#!/usr/bin/env python3
"""Compare two completed official-backend suites case by case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "standard_deviation": statistics.pstdev(values),
    }


def _case_result(root: Path, case_id: str) -> tuple[dict[str, Any], Path]:
    case_root = root / "cases" / case_id
    result = _read(case_root / "result.json")
    artifact = case_root / result["outputs"]["path"]
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    actual = _sha256(artifact)
    recorded = result["outputs"]["sha256"]
    if actual != recorded:
        raise ValueError(f"{artifact}: recorded SHA256 does not match the file")
    return result, artifact


def compare(
    suite_path: Path, baseline_root: Path, candidate_root: Path
) -> dict[str, Any]:
    suite = _read(suite_path)
    baseline_summary = _read(baseline_root / "summary.json")
    candidate_summary = _read(candidate_root / "summary.json")
    records = []
    timings: dict[str, dict[str, list[float]]] = {}
    artifact_matches = 0

    for case in suite["cases"]:
        case_id = case["case_id"]
        baseline, baseline_artifact = _case_result(baseline_root, case_id)
        candidate, candidate_artifact = _case_result(candidate_root, case_id)
        if baseline["request"] != candidate["request"]:
            raise ValueError(f"{case_id}: generation requests differ")
        if baseline["sampling"] != candidate["sampling"]:
            raise ValueError(f"{case_id}: sampling contracts differ")

        artifact_equal = baseline["outputs"]["sha256"] == candidate["outputs"]["sha256"]
        artifact_matches += int(artifact_equal)
        common_timers = sorted(
            set(baseline["timing_seconds"]) & set(candidate["timing_seconds"])
        )
        paired = {}
        for timer in common_timers:
            baseline_value = float(baseline["timing_seconds"][timer])
            candidate_value = float(candidate["timing_seconds"][timer])
            reduction = baseline_value - candidate_value
            timings.setdefault(
                timer, {"baseline": [], "candidate": [], "reduction": []}
            )
            timings[timer]["baseline"].append(baseline_value)
            timings[timer]["candidate"].append(candidate_value)
            timings[timer]["reduction"].append(reduction)
            paired[timer] = {
                "baseline_seconds": baseline_value,
                "candidate_seconds": candidate_value,
                "reduction_seconds": reduction,
                "reduction_percent": (
                    100 * reduction / baseline_value if baseline_value else None
                ),
            }
        records.append(
            {
                "case_id": case_id,
                "artifact_sha256_equal": artifact_equal,
                "baseline_artifact": str(baseline_artifact),
                "candidate_artifact": str(candidate_artifact),
                "candidate_host_transfer": candidate["outputs"].get("host_transfer"),
                "candidate_host_transfer_pool_reused": candidate["outputs"].get(
                    "host_transfer_pool_reused"
                ),
                "timing": paired,
            }
        )

    timing_summary = {}
    for name, samples in timings.items():
        baseline_distribution = _distribution(samples["baseline"])
        candidate_distribution = _distribution(samples["candidate"])
        reduction_distribution = _distribution(samples["reduction"])
        timing_summary[name] = {
            "baseline_seconds": baseline_distribution,
            "candidate_seconds": candidate_distribution,
            "paired_reduction_seconds": reduction_distribution,
            "median_reduction_percent": (
                100
                * (baseline_distribution["median"] - candidate_distribution["median"])
                / baseline_distribution["median"]
                if baseline_distribution["median"]
                else None
            ),
        }

    reuse_groups = {}
    for reused in (False, True):
        selected = [
            record
            for record in records
            if record["candidate_host_transfer_pool_reused"] is reused
        ]
        if selected:
            reuse_groups[str(reused).lower()] = {
                "count": len(selected),
                "candidate_device_to_host_seconds": _distribution(
                    [
                        record["timing"]["device_to_host"]["candidate_seconds"]
                        for record in selected
                    ]
                ),
            }

    return {
        "schema_version": 1,
        "suite_id": suite["suite_id"],
        "case_count": len(records),
        "baseline": {
            "root": str(baseline_root.resolve()),
            "profile": baseline_summary["profile"],
        },
        "candidate": {
            "root": str(candidate_root.resolve()),
            "profile": candidate_summary["profile"],
        },
        "contract": {
            "requests_equal": True,
            "sampling_equal": True,
            "actual_artifacts_sha256_verified": True,
        },
        "artifact_sha256": {
            "equal": artifact_matches,
            "different": len(records) - artifact_matches,
            "pass_rate": artifact_matches / len(records),
        },
        "timing_seconds": timing_summary,
        "candidate_host_transfer_pool_reuse": reuse_groups,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.suite, args.baseline_root, args.candidate_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
