from pathlib import Path

import pytest

from h3_flash.benchmark import (
    active_worker_gpus,
    partition_case_ids,
    process_exit_status,
    suite_summary_is_complete,
)
from h3_flash.evals import EvaluationSuite

ROOT = Path(__file__).resolve().parents[1]


def test_broad40_partitions_into_eight_disjoint_workers() -> None:
    suite = EvaluationSuite.load(ROOT / "configs/evals/h3-broad40-v1.1.json")
    shards = [partition_case_ids(suite, index, 8) for index in range(8)]
    assert all(len(shard) == 5 for shard in shards)
    assert len({case_id for shard in shards for case_id in shard}) == 40


def test_partition_rejects_invalid_worker() -> None:
    suite = EvaluationSuite.load(ROOT / "configs/evals/h3-broad40-v1.1.json")
    with pytest.raises(ValueError, match="worker_index"):
        partition_case_ids(suite, 8, 8)


def test_worker_gpus_are_capped_to_independent_cases() -> None:
    suite = EvaluationSuite.load(ROOT / "configs/evals/h3-smoke1-768p-5s-v1.json")
    assert active_worker_gpus(suite, tuple(str(index) for index in range(8))) == ("0",)


@pytest.mark.parametrize(
    ("codes", "expected"),
    [([0, 0], 0), ([0, 2], 2), ([-9, 0], 137), ([0, -15], 143)],
)
def test_process_exit_status_never_hides_a_failure(
    codes: list[int], expected: int
) -> None:
    assert process_exit_status(codes) == expected


def test_resume_accepts_only_a_matching_complete_summary(tmp_path: Path) -> None:
    import json

    suite_path = ROOT / "configs/evals/h3-smoke1-768p-5s-v1.json"
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema_version": 1,
        "suite_id": "h3-smoke1-768p-5s-v1",
        "profile": "flash",
        "counts": {"ok": 1, "error": 0, "total": 1},
        "latency_seconds": {},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert suite_summary_is_complete(summary_path, suite_path, "flash")

    summary["counts"]["error"] = 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert not suite_summary_is_complete(summary_path, suite_path, "flash")
