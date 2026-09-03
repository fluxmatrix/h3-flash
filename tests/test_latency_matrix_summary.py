import json
from pathlib import Path

from tools.reporting.summarize_latency_matrix import _peak_allocated_gib, _suite_name

ROOT = Path(__file__).resolve().parents[1]


def test_all_latency_matrix_suite_names_exist() -> None:
    names = {
        _suite_name(resolution, duration, orientation)
        for resolution in (480, 768)
        for duration in (5, 10, 15)
        for orientation in ("landscape", "portrait")
    }

    assert len(names) == 12
    assert all((ROOT / "configs/evals" / f"{name}.json").is_file() for name in names)


def test_peak_allocated_gib_uses_maximum_case_and_rank(tmp_path: Path) -> None:
    for case_id, peaks in (("a", [1024**3, 2 * 1024**3]), ("b", [3 * 1024**3])):
        case_root = tmp_path / "cases" / case_id
        case_root.mkdir(parents=True)
        (case_root / "result.json").write_text(
            json.dumps({"peak_gpu_memory_bytes": peaks}), encoding="utf-8"
        )

    assert _peak_allocated_gib(tmp_path, ("a", "b")) == 3.0
    assert _peak_allocated_gib(tmp_path, ("a", "missing")) is None
