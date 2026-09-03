import hashlib
import json
from pathlib import Path

import pytest


def _write_case(
    root: Path, case_id: str, payload: bytes, d2h: float, reused: bool
) -> None:
    case_root = root / "cases" / case_id
    case_root.mkdir(parents=True)
    artifact = case_root / "output.mp4"
    artifact.write_bytes(payload)
    result = {
        "request": {"prompt": case_id, "seed": 1},
        "sampling": {"model_evaluations": 49},
        "timing_seconds": {"device_to_host": d2h, "request_process": 10 + d2h},
        "outputs": {
            "path": artifact.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "host_transfer": "persistent_pinned",
            "host_transfer_pool_reused": reused,
        },
    }
    (case_root / "result.json").write_text(json.dumps(result), encoding="utf-8")


def test_compare_verifies_artifacts_and_paired_timings(tmp_path: Path) -> None:
    import importlib.util

    script = (
        Path(__file__).resolve().parents[1] / "tools/reporting/compare_official_runs.py"
    )
    spec = importlib.util.spec_from_file_location("compare_official_runs", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps({"suite_id": "tiny", "cases": [{"case_id": "a"}, {"case_id": "b"}]}),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for root, profile in ((baseline, "official"), (candidate, "lossless")):
        root.mkdir()
        (root / "summary.json").write_text(
            json.dumps({"profile": profile}), encoding="utf-8"
        )
    _write_case(baseline, "a", b"same-a", 1.0, False)
    _write_case(candidate, "a", b"same-a", 0.8, False)
    _write_case(baseline, "b", b"same-b", 1.0, False)
    _write_case(candidate, "b", b"same-b", 0.2, True)

    report = module.compare(suite, baseline, candidate)
    assert report["artifact_sha256"] == {"equal": 2, "different": 0, "pass_rate": 1.0}
    assert (
        report["timing_seconds"]["device_to_host"]["baseline_seconds"]["median"] == 1.0
    )
    assert (
        report["timing_seconds"]["device_to_host"]["candidate_seconds"]["median"] == 0.5
    )
    assert report["candidate_host_transfer_pool_reuse"]["true"]["count"] == 1


def test_compare_rejects_recorded_artifact_hash_mismatch(tmp_path: Path) -> None:
    import importlib.util

    script = (
        Path(__file__).resolve().parents[1] / "tools/reporting/compare_official_runs.py"
    )
    spec = importlib.util.spec_from_file_location("compare_official_runs_bad", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps({"suite_id": "tiny", "cases": [{"case_id": "a"}]}), encoding="utf-8"
    )
    for name in ("baseline", "candidate"):
        root = tmp_path / name
        root.mkdir()
        (root / "summary.json").write_text(
            json.dumps({"profile": name}), encoding="utf-8"
        )
        _write_case(root, "a", b"payload", 1.0, False)
    result_path = tmp_path / "candidate" / "cases" / "a" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["outputs"]["sha256"] = "0" * 64
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="recorded SHA256"):
        module.compare(suite, tmp_path / "baseline", tmp_path / "candidate")
