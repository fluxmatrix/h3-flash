import json
from pathlib import Path

import pytest

from h3_flash.evals import EvaluationSuite, EvaluationSuiteError

ROOT = Path(__file__).resolve().parents[1]


def test_broad40_loads_and_resolves_a_request() -> None:
    suite = EvaluationSuite.load(ROOT / "configs/evals/h3-broad40-v1.1.json")
    request = suite.request("people_mandarin_interview")
    assert suite.suite_id == "h3-broad40-v1.1"
    assert len(suite.cases) == 40
    assert (request.width, request.height, request.num_frames, request.fps) == (
        1344,
        768,
        124,
        24,
    )
    assert request.seed == 2026090201


def test_lossless_gate_suites_select_frozen_broad40_cases() -> None:
    broad40 = EvaluationSuite.load(ROOT / "configs/evals/h3-broad40-v1.1.json")
    gate4 = EvaluationSuite.load(ROOT / "configs/evals/h3-lossless-gate4-v1.json")
    gate8 = EvaluationSuite.load(ROOT / "configs/evals/h3-lossless-gate8-v1.json")

    assert len(gate4.cases) == 4
    assert len(gate8.cases) == 8
    assert {case.case_id for case in gate4.cases} < {
        case.case_id for case in gate8.cases
    }
    for gate in (gate4, gate8):
        assert (gate.width, gate.height, gate.num_frames, gate.fps) == (
            broad40.width,
            broad40.height,
            broad40.num_frames,
            broad40.fps,
        )
        for case in gate.cases:
            assert case == broad40.case(case.case_id)


def test_suite_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    value = {
        "schema_version": 1,
        "suite_id": "duplicate",
        "resolution": [32, 32],
        "num_frames": 5,
        "fps": 1,
        "cases": [
            {"case_id": "same", "prompt": "a", "seed": 1},
            {"case_id": "same", "prompt": "b", "seed": 2},
        ],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(EvaluationSuiteError, match="duplicate"):
        EvaluationSuite.load(path)


def test_derived_suite_can_override_geometry_without_copying_cases(
    tmp_path: Path,
) -> None:
    broad40 = ROOT / "configs/evals/h3-broad40-v1.1.json"
    value = {
        "schema_version": 1,
        "suite_id": "derived-geometry",
        "base_suite": str(broad40),
        "resolution": [832, 480],
        "num_frames": 243,
        "fps": 24,
        "case_ids": ["people_mandarin_interview"],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    suite = EvaluationSuite.load(path)

    assert (suite.width, suite.height, suite.num_frames, suite.fps) == (
        832,
        480,
        243,
        24,
    )
    assert suite.cases[0] == EvaluationSuite.load(broad40).cases[0]
