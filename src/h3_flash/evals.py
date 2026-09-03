"""Load and validate frozen evaluation suites."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import GenerationRequest


class EvaluationSuiteError(ValueError):
    """Raised when an evaluation suite is malformed or a case is unknown."""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    prompt: str
    seed: int
    mode: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    suite_id: str
    width: int
    height: int
    num_frames: int
    fps: int
    cases: tuple[EvaluationCase, ...]

    @classmethod
    def load(cls, path: Path) -> EvaluationSuite:
        path = Path(path).expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise EvaluationSuiteError("evaluation schema_version must be 1")
        base_suite = data.get("base_suite")
        if base_suite is not None:
            if not isinstance(base_suite, str) or not base_suite:
                raise EvaluationSuiteError("base_suite must be a non-empty string")
            base_path = (path.parent / base_suite).resolve()
            if base_path == path:
                raise EvaluationSuiteError("evaluation suite cannot inherit itself")
            base = cls.load(base_path)
            selected = data.get("case_ids")
            if not isinstance(selected, list) or not selected:
                raise EvaluationSuiteError("derived suite case_ids must be non-empty")
            if not all(isinstance(case_id, str) and case_id for case_id in selected):
                raise EvaluationSuiteError("derived suite case_ids must be strings")
            if len(set(selected)) != len(selected):
                raise EvaluationSuiteError("derived suite case_ids must be unique")
            suite_id = data.get("suite_id")
            if not isinstance(suite_id, str) or not suite_id:
                raise EvaluationSuiteError("suite_id is required")
            resolution = data.get("resolution", [base.width, base.height])
            if not isinstance(resolution, list) or len(resolution) != 2:
                raise EvaluationSuiteError("resolution must be [width, height]")
            suite = cls(
                suite_id=suite_id,
                width=resolution[0],
                height=resolution[1],
                num_frames=data.get("num_frames", base.num_frames),
                fps=data.get("fps", base.fps),
                cases=tuple(base.case(case_id) for case_id in selected),
            )
            suite.request(suite.cases[0].case_id)
            return suite
        resolution = data.get("resolution")
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise EvaluationSuiteError("resolution must be [width, height]")
        rows = data.get("cases")
        if not isinstance(rows, list) or not rows:
            raise EvaluationSuiteError("cases must be a non-empty list")
        cases: list[EvaluationCase] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise EvaluationSuiteError(f"cases[{index}] must be an object")
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise EvaluationSuiteError(f"cases[{index}].case_id is required")
            if case_id in seen:
                raise EvaluationSuiteError(f"duplicate case_id: {case_id}")
            seen.add(case_id)
            case = EvaluationCase(
                case_id=case_id,
                prompt=row.get("prompt"),
                seed=row.get("seed"),
                mode=row.get("mode", "t2va"),
                metadata={
                    key: value
                    for key, value in row.items()
                    if key not in {"case_id", "prompt", "seed", "mode"}
                },
            )
            # Reuse the generation contract for prompt/seed/mode validation.
            GenerationRequest(
                prompt=case.prompt, seed=case.seed, mode=case.mode
            ).validate()
            cases.append(case)
        suite = cls(
            suite_id=data.get("suite_id"),
            width=resolution[0],
            height=resolution[1],
            num_frames=data.get("num_frames"),
            fps=data.get("fps"),
            cases=tuple(cases),
        )
        if not isinstance(suite.suite_id, str) or not suite.suite_id:
            raise EvaluationSuiteError("suite_id is required")
        # Validate suite-wide generation geometry with one already-valid case.
        suite.request(cases[0].case_id)
        return suite

    def case(self, case_id: str) -> EvaluationCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise EvaluationSuiteError(f"unknown case_id: {case_id}")

    def request(self, case_id: str, **overrides: int) -> GenerationRequest:
        case = self.case(case_id)
        request = GenerationRequest(
            prompt=case.prompt,
            seed=overrides.get("seed", case.seed),
            width=overrides.get("width", self.width),
            height=overrides.get("height", self.height),
            num_frames=overrides.get("num_frames", self.num_frames),
            fps=overrides.get("fps", self.fps),
            mode=case.mode,
        )
        request.validate()
        return request
