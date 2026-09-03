"""Machine-readable optimization taxonomy and evidence metadata."""

from __future__ import annotations

import copy
import os
import sysconfig
import tomllib
from pathlib import Path
from typing import Any

from .profiles import sha256_json


class MechanismError(ValueError):
    """Raised when the optimization mechanism registry is invalid."""


def discover_mechanisms_file(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_root = os.environ.get("H3_FLASH_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "configs" / "mechanisms.toml")
    candidates.extend(
        [
            Path.cwd() / "configs" / "mechanisms.toml",
            Path(__file__).resolve().parents[2] / "configs" / "mechanisms.toml",
            Path(sysconfig.get_path("data"))
            / "h3_flash"
            / "configs"
            / "mechanisms.toml",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise MechanismError(f"mechanism registry not found; checked: {rendered}")


class MechanismRegistry:
    """Load and validate public lanes and internal verification labels."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = discover_mechanisms_file(path)
        with self.path.open("rb") as handle:
            document = tomllib.load(handle)
        self._validate_document(document)
        self._document = document
        self._by_id = {entry["id"]: entry for entry in document["mechanisms"]}

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_id))

    def get(self, mechanism_id: str) -> dict[str, Any]:
        try:
            return copy.deepcopy(self._by_id[mechanism_id])
        except KeyError as error:
            raise MechanismError(f"unknown mechanism: {mechanism_id}") from error

    def entries(self, *, public_lane: str | None = None) -> list[dict[str, Any]]:
        entries = [self.get(name) for name in self.names()]
        if public_lane is not None:
            if public_lane not in {"lossless", "flash"}:
                raise MechanismError(f"unknown public lane: {public_lane}")
            entries = [
                entry for entry in entries if entry["public_lane"] == public_lane
            ]
        return entries

    def digest(self) -> str:
        return sha256_json(self._document)

    @staticmethod
    def _validate_document(document: dict[str, Any]) -> None:
        if document.get("schema_version") != 1:
            raise MechanismError("schema_version must be 1")
        threshold = document.get("headline_minimum_median_reduction_percent")
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or threshold <= 0
        ):
            raise MechanismError("headline threshold must be positive")
        if document.get("headline_must_exceed_run_noise") is not True:
            raise MechanismError(
                "headline results must be required to exceed run noise"
            )
        entries = document.get("mechanisms")
        if not isinstance(entries, list) or not entries:
            raise MechanismError("mechanisms must be a non-empty array")

        seen: set[str] = set()
        allowed_statuses = {
            "historical_measured_pending_port",
            "historical_unisolated_pending_port",
            "planned_optional",
            "rejected_experiment",
            "implemented_baseline_behavior",
            "implemented_bit_exact_gate_passed",
            "implemented_broad40_measured",
            "implemented_gate4_measured",
            "implemented_in_official_backend",
            "implemented_measured_negative_excluded",
        }
        validation_profiles = {
            "bit_exact": "exact",
            "mathematically_equivalent": "equivalent",
            "generated_tensor_exact": "output",
            "approximate": "fast",
        }
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise MechanismError(f"mechanisms[{index}] must be a table")
            mechanism_id = entry.get("id")
            if not isinstance(mechanism_id, str) or not mechanism_id:
                raise MechanismError(
                    f"mechanisms[{index}].id must be a non-empty string"
                )
            if mechanism_id in seen:
                raise MechanismError(f"duplicate mechanism id: {mechanism_id}")
            seen.add(mechanism_id)

            verification = entry.get("verification")
            if verification not in validation_profiles:
                raise MechanismError(f"{mechanism_id}: invalid verification label")
            public_lane = entry.get("public_lane")
            expected_lane = "flash" if verification == "approximate" else "lossless"
            if public_lane != expected_lane:
                raise MechanismError(
                    f"{mechanism_id}: {verification} mechanisms belong to {expected_lane}"
                )
            expected_profile = validation_profiles[verification]
            if entry.get("validation_profile") != expected_profile:
                raise MechanismError(
                    f"{mechanism_id}: validation_profile must be {expected_profile}"
                )
            if entry.get("status") not in allowed_statuses:
                raise MechanismError(f"{mechanism_id}: invalid status")
            for field in ("title", "origin", "timing_scope", "evidence_note"):
                if not isinstance(entry.get(field), str) or not entry[field]:
                    raise MechanismError(f"{mechanism_id}: {field} must be non-empty")
            for field in ("changes", "gates"):
                value = entry.get(field)
                if (
                    not isinstance(value, list)
                    or not value
                    or not all(isinstance(item, str) and item for item in value)
                ):
                    raise MechanismError(
                        f"{mechanism_id}: {field} must be non-empty strings"
                    )
            if not isinstance(entry.get("headline_candidate"), bool):
                raise MechanismError(
                    f"{mechanism_id}: headline_candidate must be boolean"
                )
