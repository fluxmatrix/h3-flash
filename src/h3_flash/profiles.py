"""Load, inherit, validate, and fingerprint H3-Flash profiles."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sysconfig
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ProfileError(ValueError):
    """Raised when a profile repository violates the public contract."""


def canonical_json(value: Any) -> bytes:
    """Return the one canonical encoding used for all H3-Flash digests."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _deep_merge(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    if child.get("_replace") is True:
        return copy.deepcopy(
            {key: value for key, value in child.items() if key != "_replace"}
        )
    result = copy.deepcopy(dict(parent))
    for key, value in child.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def discover_profiles_dir(explicit: Path | None = None) -> Path:
    """Find the canonical profile directory in a checkout or installation."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_root = os.environ.get("H3_FLASH_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "profiles")
    candidates.extend(
        [
            Path.cwd() / "profiles",
            Path(__file__).resolve().parents[2] / "profiles",
            Path(sysconfig.get_path("data")) / "h3_flash" / "profiles",
        ]
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.toml")):
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise ProfileError(f"profile directory not found; checked: {rendered}")


class ProfileRepository:
    """A directory of immutable, inheritable TOML profile definitions."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = discover_profiles_dir(directory)
        self._raw: dict[str, dict[str, Any]] = {}
        self._resolved: dict[str, dict[str, Any]] = {}
        for path in sorted(self.directory.glob("*.toml")):
            with path.open("rb") as handle:
                profile = tomllib.load(handle)
            profile_id = profile.get("id")
            if not isinstance(profile_id, str) or not profile_id:
                raise ProfileError(f"{path}: non-empty string id is required")
            if profile_id in self._raw:
                raise ProfileError(f"duplicate profile id: {profile_id}")
            if profile_id != path.stem:
                raise ProfileError(
                    f"{path}: profile id {profile_id!r} must match filename"
                )
            self._raw[profile_id] = profile
        if not self._raw:
            raise ProfileError(f"no TOML profiles found in {self.directory}")

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._raw))

    def resolve(self, name: str) -> dict[str, Any]:
        return copy.deepcopy(self._resolve(name, stack=()))

    def digest(self, name: str) -> str:
        return sha256_json(self.resolve(name))

    def _resolve(self, name: str, stack: tuple[str, ...]) -> dict[str, Any]:
        if name in self._resolved:
            return self._resolved[name]
        if name not in self._raw:
            raise ProfileError(f"unknown profile: {name}")
        if name in stack:
            cycle = " -> ".join((*stack, name))
            raise ProfileError(f"profile inheritance cycle: {cycle}")

        raw = self._raw[name]
        parent_name = raw.get("parent")
        if parent_name is None:
            resolved = copy.deepcopy(raw)
        elif not isinstance(parent_name, str):
            raise ProfileError(f"{name}: parent must be a string")
        else:
            parent = self._resolve(parent_name, (*stack, name))
            resolved = _deep_merge(parent, raw)

        self._validate(name, resolved)
        self._resolved[name] = resolved
        return resolved

    @staticmethod
    def _validate(name: str, profile: Mapping[str, Any]) -> None:
        if profile.get("schema_version") != 1:
            raise ProfileError(f"{name}: schema_version must be 1")
        quality_class = profile.get("quality_class")
        allowed = {
            "reference",
            "bit_exact",
            "numerically_equivalent",
            "approximate",
        }
        if quality_class not in allowed:
            raise ProfileError(f"{name}: invalid quality_class {quality_class!r}")

        sampling = profile.get("sampling")
        attention = profile.get("attention")
        model = profile.get("model")
        if not all(isinstance(item, Mapping) for item in (sampling, attention, model)):
            raise ProfileError(
                f"{name}: resolved model, sampling, and attention are required"
            )
        api_steps = sampling.get("api_num_inference_steps")
        evaluations = sampling.get("model_evaluations")
        for field, value in (
            ("api_num_inference_steps", api_steps),
            ("model_evaluations", evaluations),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ProfileError(
                    f"{name}: sampling.{field} must be a positive integer"
                )

        if quality_class in {"reference", "bit_exact", "numerically_equivalent"}:
            if model.get("turbo_lora") is not False:
                raise ProfileError(
                    f"{name}: non-approximate profiles cannot use Turbo LoRA"
                )
            if attention.get("semantics") != "dense":
                raise ProfileError(
                    f"{name}: non-approximate profiles require dense attention"
                )
            if api_steps != 50 or evaluations != 49:
                raise ProfileError(
                    f"{name}: non-approximate profiles require the official "
                    "50-point sigma grid and 49 model evaluations"
                )
            if (
                sampling.get("count_semantics")
                != "sigma_grid_points_including_terminal_zero"
            ):
                raise ProfileError(f"{name}: invalid official step-count semantics")
