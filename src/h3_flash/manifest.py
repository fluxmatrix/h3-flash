"""Build immutable, content-addressed H3-Flash run manifests."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

from .locks import LockRepository
from .profiles import ProfileRepository, sha256_json


class ManifestError(ValueError):
    """Raised when a generation request cannot form a valid manifest."""


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    seed: int
    width: int = 1344
    height: int = 768
    num_frames: int = 124
    fps: int = 24
    mode: str = "t2va"

    def validate(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ManifestError("prompt must not be empty")
        if not isinstance(self.mode, str):
            raise ManifestError("mode must be a string")
        if self.mode not in {"t2va", "i2va"}:
            raise ManifestError("mode must be t2va or i2va")
        integer_fields = {
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "fps": self.fps,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ManifestError(f"{name} must be an integer")
        if self.width < 32 or self.height < 32:
            raise ManifestError("width and height must be at least 32")
        if self.width % 32 or self.height % 32:
            raise ManifestError("width and height must be divisible by 32")
        if self.num_frames < 5:
            raise ManifestError("num_frames must be at least 5")
        if self.fps < 1:
            raise ManifestError("fps must be positive")
        if self.seed < 0 or self.seed > (2**63 - 1):
            raise ManifestError("seed must be in [0, 2^63 - 1]")


def build_manifest(
    repository: ProfileRepository,
    profile_name: str,
    request: GenerationRequest,
    *,
    backend: str,
    locks: LockRepository,
    runtime_source: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a complete manifest and attach its canonical content digest."""

    request.validate()
    if not backend or any(character.isspace() for character in backend):
        raise ManifestError("backend must be a non-empty identifier without whitespace")
    profile = repository.resolve(profile_name)
    provenance = profile.get("provenance", {})
    upstream_lock = provenance.get("upstreams_lock")
    model_lock = provenance.get("model_lock")
    if not isinstance(upstream_lock, str) or not isinstance(model_lock, str):
        raise ManifestError(f"profile {profile_name!r} has incomplete provenance locks")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "backend": backend,
        "locks": {
            "models": locks.reference(model_lock),
            "upstreams": locks.reference(upstream_lock),
        },
        "profile": profile,
        "profile_sha256": sha256_json(profile),
        "request": asdict(request),
    }
    if execution is not None:
        payload["execution"] = copy.deepcopy(execution)
    # This digest is stable across machines and records every user-visible
    # execution override. Runtime paths and imported module locations are
    # deliberately excluded.
    payload["configuration_sha256"] = sha256_json(payload)
    if runtime_source is not None:
        payload["runtime_source"] = copy.deepcopy(runtime_source)
    payload["manifest_sha256"] = sha256_json(payload)
    return payload
