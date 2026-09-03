from pathlib import Path

import pytest

from h3_flash.locks import LockRepository
from h3_flash.manifest import GenerationRequest, ManifestError, build_manifest
from h3_flash.profiles import ProfileRepository

ROOT = Path(__file__).resolve().parents[1]


def request(seed: int = 42) -> GenerationRequest:
    return GenerationRequest(
        prompt=(
            "integrated_multimodal_description: A brass pendulum swings once.\n\n"
            "overall_soundscape: One synchronized mechanical tick.\n\n"
            "non_diegetic_music: N/A"
        ),
        seed=seed,
    )


def test_manifest_digest_is_deterministic_and_sensitive() -> None:
    repository = ProfileRepository(ROOT / "profiles")
    locks = LockRepository(ROOT / "locks")
    first = build_manifest(
        repository, "official", request(), backend="ai_toolkit", locks=locks
    )
    second = build_manifest(
        repository, "official", request(), backend="ai_toolkit", locks=locks
    )
    changed = build_manifest(
        repository, "official", request(43), backend="ai_toolkit", locks=locks
    )

    assert first == second
    assert first["manifest_sha256"] != changed["manifest_sha256"]
    assert first["profile_sha256"] == repository.digest("official")
    assert first["locks"]["models"]["name"] == "models.official"


def test_runtime_source_is_content_addressed() -> None:
    repository = ProfileRepository(ROOT / "profiles")
    locks = LockRepository(ROOT / "locks")
    first = build_manifest(
        repository,
        "official",
        request(),
        backend="official-diffusers",
        locks=locks,
        runtime_source={"diffusers": {"git_commit": "a" * 40}},
    )
    changed = build_manifest(
        repository,
        "official",
        request(),
        backend="official-diffusers",
        locks=locks,
        runtime_source={"diffusers": {"git_commit": "b" * 40}},
    )

    assert first["manifest_sha256"] != changed["manifest_sha256"]
    assert first["configuration_sha256"] == changed["configuration_sha256"]


def test_execution_overrides_are_content_addressed() -> None:
    repository = ProfileRepository(ROOT / "profiles")
    locks = LockRepository(ROOT / "locks")
    first = build_manifest(
        repository,
        "official",
        request(),
        backend="official-diffusers",
        locks=locks,
        execution={"artifact_format": "mp4", "generation_device": "cuda:0"},
    )
    changed = build_manifest(
        repository,
        "official",
        request(),
        backend="official-diffusers",
        locks=locks,
        execution={"artifact_format": "latent", "generation_device": "cuda:0"},
    )

    assert first["configuration_sha256"] != changed["configuration_sha256"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"prompt": ""}, "prompt"),
        ({"width": 1300}, "divisible by 32"),
        ({"height": 700}, "divisible by 32"),
        ({"num_frames": 4}, "at least 5"),
        ({"fps": 0}, "positive"),
        ({"seed": -1}, "seed"),
        ({"mode": "unsupported"}, "t2va or i2va"),
        ({"prompt": None}, "prompt"),
        ({"seed": "42"}, "seed must be an integer"),
        ({"width": True}, "width must be an integer"),
        ({"mode": None}, "mode must be a string"),
    ],
)
def test_invalid_requests_are_rejected(changes: dict, message: str) -> None:
    values = {
        "prompt": request().prompt,
        "seed": 42,
        "width": 1344,
        "height": 768,
        "num_frames": 124,
        "fps": 24,
        "mode": "t2va",
    }
    values.update(changes)
    with pytest.raises(ManifestError, match=message):
        GenerationRequest(**values).validate()
