from pathlib import Path

import pytest

from h3_flash.locks import LockError, LockRepository

ROOT = Path(__file__).resolve().parents[1]


def test_model_locks_have_consistent_sizes_and_digests() -> None:
    repository = LockRepository(ROOT / "locks")
    for name in (
        "models.fast-turbo4-bf16-inputs",
        "models.official",
        "models.prompt-enhancer",
        "upstreams",
    ):
        assert len(repository.digest(name)) == 64
        assert repository.reference(name)["name"] == name

    turbo = repository.reference("models.fast-turbo4-bf16-inputs")
    assert turbo["base_model_lock"] == "models.official"


def test_lock_name_cannot_escape_directory() -> None:
    repository = LockRepository(ROOT / "locks")
    with pytest.raises(LockError, match="invalid lock name"):
        repository.load("../profiles/official")
