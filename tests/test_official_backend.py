from pathlib import Path

import pytest

from h3_flash.backends.official_diffusers import (
    OfficialBackendError,
    OfficialDiffusersBackend,
)


def test_official_backend_requires_diffusers_layout(tmp_path: Path) -> None:
    backend = OfficialDiffusersBackend(tmp_path)
    with pytest.raises(OfficialBackendError, match="model not found"):
        backend.load()


def test_official_backend_load_is_lazy(tmp_path: Path) -> None:
    backend = OfficialDiffusersBackend(tmp_path)
    assert backend.pipeline is None
    assert backend.load_seconds == 0.0


def test_official_backend_records_parallel_intent(tmp_path: Path) -> None:
    backend = OfficialDiffusersBackend(tmp_path, ulysses_degree=8)
    assert backend.ulysses_degree == 8
    assert backend.parallel_runtime == {"backend": "none", "degree": 1}
    assert backend.fusion_runtime == {"enabled": False}
