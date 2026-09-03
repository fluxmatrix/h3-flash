from pathlib import Path

import pytest

from h3_flash.manifest import GenerationRequest
from h3_flash.run import RunError, _video_to_uint8, _write_json_atomic


def test_atomic_json_writer_is_stable(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "result.json"
    _write_json_atomic(target, {"z": 1, "a": "音视频"})
    assert target.read_text(encoding="utf-8") == '{\n  "a": "音视频",\n  "z": 1\n}\n'


def test_official_run_rejects_unknown_artifact_before_loading(tmp_path: Path) -> None:
    from h3_flash.run import run_official_diffusers

    with pytest.raises(RunError, match="artifact_format"):
        run_official_diffusers(
            request=GenerationRequest(prompt="x", seed=1),
            model_root=tmp_path,
            output_dir=tmp_path / "out",
            artifact_format="avi",
            profiles=None,
            locks=None,
        )


def test_video_float_contract_is_converted_to_encoder_uint8() -> None:
    torch = pytest.importorskip("torch")
    video = torch.tensor([-0.1, 0.0, 0.5, 1.0, 1.1], dtype=torch.float32)
    converted = _video_to_uint8(video)
    assert converted.dtype == torch.uint8
    assert converted.tolist() == [0, 0, 128, 255, 255]


def test_video_channel_layout_is_converted_for_pyav() -> None:
    torch = pytest.importorskip("torch")
    video = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
    converted = _video_to_uint8(video)
    assert converted.shape == (2, 4, 5, 3)
    assert converted.is_contiguous()
