from pathlib import Path
import shutil

import pytest

from h3_flash.runtime.media.ffmpeg import encode_video_ffmpeg


def test_ffmpeg_rejects_an_undeclared_output_contract(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    video = torch.zeros((1, 32, 32, 3), dtype=torch.uint8)
    audio = torch.zeros((32, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="pinned H3-Flash output contract"):
        encode_video_ffmpeg(
            video,
            fps=24,
            output_path=tmp_path / "output.mp4",
            audio=audio,
            audio_sample_rate=32000,
            video_codec="hevc",
        )


def test_ffmpeg_places_mp4_metadata_before_media(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    output = tmp_path / "output.mp4"
    metadata = encode_video_ffmpeg(
        torch.zeros((2, 32, 32, 3), dtype=torch.uint8),
        fps=24,
        output_path=output,
        audio=torch.zeros((3200, 2), dtype=torch.float32),
        audio_sample_rate=32000,
    )
    payload = output.read_bytes()
    assert 0 < payload.find(b"moov") < payload.find(b"mdat")
    assert metadata["streaming"] == "faststart"
