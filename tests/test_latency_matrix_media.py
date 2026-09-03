import hashlib
import json
from pathlib import Path
from tools.reporting import validate_latency_matrix_media as validator


def _task(tmp_path: Path) -> dict:
    artifact = tmp_path / "output.mp4"
    artifact.write_bytes(b"synthetic media")
    result = {
        "outputs": {
            "path": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return {
        "version": "FLASH",
        "suite_id": "suite",
        "case_id": "case",
        "width": 832,
        "height": 480,
        "num_frames": 124,
        "fps": 24,
        "result_path": result_path,
    }


def test_safe_check_returns_complete_success_record(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        validator,
        "_probe",
        lambda _ffprobe, _artifact: {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 832,
                    "height": 480,
                    "nb_read_frames": "124",
                    "avg_frame_rate": "24/1",
                    "duration": "5.166667",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "sample_rate": "32000",
                    "duration": "5.184000",
                },
            ]
        },
    )

    record = validator._safe_check(_task(tmp_path), Path("ffprobe"))

    assert record["status"] == "ok"
    assert record["failures"] == []
    assert all(record["checks"].values())


def test_safe_check_retains_probe_failure(tmp_path: Path, monkeypatch) -> None:
    def fail(_ffprobe, _artifact):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(validator, "_probe", fail)
    record = validator._safe_check(_task(tmp_path), Path("ffprobe"))

    assert record["status"] == "error"
    assert record["failures"] == ["probe_or_validation_exception"]
    assert "probe failed" in record["error"]
