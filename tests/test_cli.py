import json
from pathlib import Path

import pytest

from h3_flash import cli
from h3_flash.evals import EvaluationSuite


def test_generate_flash_writes_a_top_level_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    captured = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        suite = EvaluationSuite.load(kwargs["suite_path"])
        assert suite.cases[0].prompt == "A tiny robot waves."
        assert suite.cases[0].seed == 7
        case_root = kwargs["output_root"] / "cases" / "generated"
        case_root.mkdir(parents=True)
        (case_root / "output.mp4").write_bytes(b"video")
        (case_root / "result.json").write_text("{}\n", encoding="utf-8")
        return {"counts": {"ok": 1}}, 0

    monkeypatch.setattr(cli, "launch_distributed_suite", fake_launch)

    exit_code = cli.main(
        [
            "generate",
            "--mode",
            "FLASH",
            "--prompt",
            "A tiny robot waves.",
            "--output-dir",
            str(output_root),
            "--weights-root",
            str(tmp_path / "weights"),
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    assert captured["profile_name"] == "flash"
    assert captured["model_root"] == (
        tmp_path / "weights" / "official-diffusers-turbo4-bf16"
    )
    assert captured["physical_gpus"] == tuple(str(index) for index in range(8))
    assert (output_root / "output.mp4").read_bytes() == b"video"
    request = json.loads((output_root / "request.json").read_text(encoding="utf-8"))
    assert request["cases"][0]["prompt"] == "A tiny robot waves."


def test_generate_lossless_requires_eight_unique_gpus(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="exactly eight GPUs"):
        cli.main(
            [
                "generate",
                "--mode",
                "LOSSLESS",
                "--prompt",
                "A tiny robot waves.",
                "--output-dir",
                str(tmp_path / "output"),
                "--weights-root",
                str(tmp_path / "weights"),
                "--gpus",
                "0,1",
            ]
        )
