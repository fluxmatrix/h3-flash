import json
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readmes_embed_github_video_attachments() -> None:
    for readme_name in ("README.md", "README_zh-CN.md"):
        readme = (ROOT / readme_name).read_text(encoding="utf-8")
        sources = re.findall(
            r'<video src="(https://github\.com/user-attachments/assets/'
            r'[a-f0-9-]+)" controls></video>',
            readme,
        )
        assert len(sources) == 11
        assert len(set(sources)) == 11
        assert "assets/previews" not in readme

    repository_mp4s = list((ROOT / "assets" / "samples" / "flash").glob("*.mp4"))
    repository_mp4s += list(
        (ROOT / "assets" / "comparisons" / "modes").glob("*.mp4")
    )
    assert len(repository_mp4s) == 10


def test_readme_performance_matches_canonical_results() -> None:
    results = json.loads(
        (ROOT / "benchmarks" / "results" / "b200_e2e.json").read_text(
            encoding="utf-8"
        )
    )
    assert results["metric"]["name"] == "resident_worker_e2e_seconds"
    assert results["metric"]["statistic"] == "median"
    assert results["metric"]["cases_per_cell"] == 4
    for readme_name in ("README.md", "README_zh-CN.md"):
        readme = (ROOT / readme_name).read_text(encoding="utf-8")
        for row in results["results"]:
            for mode in results["modes"]:
                assert f'{row[mode]:.3f}' in readme


def test_directly_invoked_scripts_are_executable() -> None:
    for relative in (
        "scripts/setup_prompt_enhancer.sh",
        "scripts/serve_prompt_enhancer.sh",
    ):
        assert os.access(ROOT / relative, os.X_OK), f"{relative} must be executable"


def test_profile_ladder_is_explicit() -> None:
    profiles = {}
    for name in ("official", "lossless", "flash"):
        with (ROOT / "profiles" / f"{name}.toml").open("rb") as handle:
            profiles[name] = tomllib.load(handle)

    assert profiles["official"]["status"] == "measured_local"
    assert profiles["lossless"]["status"] == "measured_local"
    assert profiles["flash"]["status"] == "technical_measured_pending_human_review"
    assert profiles["official"]["quality_class"] == "reference"
    assert profiles["lossless"]["parent"] == "lossless-8xb200-ffmpeg"
    assert profiles["flash"]["parent"] == "fast-turbo4-bf16-dense-8xb200-ffmpeg"
    assert profiles["official"]["sampling"]["api_num_inference_steps"] == 50
    assert profiles["official"]["sampling"]["model_evaluations"] == 49
    assert profiles["official"]["execution"]["accelerator_count"] == 1
    assert profiles["official"]["attention"]["semantics"] == "dense"


def test_broad40_contract() -> None:
    with (ROOT / "configs" / "evals" / "h3-broad40-v1.1.json").open() as handle:
        suite = json.load(handle)

    cases = suite["cases"]
    assert suite["suite_id"] == "h3-broad40-v1.1"
    assert suite["resolution"] == [1344, 768]
    assert suite["num_frames"] == 124
    assert suite["fps"] == 24
    assert len(cases) == 40
    assert len({case["case_id"] for case in cases}) == 40
    assert all(case["mode"] == "t2va" for case in cases)
    assert all(
        case["prompt"].count("integrated_multimodal_description:") == 1
        for case in cases
    )
    assert all(case["prompt"].count("overall_soundscape:") == 1 for case in cases)
    assert all(case["prompt"].count("non_diegetic_music:") == 1 for case in cases)
