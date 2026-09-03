import tomllib
from pathlib import Path

import pytest

from h3_flash.cli import main
from h3_flash.profiles import ProfileError, ProfileRepository

ROOT = Path(__file__).resolve().parents[1]


def test_profile_inheritance_preserves_official_contract() -> None:
    repository = ProfileRepository(ROOT / "profiles")

    official = repository.resolve("official")
    lossless = repository.resolve("lossless")
    flash = repository.resolve("flash")

    assert official["sampling"]["api_num_inference_steps"] == 50
    assert official["sampling"]["model_evaluations"] == 49
    assert official["model"]["text_encoder"] == "official_bf16"
    assert lossless["attention"]["semantics"] == "dense"
    assert lossless["parallel"]["sequence_parallel"] == "ulysses_anything"
    assert lossless["correctness"]["contract"] == "numerically_equivalent"
    assert flash["sampling"]["model_evaluations"] == 4
    assert flash["model"]["text_encoder"] == "official_bf16"
    assert flash["correctness"]["contract"] == "approximate"
    assert "tensor_gate" not in flash["correctness"]


def test_resolved_profile_is_not_mutable_repository_state() -> None:
    repository = ProfileRepository(ROOT / "profiles")
    first = repository.resolve("lossless")
    first["sampling"]["model_evaluations"] = 7
    assert repository.resolve("lossless")["sampling"]["model_evaluations"] == 49


def test_profile_digest_is_stable() -> None:
    first = ProfileRepository(ROOT / "profiles")
    second = ProfileRepository(ROOT / "profiles")
    assert first.digest("flash") == second.digest("flash")


def test_public_profiles_resolve_to_the_measured_runtime_contracts() -> None:
    repository = ProfileRepository(ROOT / "profiles")
    lossless = repository.resolve("lossless")
    flash = repository.resolve("flash")

    assert lossless["id"] == "lossless"
    assert lossless["quality_class"] == "numerically_equivalent"
    assert lossless["execution"]["accelerator_count"] == 8
    assert lossless["sampling"]["model_evaluations"] == 49
    assert lossless["output"]["encoder"] == "ffmpeg_raw_pipe"
    assert flash["id"] == "flash"
    assert flash["execution"]["accelerator_count"] == 8
    assert flash["sampling"]["model_evaluations"] == 4
    assert flash["model"]["text_encoder"] == "official_bf16"
    assert flash["attention"]["semantics"] == "dense"
    assert flash["output"]["encoder"] == "ffmpeg_raw_pipe"


def test_cycle_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.toml").write_text(
        'schema_version=1\nid="a"\nparent="b"\nquality_class="reference"\n'
    )
    (tmp_path / "b.toml").write_text(
        'schema_version=1\nid="b"\nparent="a"\nquality_class="reference"\n'
    )
    repository = ProfileRepository(tmp_path)
    with pytest.raises(ProfileError, match="inheritance cycle"):
        repository.resolve("a")


def test_all_source_profiles_parse_as_toml() -> None:
    for path in (ROOT / "profiles").glob("*.toml"):
        with path.open("rb") as handle:
            assert tomllib.load(handle)["id"] == path.stem


def test_pinned_d2h_ablation_changes_only_the_declared_mechanism() -> None:
    repository = ProfileRepository(ROOT / "profiles")
    official = repository.resolve("official")
    ablation = repository.resolve("lossless-pinned-d2h")

    assert ablation["quality_class"] == "bit_exact"
    assert ablation["optimizations"] == {"persistent_pinned_d2h": True}
    for section in ("model", "sampling", "attention", "execution"):
        assert ablation[section] == official[section]


def test_dense_fa4_ablation_preserves_model_and_dense_attention() -> None:
    repository = ProfileRepository(ROOT / "profiles")
    official = repository.resolve("official")
    ablation = repository.resolve("lossless-dense-fa4")

    assert ablation["quality_class"] == "numerically_equivalent"
    assert ablation["attention"] == {
        "semantics": "dense",
        "backend": "flash_attention_4",
    }
    for section in ("model", "sampling", "execution"):
        assert ablation[section] == official[section]


def test_turbo4_profile_changes_model_and_schedule_but_keeps_dense_runtime() -> None:
    repository = ProfileRepository(ROOT / "profiles")
    lossless = repository.resolve("lossless-8xb200")
    turbo = repository.resolve("fast-turbo4-bf16-dense-8xb200")

    assert turbo["quality_class"] == "approximate"
    assert turbo["sampling"]["api_num_inference_steps"] == 5
    assert turbo["sampling"]["model_evaluations"] == 4
    assert turbo["model"]["text_encoder"] == "official_bf16"
    assert turbo["model"]["turbo_lora"] == "lightx2v_fl2v_4step_v1"
    assert turbo["attention"] == lossless["attention"]
    assert turbo["parallel"] == lossless["parallel"]
    assert turbo["vae"] == lossless["vae"]


def test_profile_list_defaults_to_three_public_versions(capsys) -> None:
    assert main(["--profiles-dir", str(ROOT / "profiles"), "profile", "list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines] == ["official", "lossless", "flash"]


def test_profile_list_all_includes_internal_ablations(capsys) -> None:
    assert (
        main(
            [
                "--profiles-dir",
                str(ROOT / "profiles"),
                "profile",
                "list",
                "--all",
            ]
        )
        == 0
    )
    names = {line.split()[0] for line in capsys.readouterr().out.splitlines()}
    assert "official" in names
    assert "lossless-8xb200-no-vae-parallel" in names
