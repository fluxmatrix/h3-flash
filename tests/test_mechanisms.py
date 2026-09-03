from pathlib import Path

import pytest

from h3_flash.mechanisms import MechanismError, MechanismRegistry

ROOT = Path(__file__).resolve().parents[1]


def test_registry_uses_one_public_lossless_lane() -> None:
    registry = MechanismRegistry(ROOT / "configs" / "mechanisms.toml")
    lossless = registry.entries(public_lane="lossless")

    assert lossless
    assert {item["public_lane"] for item in lossless} == {"lossless"}
    assert {item["verification"] for item in lossless} == {
        "bit_exact",
        "generated_tensor_exact",
        "mathematically_equivalent",
    }
    assert all(
        item["validation_profile"] in {"exact", "equivalent", "output"}
        for item in lossless
    )


def test_approximation_is_never_registered_as_lossless() -> None:
    registry = MechanismRegistry(ROOT / "configs" / "mechanisms.toml")
    fast = registry.entries(public_lane="flash")

    assert {item["verification"] for item in fast} == {"approximate"}
    assert "lightx2v_turbo4" in {item["id"] for item in fast}
    assert "qwen_int8_convrot" in {item["id"] for item in fast}


def test_headline_mechanisms_report_median_e2e_reductions() -> None:
    registry = MechanismRegistry(ROOT / "configs" / "mechanisms.toml")
    headline = [item for item in registry.entries() if item["headline_candidate"]]

    assert headline
    assert all(item["median_e2e_reduction_seconds"] > 0 for item in headline)
    assert all(0 < item["median_e2e_reduction_percent"] < 100 for item in headline)


def test_registry_digest_is_stable_and_records_are_copies() -> None:
    first = MechanismRegistry(ROOT / "configs" / "mechanisms.toml")
    second = MechanismRegistry(ROOT / "configs" / "mechanisms.toml")
    record = first.get("request_flush_elision")
    record["public_lane"] = "flash"

    assert first.get("request_flush_elision")["public_lane"] == "lossless"
    assert first.digest() == second.digest()


def test_invalid_lane_and_verification_pair_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mechanisms.toml"
    path.write_text(
        """schema_version = 1
headline_minimum_median_reduction_percent = 5.0
headline_must_exceed_run_noise = true
[[mechanisms]]
id = "bad"
title = "bad"
public_lane = "lossless"
verification = "approximate"
validation_profile = "fast"
status = "planned_optional"
origin = "test"
changes = ["weights"]
gates = ["quality"]
timing_scope = "diffusion"
headline_candidate = false
evidence_note = "test"
""",
        encoding="utf-8",
    )

    with pytest.raises(MechanismError, match="belong to flash"):
        MechanismRegistry(path)
