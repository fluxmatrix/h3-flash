from pathlib import Path

from h3_flash.doctor import _direct_requirements, format_doctor, run_doctor

ROOT = Path(__file__).resolve().parents[1]


def test_doctor_report_has_machine_readable_summary() -> None:
    report = run_doctor(profiles_dir=ROOT / "profiles", locks_dir=ROOT / "locks")

    assert report["schema_version"] == 1
    assert sum(report["summary"].values()) == len(report["checks"])
    assert any(check["name"] == "profiles" for check in report["checks"])
    rendered = format_doctor(report)
    assert "H3-Flash doctor" in rendered
    assert "profiles" in rendered


def test_default_runtime_requirements_exclude_research_only_packages() -> None:
    requirements = _direct_requirements()

    assert requirements["torch"] == "2.13.0+cu130"
    assert "flash_attn" not in requirements
    assert "cuda-python" not in requirements
