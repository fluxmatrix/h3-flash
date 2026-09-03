"""Read-only host diagnostics for the first reproducible H3-Flash lane."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import asdict, dataclass
from pathlib import Path

from .locks import LockError, LockRepository, sha256_file
from .profiles import ProfileError, ProfileRepository


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: str
    actual: str
    expected: str = ""
    detail: str = ""


def _command_version(
    executable: str, *arguments: str, configured_path: str | None = None
) -> Check:
    resolved = (
        configured_path
        if configured_path and Path(configured_path).is_file()
        else shutil.which(executable)
    )
    if not resolved:
        return Check(executable, "error", "not found", "available on PATH")
    try:
        result = subprocess.run(
            [resolved, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return Check(executable, "error", resolved, detail=str(error))
    output = (result.stdout or result.stderr).strip().splitlines()
    actual = output[0] if output else resolved
    status = "ok" if result.returncode == 0 else "error"
    return Check(executable, status, actual, detail=f"path={resolved}")


def _ffmpeg_check() -> Check:
    configured = os.environ.get("H3_FLASH_FFMPEG_BIN")
    version = _command_version("ffmpeg", "-version", configured_path=configured)
    if version.status != "ok":
        return version
    resolved = configured or shutil.which("ffmpeg")
    assert resolved is not None
    try:
        result = subprocess.run(
            [resolved, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return Check("ffmpeg", "error", resolved, detail=str(error))
    encoders = result.stdout + result.stderr
    missing = [name for name in ("libx264", "aac") if name not in encoders]
    return Check(
        "ffmpeg",
        "ok" if result.returncode == 0 and not missing else "error",
        version.actual,
        "FFmpeg with libx264 and AAC encoders",
        f"path={resolved}" + (f"; missing={','.join(missing)}" if missing else ""),
    )


def _gpu_check() -> Check:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return Check("nvidia_gpus", "error", "nvidia-smi not found", "8x NVIDIA B200")
    query = "index,name,driver_version,memory.total,compute_cap"
    try:
        result = subprocess.run(
            [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return Check("nvidia_gpus", "error", "query failed", detail=str(error))
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    parsed = [[field.strip() for field in row.split(",")] for row in rows]
    compatible = bool(
        result.returncode == 0
        and len(parsed) == 8
        and all(
            len(fields) == 5 and "B200" in fields[1] and fields[4] == "10.0"
            for fields in parsed
        )
    )
    return Check(
        "nvidia_gpus",
        "ok" if compatible else "error",
        f"count={len(rows)}; " + " | ".join(rows),
        "8x NVIDIA B200 (compute capability 10.0)",
        result.stderr.strip(),
    )


def _direct_requirements() -> dict[str, str]:
    candidates = (
        Path.cwd() / "requirements" / "b200-cu130.official.direct.txt",
        Path(__file__).resolve().parents[2]
        / "requirements"
        / "b200-cu130.official.direct.txt",
        Path(sysconfig.get_path("data"))
        / "h3_flash"
        / "requirements"
        / "b200-cu130.official.direct.txt",
    )
    for path in candidates:
        if not path.is_file():
            continue
        requirements = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.count("==") != 1:
                raise ValueError(f"unsupported direct requirement in {path}: {line}")
            name, version = line.split("==", 1)
            requirements[name] = version
        return requirements
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"B200 direct requirements not found; checked: {rendered}")


def _package_checks() -> list[Check]:
    checks = []
    try:
        requirements = _direct_requirements()
    except (OSError, ValueError) as error:
        return [Check("package_lock", "error", "invalid", detail=str(error))]
    for package, expected in requirements.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            checks.append(
                Check(f"package:{package}", "error", "not installed", expected)
            )
        else:
            status = "ok" if version == expected else "error"
            checks.append(Check(f"package:{package}", status, version, expected))
    return checks


def _checkout_checks(
    deps_root: Path | None,
    locks_dir: Path | None,
    *,
    research_deps: bool,
) -> list[Check]:
    if deps_root is None:
        return [Check("upstream_checkouts", "warn", "not checked", "pass --deps-root")]
    try:
        upstreams = LockRepository(locks_dir).load("upstreams")["git"]
    except (LockError, KeyError) as error:
        return [Check("upstream_checkouts", "error", "invalid lock", detail=str(error))]
    names = {"diffusers-minimax-h3": "diffusers_h3"}
    if research_deps:
        names.update(
            {
                "ai-toolkit": "ai_toolkit",
                "Sana": "sana_sol",
                "FastVideo": "fastvideo",
                "MiniMax-H3": "minimax_h3",
            }
        )
    expected = {
        directory: upstreams[lock_name]["commit"]
        for directory, lock_name in names.items()
    }
    checks = []
    for directory, commit in expected.items():
        checkout = deps_root / directory
        if not (checkout / ".git").exists():
            checks.append(Check(f"checkout:{directory}", "warn", "not found", commit))
            continue
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        actual = result.stdout.strip()
        dirty = ""
        if result.returncode == 0 and actual == commit:
            status_result = subprocess.run(
                ["git", "-C", str(checkout), "status", "--porcelain"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            dirty = status_result.stdout.strip()
        status = (
            "ok"
            if result.returncode == 0 and actual == commit and not dirty
            else "error"
        )
        checks.append(
            Check(
                f"checkout:{directory}",
                status,
                actual or "unknown",
                commit,
                "working tree has local changes" if dirty else "",
            )
        )
    return checks


def _lock_checks(
    locks_dir: Path | None,
    weights_root: Path | None,
    model_lock: str,
    hash_weights: bool,
) -> list[Check]:
    try:
        locks = LockRepository(locks_dir)
        upstream = locks.reference("upstreams")
        model = locks.load(model_lock)
    except LockError as error:
        return [Check("locks", "error", "invalid", detail=str(error))]
    checks = [
        Check("locks", "ok", f"{locks.directory}: {', '.join(locks.names())}"),
        Check("lock:upstreams", "ok", upstream["sha256"]),
        Check("lock:models", "ok", locks.digest(model_lock), model_lock),
    ]
    if weights_root is None:
        checks.append(
            Check("model_files", "warn", "not checked", "pass --weights-root")
        )
        return checks

    missing = []
    wrong_size = []
    wrong_hash = []
    verified_bytes = 0
    for entry in model.get("files", []):
        path = weights_root / entry["local_path"]
        if not path.is_file():
            missing.append(entry["local_path"])
            continue
        size = path.stat().st_size
        if size != entry["bytes"]:
            wrong_size.append(f"{entry['local_path']}:{size}!={entry['bytes']}")
            continue
        verified_bytes += size
        if hash_weights and sha256_file(path) != entry["sha256"]:
            wrong_hash.append(entry["local_path"])
    failures = missing + wrong_size + wrong_hash
    checks.append(
        Check(
            "model_files",
            "error" if failures else "ok",
            f"{len(model.get('files', [])) - len(failures)}/{len(model.get('files', []))} files; "
            f"{verified_bytes} bytes; sha256={'checked' if hash_weights else 'not checked'}",
            model_lock,
            "; ".join(failures[:10]),
        )
    )
    return checks


def run_doctor(
    *,
    profiles_dir: Path | None = None,
    locks_dir: Path | None = None,
    deps_root: Path | None = None,
    weights_root: Path | None = None,
    model_lock: str = "models.official",
    hash_weights: bool = False,
    research_deps: bool = False,
) -> dict:
    checks = [
        Check(
            "python",
            "ok" if sys.version_info[:2] == (3, 12) else "error",
            platform.python_version(),
            "3.12.x",
        ),
        Check(
            "platform",
            "ok" if sys.platform.startswith("linux") else "error",
            platform.platform(),
            "Linux",
        ),
        _command_version("git", "--version"),
        _ffmpeg_check(),
        _gpu_check(),
    ]
    try:
        repository = ProfileRepository(profiles_dir)
    except ProfileError as error:
        checks.append(Check("profiles", "error", "invalid", detail=str(error)))
    else:
        checks.append(
            Check(
                "profiles",
                "ok",
                f"{repository.directory}: {', '.join(repository.names())}",
            )
        )
    checks.extend(_package_checks())
    checks.extend(_checkout_checks(deps_root, locks_dir, research_deps=research_deps))
    checks.extend(_lock_checks(locks_dir, weights_root, model_lock, hash_weights))
    summary = {
        status: sum(check.status == status for check in checks)
        for status in ("ok", "warn", "error")
    }
    return {
        "schema_version": 1,
        "target_lane": "linux-python312-cuda130-8xb200",
        "summary": summary,
        "checks": [asdict(check) for check in checks],
    }


def format_doctor(report: dict) -> str:
    lines = [
        f"H3-Flash doctor ({report['target_lane']})",
        f"summary: {json.dumps(report['summary'], sort_keys=True)}",
    ]
    for check in report["checks"]:
        marker = {"ok": "OK", "warn": "WARN", "error": "ERROR"}[check["status"]]
        line = f"[{marker:5}] {check['name']}: {check['actual']}"
        if check["expected"]:
            line += f" (expected: {check['expected']})"
        if check["detail"]:
            line += f" [{check['detail']}]"
        lines.append(line)
    return "\n".join(lines)
