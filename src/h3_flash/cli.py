"""Command-line entry point for H3-Flash."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from .benchmark import launch_official_suite, run_official_worker
from .distributed_benchmark import (
    launch_distributed_suite,
    run_distributed_suite_worker,
)
from .doctor import format_doctor, run_doctor
from .evals import EvaluationSuite
from .locks import LockRepository
from .manifest import GenerationRequest, build_manifest
from .mechanisms import MechanismRegistry
from .models import (
    download_commands,
    download_models,
    format_download_plan,
    write_verification_marker,
    verification_ok,
    verify_models,
)
from .profiles import ProfileRepository
from .run import run_official_diffusers


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def _repository(args: argparse.Namespace) -> ProfileRepository:
    return ProfileRepository(_path(args.profiles_dir))


def _locks(args: argparse.Namespace) -> LockRepository:
    return LockRepository(_path(args.locks_dir))


def _mechanisms(args: argparse.Namespace) -> MechanismRegistry:
    return MechanismRegistry(_path(args.mechanisms_file))


def _doctor(args: argparse.Namespace) -> int:
    report = run_doctor(
        profiles_dir=_path(args.profiles_dir),
        locks_dir=_path(args.locks_dir),
        deps_root=_path(args.deps_root),
        weights_root=_path(args.weights_root),
        model_lock=args.model_lock,
        hash_weights=args.hash_weights,
        research_deps=args.research_deps,
    )
    print(json.dumps(report, indent=2) if args.json else format_doctor(report))
    return 2 if report["summary"]["error"] else 0


def _profile_list(args: argparse.Namespace) -> int:
    repository = _repository(args)
    names = repository.names() if args.all else ("official", "lossless", "flash")
    for name in names:
        profile = repository.resolve(name)
        print(
            f"{name:12} {profile['quality_class']:24} "
            f"{profile.get('status', 'unknown')} {repository.digest(name)}"
        )
    return 0


def _profile_show(args: argparse.Namespace) -> int:
    repository = _repository(args)
    profile = repository.resolve(args.name)
    result = {
        "profile": profile,
        "profile_sha256": repository.digest(args.name),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _mechanism_list(args: argparse.Namespace) -> int:
    registry = _mechanisms(args)
    for entry in registry.entries(public_lane=args.lane):
        headline = "headline-candidate" if entry["headline_candidate"] else "bundle"
        print(
            f"{entry['id']:34} {entry['public_lane']:8} "
            f"{entry['verification']:27} {headline}"
        )
    return 0


def _mechanism_show(args: argparse.Namespace) -> int:
    registry = _mechanisms(args)
    result = {
        "mechanism": registry.get(args.name),
        "registry_sha256": registry.digest(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _manifest(args: argparse.Namespace) -> int:
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    request = GenerationRequest(
        prompt=prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        fps=args.fps,
    )
    manifest = build_manifest(
        _repository(args),
        args.profile,
        request,
        backend=args.backend,
        locks=_locks(args),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _model_plan(args: argparse.Namespace) -> int:
    commands = download_commands(
        _locks(args), args.lock, args.weights_root.resolve(), hf_bin=args.hf_bin
    )
    print(format_download_plan(commands))
    return 0


def _model_download(args: argparse.Namespace) -> int:
    download_models(
        _locks(args), args.lock, args.weights_root.resolve(), hf_bin=args.hf_bin
    )
    report = verify_models(
        _locks(args), args.lock, args.weights_root.resolve(), hash_files=True
    )
    if verification_ok(report):
        write_verification_marker(report)
    print(json.dumps(report["counts"], sort_keys=True))
    return 0 if verification_ok(report) else 2


def _model_verify(args: argparse.Namespace) -> int:
    report = verify_models(
        _locks(args),
        args.lock,
        args.weights_root.resolve(),
        hash_files=args.hash,
    )
    if args.write_marker:
        if not args.hash:
            raise SystemExit("--write-marker requires --hash")
        if verification_ok(report):
            marker = write_verification_marker(report)
            report["verification_marker"] = str(marker)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"lock: {args.lock}")
        print(f"weights_root: {report['weights_root']}")
        print(f"sha256: {'checked' if args.hash else 'not checked'}")
        print(f"counts: {json.dumps(report['counts'], sort_keys=True)}")
        for record in report["files"]:
            if record["status"] != "ok":
                print(f"[{record['status']}] {record['local_path']}")
    return 0 if verification_ok(report) else 2


def _run_official(args: argparse.Namespace) -> int:
    if args.suite_file:
        if not args.case_id:
            raise SystemExit("--case-id is required with --suite-file")
        suite = EvaluationSuite.load(args.suite_file)
        overrides = {
            name: value
            for name, value in (
                ("seed", args.seed),
                ("width", args.width),
                ("height", args.height),
                ("num_frames", args.frames),
                ("fps", args.fps),
            )
            if value is not None
        }
        request = suite.request(args.case_id, **overrides)
    else:
        if args.seed is None:
            raise SystemExit("--seed is required with --prompt-file")
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        request = GenerationRequest(
            prompt=prompt,
            seed=args.seed,
            width=args.width or 1344,
            height=args.height or 768,
            num_frames=args.frames or 124,
            fps=args.fps or 24,
        )
    result = run_official_diffusers(
        request=request,
        model_root=args.model_root,
        output_dir=args.output_dir,
        artifact_format=args.output_format,
        profiles=_repository(args),
        locks=_locks(args),
        generation_device=args.generation_device,
        text_device=args.text_device,
        profile_name=args.profile,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _benchmark_official_worker(args: argparse.Namespace) -> int:
    summary = run_official_worker(
        suite_path=args.suite_file,
        model_root=args.model_root,
        output_root=args.output_root,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        profiles=_repository(args),
        locks=_locks(args),
        device=args.device,
        profile_name=args.profile,
        fa4_site_packages=args.fa4_site_packages,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["counts"]["error"] == 0 else 2


def _benchmark_official(args: argparse.Namespace) -> int:
    physical_gpus = tuple(
        value.strip() for value in args.gpus.split(",") if value.strip()
    )
    summary, exit_code = launch_official_suite(
        suite_path=args.suite_file,
        model_root=args.model_root,
        output_root=args.output_root,
        physical_gpus=physical_gpus,
        profiles_dir=_path(args.profiles_dir),
        locks_dir=_path(args.locks_dir),
        profile_name=args.profile,
        fa4_site_packages=args.fa4_site_packages,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


def _benchmark_distributed_worker(args: argparse.Namespace) -> int:
    summary = run_distributed_suite_worker(
        suite_path=args.suite_file,
        model_root=args.model_root,
        output_root=args.output_root,
        profiles=_repository(args),
        locks=_locks(args),
        profile_name=args.profile,
    )
    if summary is not None:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _benchmark_distributed(args: argparse.Namespace) -> int:
    physical_gpus = tuple(
        value.strip() for value in args.gpus.split(",") if value.strip()
    )
    summary, exit_code = launch_distributed_suite(
        suite_path=args.suite_file,
        model_root=args.model_root,
        output_root=args.output_root,
        physical_gpus=physical_gpus,
        profiles_dir=_path(args.profiles_dir),
        locks_dir=_path(args.locks_dir),
        profile_name=args.profile,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


def _generate(args: argparse.Namespace) -> int:
    prompt = (
        args.prompt.strip()
        if args.prompt is not None
        else args.prompt_file.read_text(encoding="utf-8").strip()
    )
    if not prompt:
        raise SystemExit("prompt must not be empty")

    mode = args.mode.lower()
    physical_gpus = tuple(
        value.strip() for value in args.gpus.split(",") if value.strip()
    )
    if not physical_gpus or len(set(physical_gpus)) != len(physical_gpus):
        raise SystemExit("--gpus must be a non-empty unique comma-separated list")
    if mode in {"lossless", "flash"} and len(physical_gpus) != 8:
        raise SystemExit(f"{mode.upper()} requires exactly eight GPUs")

    request = GenerationRequest(
        prompt=prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        fps=args.fps,
    )
    request.validate()
    weights_root = args.weights_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    model_root = weights_root / (
        "official-diffusers-turbo4-bf16"
        if mode == "flash"
        else "official-diffusers"
    )
    suite_data = {
        "schema_version": 1,
        "suite_id": "h3-flash-user-generation",
        "resolution": [request.width, request.height],
        "num_frames": request.num_frames,
        "fps": request.fps,
        "cases": [
            {
                "case_id": "generated",
                "mode": request.mode,
                "seed": request.seed,
                "prompt": request.prompt,
            }
        ],
    }

    with tempfile.TemporaryDirectory(prefix="h3-flash-generate-") as temporary:
        suite_path = Path(temporary) / "request.json"
        suite_path.write_text(
            json.dumps(suite_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if mode == "official":
            summary, exit_code = launch_official_suite(
                suite_path=suite_path,
                model_root=model_root,
                output_root=output_root,
                physical_gpus=(physical_gpus[0],),
                profiles_dir=_path(args.profiles_dir),
                locks_dir=_path(args.locks_dir),
                profile_name="official",
            )
        else:
            summary, exit_code = launch_distributed_suite(
                suite_path=suite_path,
                model_root=model_root,
                output_root=output_root,
                physical_gpus=physical_gpus,
                profiles_dir=_path(args.profiles_dir),
                locks_dir=_path(args.locks_dir),
                profile_name=mode,
            )
        if exit_code != 0:
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return exit_code
        shutil.copy2(suite_path, output_root / "request.json")

    generated_video = output_root / "cases" / "generated" / "output.mp4"
    public_video = output_root / "output.mp4"
    shutil.copy2(generated_video, public_video)
    result_path = output_root / "cases" / "generated" / "result.json"
    print(f"[h3-flash] video: {public_video}")
    print(f"[h3-flash] metadata: {result_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="h3-flash")
    parser.add_argument("--profiles-dir", help="override the profile directory")
    parser.add_argument("--locks-dir", help="override the lockfile directory")
    parser.add_argument("--mechanisms-file", help="override the optimization registry")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect the local runtime")
    doctor.add_argument("--deps-root", help="directory containing pinned checkouts")
    doctor.add_argument("--weights-root", help="root containing model files")
    doctor.add_argument("--model-lock", default="models.official")
    doctor.add_argument("--hash-weights", action="store_true")
    doctor.add_argument(
        "--research-deps",
        action="store_true",
        help="also inspect optional AI Toolkit, FastVideo, Sana, and MiniMax checkouts",
    )
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=_doctor)

    profile = subparsers.add_parser("profile", help="inspect resolved profiles")
    profile_subparsers = profile.add_subparsers(dest="profile_command", required=True)
    profile_list = profile_subparsers.add_parser("list")
    profile_list.add_argument(
        "--all", action="store_true", help="include internal ablation profiles"
    )
    profile_list.set_defaults(handler=_profile_list)
    profile_show = profile_subparsers.add_parser("show")
    profile_show.add_argument("name")
    profile_show.set_defaults(handler=_profile_show)

    mechanism = subparsers.add_parser("mechanism", help="inspect optimization taxonomy")
    mechanism_subparsers = mechanism.add_subparsers(
        dest="mechanism_command", required=True
    )
    mechanism_list = mechanism_subparsers.add_parser("list")
    mechanism_list.add_argument("--lane", choices=("lossless", "flash"))
    mechanism_list.set_defaults(handler=_mechanism_list)
    mechanism_show = mechanism_subparsers.add_parser("show")
    mechanism_show.add_argument("name")
    mechanism_show.set_defaults(handler=_mechanism_show)

    manifest = subparsers.add_parser(
        "manifest", help="resolve one immutable run manifest"
    )
    manifest.add_argument("--profile", required=True)
    manifest.add_argument("--backend", required=True)
    manifest.add_argument("--prompt-file", type=Path, required=True)
    manifest.add_argument("--seed", type=int, required=True)
    manifest.add_argument("--width", type=int, default=1344)
    manifest.add_argument("--height", type=int, default=768)
    manifest.add_argument("--frames", type=int, default=124)
    manifest.add_argument("--fps", type=int, default=24)
    manifest.set_defaults(handler=_manifest)

    model = subparsers.add_parser("model", help="prepare locked model files")
    model_subparsers = model.add_subparsers(dest="model_command", required=True)
    for name, handler in (
        ("plan", _model_plan),
        ("download", _model_download),
        ("verify", _model_verify),
    ):
        command = model_subparsers.add_parser(name)
        command.add_argument("--lock", required=True)
        command.add_argument("--weights-root", type=Path, required=True)
        command.set_defaults(handler=handler)
        if name in {"plan", "download"}:
            command.add_argument("--hf-bin", default="hf")
        if name == "verify":
            command.add_argument("--hash", action="store_true")
            command.add_argument("--write-marker", action="store_true")
            command.add_argument("--json", action="store_true")

    run = subparsers.add_parser("run", help="execute a pinned inference backend")
    run_subparsers = run.add_subparsers(dest="run_backend", required=True)
    official = run_subparsers.add_parser(
        "official", help="run the official Diffusers reference"
    )
    official.add_argument("--model-root", type=Path, required=True)
    official_input = official.add_mutually_exclusive_group(required=True)
    official_input.add_argument("--prompt-file", type=Path)
    official_input.add_argument("--suite-file", type=Path)
    official.add_argument("--case-id")
    official.add_argument("--output-dir", type=Path, required=True)
    official.add_argument("--output-format", choices=("latent", "mp4"), default="mp4")
    official.add_argument("--seed", type=int)
    official.add_argument("--width", type=int)
    official.add_argument("--height", type=int)
    official.add_argument("--frames", type=int)
    official.add_argument("--fps", type=int)
    official.add_argument("--generation-device", default="cuda:0")
    official.add_argument("--text-device", default="cuda:0")
    official.add_argument(
        "--profile",
        choices=(
            "official",
            "lossless-pinned-d2h",
            "lossless-dense-fa4",
            "lossless-1xb200",
        ),
        default="official",
    )
    official.set_defaults(handler=_run_official)

    generate = subparsers.add_parser(
        "generate", help="generate one video from a local prompt"
    )
    generate.add_argument(
        "--mode", type=str.upper, choices=("OFFICIAL", "LOSSLESS", "FLASH"),
        default="FLASH"
    )
    generate_input = generate.add_mutually_exclusive_group(required=True)
    generate_input.add_argument("--prompt")
    generate_input.add_argument("--prompt-file", type=Path)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--weights-root", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--width", type=int, default=1344)
    generate.add_argument("--height", type=int, default=768)
    generate.add_argument("--frames", type=int, default=124)
    generate.add_argument("--fps", type=int, default=24)
    generate.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    generate.set_defaults(handler=_generate)

    benchmark = subparsers.add_parser(
        "benchmark", help="run a reproducible evaluation shard"
    )
    benchmark_subparsers = benchmark.add_subparsers(
        dest="benchmark_backend", required=True
    )
    benchmark_launch = benchmark_subparsers.add_parser(
        "official",
        help="launch a complete official suite over isolated single-GPU workers",
    )
    benchmark_launch.add_argument("--suite-file", type=Path, required=True)
    benchmark_launch.add_argument("--model-root", type=Path, required=True)
    benchmark_launch.add_argument("--output-root", type=Path, required=True)
    benchmark_launch.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    benchmark_launch.add_argument(
        "--profile",
        choices=(
            "official",
            "lossless-pinned-d2h",
            "lossless-dense-fa4",
            "lossless-1xb200",
        ),
        default="official",
    )
    benchmark_launch.add_argument("--fa4-site-packages", type=Path)
    benchmark_launch.set_defaults(handler=_benchmark_official)
    benchmark_official = benchmark_subparsers.add_parser(
        "official-worker", help="run one single-GPU shard of an official suite"
    )
    benchmark_official.add_argument("--suite-file", type=Path, required=True)
    benchmark_official.add_argument("--model-root", type=Path, required=True)
    benchmark_official.add_argument("--output-root", type=Path, required=True)
    benchmark_official.add_argument("--worker-index", type=int, required=True)
    benchmark_official.add_argument("--worker-count", type=int, required=True)
    benchmark_official.add_argument("--device", default="cuda:0")
    benchmark_official.add_argument(
        "--profile",
        choices=(
            "official",
            "lossless-pinned-d2h",
            "lossless-dense-fa4",
            "lossless-1xb200",
        ),
        default="official",
    )
    benchmark_official.add_argument("--fa4-site-packages", type=Path)
    benchmark_official.set_defaults(handler=_benchmark_official_worker)

    benchmark_distributed_launch = benchmark_subparsers.add_parser(
        "distributed",
        help="launch one collective LOSSLESS/FLASH suite with Ulysses sequence parallelism",
    )
    benchmark_distributed_launch.add_argument("--suite-file", type=Path, required=True)
    benchmark_distributed_launch.add_argument("--model-root", type=Path, required=True)
    benchmark_distributed_launch.add_argument("--output-root", type=Path, required=True)
    benchmark_distributed_launch.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    benchmark_distributed_launch.add_argument(
        "--profile",
        choices=(
            "lossless",
            "flash",
            "lossless-8xb200",
            "lossless-8xb200-ffmpeg",
            "lossless-8xb200-no-vae-parallel",
            "lossless-8xb200-native-ulysses-only",
            "fast-turbo4-bf16-dense-8xb200",
            "fast-turbo4-bf16-dense-8xb200-ffmpeg",
        ),
        default="lossless",
    )
    benchmark_distributed_launch.set_defaults(handler=_benchmark_distributed)

    benchmark_distributed_worker = benchmark_subparsers.add_parser(
        "distributed-worker", help=argparse.SUPPRESS
    )
    benchmark_distributed_worker.add_argument("--suite-file", type=Path, required=True)
    benchmark_distributed_worker.add_argument("--model-root", type=Path, required=True)
    benchmark_distributed_worker.add_argument("--output-root", type=Path, required=True)
    benchmark_distributed_worker.add_argument(
        "--profile",
        choices=(
            "lossless",
            "flash",
            "lossless-8xb200",
            "lossless-8xb200-ffmpeg",
            "lossless-8xb200-no-vae-parallel",
            "lossless-8xb200-native-ulysses-only",
            "fast-turbo4-bf16-dense-8xb200",
            "fast-turbo4-bf16-dense-8xb200-ffmpeg",
        ),
        default="lossless",
    )
    benchmark_distributed_worker.set_defaults(handler=_benchmark_distributed_worker)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
