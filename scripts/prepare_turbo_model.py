#!/usr/bin/env python3
"""Build a new Turbo-derived model root from locked official inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h3_flash.turbo import prepare_turbo_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model-root", type=Path, required=True)
    parser.add_argument("--destination-model-root", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--compute-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    args = parser.parse_args()
    manifest = prepare_turbo_model(
        args.source_model_root,
        args.destination_model_root,
        args.lora,
        strength=args.strength,
        device=args.device,
        compute_dtype=args.compute_dtype,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
