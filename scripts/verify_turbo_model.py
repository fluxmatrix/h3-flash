#!/usr/bin/env python3
"""Verify an existing Turbo-derived model before it is reused."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h3_flash.locks import LockRepository
from h3_flash.turbo import verify_turbo_model, write_turbo_verification_marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination-model-root", type=Path, required=True)
    parser.add_argument("--source-model-root", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--model-lock", default="models.fast-turbo4-bf16-inputs")
    parser.add_argument("--size-only", action="store_true")
    args = parser.parse_args()
    model_lock = LockRepository().load(args.model_lock)
    expected_lora_sha256 = model_lock["files"][0]["sha256"]
    result = verify_turbo_model(
        args.destination_model_root,
        args.source_model_root,
        args.lora,
        expected_lora_sha256=expected_lora_sha256,
        hash_shards=not args.size_only,
    )
    result["model_lock"] = args.model_lock
    if not args.size_only:
        marker = write_turbo_verification_marker(result, model_lock=args.model_lock)
        result["verification_marker"] = str(marker)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
