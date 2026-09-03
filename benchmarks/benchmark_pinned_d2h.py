#!/usr/bin/env python3
"""Isolated pageable-versus-persistent-pinned D2H microbenchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from h3_flash.runtime.media import HostTransferPool


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--audio-samples", type=int, default=165600)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    import torch

    if args.repeats < 3:
        raise ValueError("at least three repeats are required")
    source = {
        "video": torch.zeros(
            (1, args.frames, 3, args.height, args.width),
            dtype=torch.float32,
            device=args.device,
        ),
        "audio": torch.zeros(
            (1, 2, args.audio_samples), dtype=torch.float32, device=args.device
        ),
    }
    torch.cuda.synchronize(args.device)

    # Fault in the pageable route and allocate/fault in persistent pinned
    # buffers before measurements. Allocation is intentionally excluded; this
    # measures steady-state repeated serving behavior.
    pageable_reference = {key: value.cpu() for key, value in source.items()}
    pool = HostTransferPool()
    pinned = pool.copy(source)
    assert all(torch.equal(pageable_reference[key], pinned[key]) for key in source)

    pageable_seconds = []
    pinned_seconds = []
    for _ in range(args.repeats):
        started = perf_counter()
        pageable = {key: value.cpu() for key, value in source.items()}
        pageable_seconds.append(perf_counter() - started)

        started = perf_counter()
        pinned = pool.copy(source)
        pinned_seconds.append(perf_counter() - started)
        assert all(torch.equal(pageable[key], pinned[key]) for key in source)

    pageable_stats = _stats(pageable_seconds)
    pinned_stats = _stats(pinned_seconds)
    delta = pageable_stats["median"] - pinned_stats["median"]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "isolated_steady_state_d2h_microbenchmark_not_end_to_end",
        "device": args.device,
        "torch": torch.__version__,
        "shapes": {key: list(value.shape) for key, value in source.items()},
        "dtypes": {key: str(value.dtype) for key, value in source.items()},
        "repeats": args.repeats,
        "correctness": {
            "torch_equal_each_repeat": True,
            "pool_buffer_count": pool.buffer_count,
        },
        "pageable_seconds": pageable_seconds,
        "persistent_pinned_seconds": pinned_seconds,
        "summary": {
            "pageable": pageable_stats,
            "persistent_pinned": pinned_stats,
            "median_reduction_seconds": delta,
            "median_reduction_percent": 100 * delta / pageable_stats["median"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
