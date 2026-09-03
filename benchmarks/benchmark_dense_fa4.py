#!/usr/bin/env python3
"""Numerical and latency gate for dense FA4 at an H3 attention shape."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from h3_flash.runtime.attention.dense_fa4 import fa4_provenance, load_fa4


def _cuda_measure(torch, function, repeats: int) -> tuple[object, list[float]]:
    output = None
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1000)
    return output, samples


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fa4-site-packages", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sequence_length, args.heads, args.head_dim, args.repeats) < 1:
        parser.error("shape and repeat values must be positive")

    import torch
    from torch.nn import functional

    if not torch.cuda.is_available():
        raise RuntimeError("dense FA4 benchmark requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    shape = (1, args.sequence_length, args.heads, args.head_dim)
    q = torch.randn(shape, dtype=torch.bfloat16, device=device)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device)
    fa4 = load_fa4(args.fa4_site_packages)

    def native():
        return functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)

    def cute():
        result = fa4(q, k, v, causal=False)
        return result[0] if isinstance(result, tuple) else result

    # Compile/dispatch warmups are intentionally outside measured samples.
    warmup_started = perf_counter()
    native()
    cute()
    torch.cuda.synchronize()
    warmup_seconds = perf_counter() - warmup_started

    native_output, native_samples = _cuda_measure(torch, native, args.repeats)
    fa4_output, fa4_samples = _cuda_measure(torch, cute, args.repeats)
    delta = native_output.float() - fa4_output.float()
    reference = native_output.float()
    max_abs = delta.abs().max().item()
    mean_abs = delta.abs().mean().item()
    rel_l2 = (delta.norm() / reference.norm().clamp_min(1e-12)).item()
    cosine = functional.cosine_similarity(
        reference.flatten(), fa4_output.float().flatten(), dim=0
    ).item()
    native_median = statistics.median(native_samples)
    fa4_median = statistics.median(fa4_samples)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "shape_bshd": list(shape),
        "dtype": str(q.dtype),
        "seed": args.seed,
        "repeats": args.repeats,
        "warmup_seconds": warmup_seconds,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "fa4": fa4_provenance(args.fa4_site_packages),
        },
        "numerical": {
            "torch_equal": torch.equal(native_output, fa4_output),
            "finite": bool(torch.isfinite(fa4_output).all().item()),
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "relative_l2": rel_l2,
            "cosine_similarity": cosine,
        },
        "latency_seconds": {
            "native_sdpa": _distribution(native_samples),
            "dense_fa4": _distribution(fa4_samples),
            "median_reduction_seconds": native_median - fa4_median,
            "median_reduction_percent": 100
            * (native_median - fa4_median)
            / native_median,
            "median_speedup": native_median / fa4_median,
        },
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    if not all(math.isfinite(value) for value in (max_abs, mean_abs, rel_l2, cosine)):
        raise RuntimeError("non-finite numerical comparison")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
