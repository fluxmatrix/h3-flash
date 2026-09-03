#!/usr/bin/env python3
"""Numerical gate and microbenchmark for the H3 Triton fusion bundle."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional

from h3_flash.run import _write_json_atomic
from h3_flash.runtime.kernels.fusions import (
    fused_qknorm_rope,
    fused_residual_gate_rmsnorm_modulate,
    fused_rmsnorm_modulate,
    fused_swiglu,
)


def _error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - reference.float()).abs()
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "relative_l2": float(delta.norm() / reference.float().norm().clamp_min(1e-12)),
        "cosine": float(
            functional.cosine_similarity(
                actual.float().flatten(), reference.float().flatten(), dim=0
            )
        ),
    }


def _median_ms(callable_, repeats: int) -> float:
    for _ in range(3):
        callable_()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=257)
    parser.add_argument("--hidden", type=int, default=5376)
    parser.add_argument("--ffn", type=int, default=14336)
    parser.add_argument("--heads", type=int, default=7)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    rows, hidden = args.rows, args.hidden
    x = torch.randn(1, rows, hidden, device=device, dtype=dtype)
    branch = torch.randn_like(x)
    weight = torch.randn(hidden, device=device, dtype=dtype)
    wide = torch.randn(9, 6 * hidden, device=device, dtype=dtype)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, _ = wide.chunk(6, -1)
    indices = torch.arange(rows, device=device) % 9
    eps = 1e-5

    def eager_first():
        normed = functional.rms_norm(x, (hidden,), weight, eps)
        return normed * (
            1.0 + scale_msa.index_select(0, indices)
        ) + shift_msa.index_select(0, indices)

    def fused_first():
        return fused_rmsnorm_modulate(x, weight, scale_msa, shift_msa, indices, eps)

    def eager_second():
        updated = x + gate_msa.index_select(0, indices) * branch
        normed = functional.rms_norm(updated, (hidden,), weight, eps)
        normed = normed * (
            1.0 + scale_mlp.index_select(0, indices)
        ) + shift_mlp.index_select(0, indices)
        return updated, normed

    def fused_second():
        return fused_residual_gate_rmsnorm_modulate(
            x, branch, gate_msa, weight, scale_mlp, shift_mlp, indices, eps
        )

    swiglu_input = torch.randn(rows, 2 * args.ffn, device=device, dtype=dtype)

    def eager_swiglu():
        value, gate = swiglu_input.chunk(2, -1)
        return value * functional.silu(gate)

    def fused_swiglu_call():
        return fused_swiglu(swiglu_input)

    query = torch.randn(1, rows, args.heads, args.head_dim, device=device, dtype=dtype)
    qk_weight = torch.randn(args.head_dim, device=device, dtype=dtype)
    angles = torch.randn(rows, args.rotary_dim, device=device, dtype=torch.float32)
    cos, sin = angles.cos(), angles.sin()

    def eager_qk():
        from diffusers.models.transformers.transformer_minimax_h3 import (
            _apply_rotary_emb,
        )

        normed = functional.rms_norm(query, (args.head_dim,), qk_weight, eps)
        return _apply_rotary_emb(normed, cos, sin)

    def fused_qk():
        return fused_qknorm_rope(query, qk_weight, cos, sin, eps)

    first_reference = eager_first()
    first_actual = fused_first()
    second_reference = eager_second()
    second_actual = fused_second()
    swiglu_reference = eager_swiglu()
    swiglu_actual = fused_swiglu_call()
    qk_reference = eager_qk()
    qk_actual = fused_qk()
    report = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "shape": {
            "rows": rows,
            "hidden": hidden,
            "ffn": args.ffn,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "rotary_dim": args.rotary_dim,
        },
        "numerical": {
            "rmsnorm_modulate": _error(first_actual, first_reference),
            "residual_output": _error(second_actual[0], second_reference[0]),
            "residual_rmsnorm_modulate": _error(second_actual[1], second_reference[1]),
            "swiglu": _error(swiglu_actual, swiglu_reference),
            "qknorm_partial_rope": _error(qk_actual, qk_reference),
        },
        "median_ms": {
            "rmsnorm_modulate_eager": _median_ms(eager_first, args.repeats),
            "rmsnorm_modulate_fused": _median_ms(fused_first, args.repeats),
            "residual_chain_eager": _median_ms(eager_second, args.repeats),
            "residual_chain_fused": _median_ms(fused_second, args.repeats),
            "swiglu_eager": _median_ms(eager_swiglu, args.repeats),
            "swiglu_fused": _median_ms(fused_swiglu_call, args.repeats),
            "qknorm_rope_eager": _median_ms(eager_qk, args.repeats),
            "qknorm_rope_fused": _median_ms(fused_qk, args.repeats),
        },
    }
    _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
