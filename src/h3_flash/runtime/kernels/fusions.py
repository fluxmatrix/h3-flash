"""MiniMax H3 elementwise fusions adapted from NVIDIA Sana's GB200 path.

The kernels retain every row and operation in the official dense model while
avoiding wide intermediate round trips through HBM.  Their evaluation order is
not bit-identical to eager PyTorch, so they require a bounded-numerical gate.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _residual_gate_rmsnorm_modulate_kernel(
        hidden_out_ptr,
        normed_out_ptr,
        residual_ptr,
        branch_ptr,
        weight_ptr,
        gate_ptr,
        scale_ptr,
        shift_ptr,
        index_ptr,
        n_cols,
        n_index,
        eps,
        stride_row,
        stride_table_row,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < n_cols
        offset = row * stride_row + columns
        table_row = tl.load(index_ptr + (row % n_index))
        table_offset = table_row * stride_table_row + columns
        residual = tl.load(residual_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        hidden = residual + gate * branch
        tl.store(
            hidden_out_ptr + offset,
            hidden.to(hidden_out_ptr.dtype.element_ty),
            mask=mask,
        )
        variance = tl.sum(hidden * hidden, axis=0) / n_cols
        normed = hidden * tl.math.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        normed *= weight
        scale = tl.load(scale_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        output = normed * (1.0 + scale) + shift
        tl.store(
            normed_out_ptr + offset,
            output.to(normed_out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _rmsnorm_modulate_kernel(
        out_ptr,
        x_ptr,
        weight_ptr,
        scale_ptr,
        shift_ptr,
        index_ptr,
        n_cols,
        n_index,
        eps,
        stride_row,
        stride_table_row,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < n_cols
        offset = row * stride_row + columns
        table_offset = tl.load(index_ptr + (row % n_index)) * stride_table_row + columns
        value = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(value * value, axis=0) / n_cols
        normed = value * tl.math.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
        output = normed * weight * (1.0 + scale) + shift
        tl.store(out_ptr + offset, output.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_kernel(
        out_ptr,
        x_ptr,
        n_cols,
        stride_in_row,
        stride_out_row,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, BLOCK)
        mask = columns < n_cols
        value = tl.load(x_ptr + row * stride_in_row + columns, mask=mask, other=0.0).to(
            tl.float32
        )
        gate = tl.load(
            x_ptr + row * stride_in_row + n_cols + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        output = value * (gate * tl.sigmoid(gate))
        tl.store(
            out_ptr + row * stride_out_row + columns,
            output.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _qknorm_partial_rope_kernel(
        out_ptr,
        x_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        head_dim,
        rotary_dim,
        half_dim,
        heads,
        sequence,
        eps,
        BLOCK: tl.constexpr,
    ):
        program = tl.program_id(0)
        token = (program // heads) % sequence
        columns = tl.arange(0, BLOCK)
        mask = columns < head_dim
        base = program * head_dim
        value = tl.load(x_ptr + base + columns, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(value * value, axis=0) / head_dim
        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        normed = value * tl.math.rsqrt(variance + eps) * weight
        in_rotary = columns < rotary_dim
        first_half = columns < half_dim
        partner = tl.where(first_half, columns + half_dim, columns - half_dim)
        partner_value = tl.load(x_ptr + base + partner, mask=in_rotary, other=0.0).to(
            tl.float32
        )
        partner_weight = tl.load(weight_ptr + partner, mask=in_rotary, other=0.0).to(
            tl.float32
        )
        partner_normed = partner_value * tl.math.rsqrt(variance + eps) * partner_weight
        rotated = tl.where(first_half, -partner_normed, partner_normed)
        cos = tl.load(
            cos_ptr + token * rotary_dim + columns,
            mask=in_rotary,
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr + token * rotary_dim + columns,
            mask=in_rotary,
            other=0.0,
        ).to(tl.float32)
        output = tl.where(in_rotary, normed * cos + rotated * sin, normed)
        tl.store(
            out_ptr + base + columns,
            output.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _warps_for(block: int) -> int:
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    return 4


def _row_addressable(table: torch.Tensor) -> torch.Tensor:
    return table if table.stride(-1) == 1 else table.contiguous()


def fused_rmsnorm_modulate(x, weight, scale, shift, index, eps):
    columns = x.shape[-1]
    flat = x.reshape(-1, columns).contiguous()
    scale, shift = _row_addressable(scale), _row_addressable(shift)
    output = torch.empty_like(flat)
    block = _next_power_of_two(columns)
    _rmsnorm_modulate_kernel[(flat.shape[0],)](
        output,
        flat,
        weight,
        scale,
        shift,
        index,
        columns,
        index.numel(),
        eps,
        flat.stride(0),
        scale.stride(0),
        BLOCK=block,
        num_warps=_warps_for(block),
    )
    return output.view_as(x)


def fused_residual_gate_rmsnorm_modulate(
    residual, branch, gate, weight, scale, shift, index, eps
):
    columns = residual.shape[-1]
    residual_flat = residual.reshape(-1, columns).contiguous()
    branch_flat = branch.reshape(-1, columns).contiguous()
    gate = _row_addressable(gate)
    scale = _row_addressable(scale)
    shift = _row_addressable(shift)
    hidden = torch.empty_like(residual_flat)
    normed = torch.empty_like(residual_flat)
    block = _next_power_of_two(columns)
    _residual_gate_rmsnorm_modulate_kernel[(residual_flat.shape[0],)](
        hidden,
        normed,
        residual_flat,
        branch_flat,
        weight,
        gate,
        scale,
        shift,
        index,
        columns,
        index.numel(),
        eps,
        residual_flat.stride(0),
        gate.stride(0),
        BLOCK=block,
        num_warps=_warps_for(block),
    )
    return hidden.view_as(residual), normed.view_as(residual)


def fused_swiglu(x):
    columns = x.shape[-1] // 2
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    output = torch.empty(flat.shape[0], columns, dtype=x.dtype, device=x.device)
    block = _next_power_of_two(columns)
    _swiglu_kernel[(flat.shape[0],)](
        output,
        flat,
        columns,
        flat.stride(0),
        output.stride(0),
        BLOCK=block,
        num_warps=_warps_for(block),
    )
    return output.view(*x.shape[:-1], columns)


def fused_qknorm_rope(x, weight, cos, sin, eps):
    batch, sequence, heads, head_dim = x.shape
    rotary_dim = cos.shape[-1]
    if rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            "fused qknorm+rope requires an even rotary dimension no wider "
            f"than the head, got rotary_dim={rotary_dim}, head_dim={head_dim}"
        )
    flat = x.reshape(-1, head_dim).contiguous()
    output = torch.empty_like(flat)
    _qknorm_partial_rope_kernel[(flat.shape[0],)](
        output,
        flat,
        weight,
        cos,
        sin,
        head_dim,
        rotary_dim,
        rotary_dim // 2,
        heads,
        sequence,
        eps,
        BLOCK=_next_power_of_two(head_dim),
        num_warps=4,
    )
    return output.view(batch, sequence, heads, head_dim)
