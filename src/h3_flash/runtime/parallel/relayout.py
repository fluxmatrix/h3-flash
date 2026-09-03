"""Bit-exact Triton relayouts used by packed Ulysses collectives."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_qkv_kernel(
    out_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    total_elements,
    rows,
    heads_local,
    head_dim,
    stride_q_row,
    stride_q_head,
    stride_k_row,
    stride_k_head,
    stride_v_row,
    stride_v_head,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    dim = offsets % head_dim
    head_slot = offsets // head_dim
    local_head = head_slot % heads_local
    row_slot = head_slot // heads_local
    row = row_slot % rows
    destination = row_slot // rows
    global_head = destination * heads_local + local_head
    q = tl.load(
        q_ptr + row * stride_q_row + global_head * stride_q_head + dim, mask=mask
    )
    k = tl.load(
        k_ptr + row * stride_k_row + global_head * stride_k_head + dim, mask=mask
    )
    v = tl.load(
        v_ptr + row * stride_v_row + global_head * stride_v_head + dim, mask=mask
    )
    base = head_slot * (3 * head_dim) + dim
    tl.store(out_ptr + base, q, mask=mask)
    tl.store(out_ptr + base + head_dim, k, mask=mask)
    tl.store(out_ptr + base + 2 * head_dim, v, mask=mask)


@triton.jit
def _merge_heads_kernel(
    out_ptr, x_ptr, total_elements, world, rows, inner, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    tail = offsets % inner
    slot = offsets // inner
    source = slot % world
    row = slot // world
    source_offset = (source * rows + row) * inner + tail
    tl.store(
        out_ptr + offsets,
        tl.load(x_ptr + source_offset, mask=mask),
        mask=mask,
    )


def can_pack_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return (
        q.is_cuda
        and q.ndim == 3
        and q.shape == k.shape == v.shape
        and q.dtype == k.dtype == v.dtype
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and not torch.compiler.is_compiling()
    )


def pack_qkv_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, world: int
) -> torch.Tensor:
    rows, heads, head_dim = q.shape
    if heads % world:
        raise ValueError(f"heads ({heads}) must divide Ulysses degree ({world})")
    heads_local = heads // world
    output = torch.empty(
        (world, rows, heads_local, 3 * head_dim), dtype=q.dtype, device=q.device
    )
    for index, tensor in enumerate((q, k, v)):
        shards = tensor.reshape(rows, world, heads_local, head_dim).permute(1, 0, 2, 3)
        output[..., index * head_dim : (index + 1) * head_dim].copy_(shards)
    return output


def pack_qkv_destination_major(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, world: int
) -> torch.Tensor:
    """Move values directly into destination-major order without arithmetic."""

    rows, heads, head_dim = q.shape
    if heads % world:
        raise ValueError(f"heads ({heads}) must divide Ulysses degree ({world})")
    heads_local = heads // world
    output = torch.empty(
        (world, rows, heads_local, 3 * head_dim), dtype=q.dtype, device=q.device
    )
    total = rows * heads * head_dim
    if total:
        block = 1024
        _pack_qkv_kernel[(triton.cdiv(total, block),)](
            output,
            q,
            k,
            v,
            total,
            rows,
            heads_local,
            head_dim,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            BLOCK=block,
            num_warps=8,
        )
    return output


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Transpose source-rank and row dimensions without changing any value."""

    world, rows, heads_local, head_dim = x.shape
    if not x.is_contiguous() or torch.compiler.is_compiling():
        return (
            x.permute(1, 0, 2, 3)
            .contiguous()
            .reshape(rows, world * heads_local, head_dim)
        )
    output = torch.empty(
        (rows, world, heads_local, head_dim), dtype=x.dtype, device=x.device
    )
    total = output.numel()
    if total:
        block = 1024
        _merge_heads_kernel[(triton.cdiv(total, block),)](
            output,
            x,
            total,
            world,
            rows,
            heads_local * head_dim,
            BLOCK=block,
            num_warps=8,
        )
    return output.reshape(rows, world * heads_local, head_dim)
