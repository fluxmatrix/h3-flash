#!/usr/bin/env python3
"""Eight-rank equality gate for packed Ulysses movement and dense SDPA."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from h3_flash.runtime.parallel.packed_ulysses import (
    PackedUlyssesRuntime,
    _packed_output_all_to_all,
    _packed_qkv_all_to_all,
)
from h3_flash.runtime.parallel.relayout import merge_heads


def exchange_reference(
    tensor: torch.Tensor, counts: list[int], group: object
) -> torch.Tensor:
    rows, heads, head_dim = tensor.shape
    world = dist.get_world_size(group)
    heads_local = heads // world
    packed = (
        tensor.reshape(rows, world, heads_local, head_dim)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    output = torch.empty(
        sum(counts) * heads_local * head_dim,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_to_all_single(
        output,
        packed.reshape(-1),
        output_split_sizes=[count * heads_local * head_dim for count in counts],
        input_split_sizes=[rows * heads_local * head_dim] * world,
        group=group,
    )
    return output.reshape(sum(counts), heads_local, head_dim)


def output_reference(
    output: torch.Tensor, rows: int, counts: list[int], group: object
) -> torch.Tensor:
    _, heads_local, head_dim = output.shape
    world = dist.get_world_size(group)
    block = heads_local * head_dim
    exchanged = torch.empty(
        rows * world * block, dtype=output.dtype, device=output.device
    )
    dist.all_to_all_single(
        exchanged,
        output.reshape(-1),
        output_split_sizes=[rows * block] * world,
        input_split_sizes=[count * block for count in counts],
        group=group,
    )
    return merge_heads(exchanged.reshape(world, rows, heads_local, head_dim))


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group = dist.group.WORLD
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    if 56 % world:
        raise RuntimeError(f"world size {world} must divide 56 heads")
    rows = 19 + (world - rank)  # deliberately uneven
    torch.manual_seed(1000 + rank)
    q = torch.randn(rows, 56, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    counts_buffer = torch.zeros(world, dtype=torch.long, device="cuda")
    counts_buffer[rank] = rows
    dist.all_reduce(counts_buffer, group=group)
    counts = [int(value) for value in counts_buffer.tolist()]

    q_ref = exchange_reference(q, counts, group)
    k_ref = exchange_reference(k, counts, group)
    v_ref = exchange_reference(v, counts, group)
    reference_attention = (
        torch.nn.functional.scaled_dot_product_attention(
            q_ref.transpose(0, 1).unsqueeze(0),
            k_ref.transpose(0, 1).unsqueeze(0),
            v_ref.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
            is_causal=False,
        )
        .squeeze(0)
        .transpose(0, 1)
        .contiguous()
    )
    reference_output = output_reference(reference_attention, rows, counts, group)

    runtime = PackedUlyssesRuntime(group)
    runtime.begin_request()
    packed = _packed_qkv_all_to_all(q, k, v, runtime)
    q_got, k_got, v_got = packed.split(128, dim=-1)
    packed_attention = (
        torch.nn.functional.scaled_dot_product_attention(
            q_got.transpose(0, 1).unsqueeze(0),
            k_got.transpose(0, 1).unsqueeze(0),
            v_got.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
            is_causal=False,
        )
        .squeeze(0)
        .transpose(0, 1)
        .contiguous()
    )
    packed_output = _packed_output_all_to_all(packed_attention, rows, runtime)
    local = torch.tensor(
        [
            torch.equal(q_got, q_ref),
            torch.equal(k_got, k_ref),
            torch.equal(v_got, v_ref),
            torch.equal(packed_output, reference_output),
        ],
        dtype=torch.int32,
        device="cuda",
    )
    dist.all_reduce(local, op=dist.ReduceOp.MIN)
    if rank == 0:
        report = {
            "world_size": world,
            "uneven_row_counts": counts,
            "q_bit_identical": bool(local[0]),
            "k_bit_identical": bool(local[1]),
            "v_bit_identical": bool(local[2]),
            "dense_attention_output_bit_identical": bool(local[3]),
            "all_bit_identical": bool(local.min()),
        }
        print(json.dumps(report, indent=2))
    if not bool(local.min()):
        raise RuntimeError("packed Ulysses equality gate failed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
