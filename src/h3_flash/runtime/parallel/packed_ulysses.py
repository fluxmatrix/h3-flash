"""Dense packed Ulysses attention with one QKV collective per block."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from .context import with_context_parallel_reapplied
from .relayout import (
    can_pack_qkv,
    merge_heads,
    pack_qkv_destination_major,
    pack_qkv_reference,
)


class PackedUlyssesRuntime:
    """Request-scoped shape metadata and auditable runtime counters."""

    def __init__(self, group: Any) -> None:
        self.group = group
        self.world = dist.get_world_size(group)
        self._row_counts: tuple[list[int], int] | None = None
        self.requests = 0
        self.attention_calls = 0

    def begin_request(self) -> None:
        # Prompt lengths vary across Broad40. Every rank executes this at the
        # same request boundary, so the next attention call gathers its new
        # uneven shard sizes collectively exactly once.
        self._row_counts = None
        self.requests += 1

    def row_counts(self, rows_local: int) -> list[int]:
        if self._row_counts is not None:
            counts, seen = self._row_counts
            if seen != rows_local:
                raise RuntimeError(
                    f"packed Ulysses row count changed from {seen} to {rows_local} "
                    "inside one request"
                )
            return counts
        buffer = torch.zeros(self.world, dtype=torch.long, device="cuda")
        buffer[dist.get_rank(self.group)] = rows_local
        dist.all_reduce(buffer, group=self.group)
        counts = [int(value) for value in buffer.tolist()]
        self._row_counts = (counts, rows_local)
        return counts

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "packed_ulysses_sdpa",
            "degree": self.world,
            "semantics": "dense",
            "qkv_collectives_per_attention": 1,
            "output_collectives_per_attention": 1,
            "relayout": "triton_bit_exact",
            "row_count_collectives_per_request": 1,
            "requests": self.requests,
            "attention_calls": self.attention_calls,
            "origin": "SGLang-style packed layout via NVIDIA Sana GB200 path, adapted by FluxMatrix",
        }


def _all_to_all_varlen(
    x: torch.Tensor,
    out_numel: int,
    input_splits: list[int],
    output_splits: list[int],
    group: Any,
) -> torch.Tensor:
    output = torch.empty(out_numel, dtype=x.dtype, device=x.device)
    dist.all_to_all_single(
        output,
        x.reshape(-1),
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    return output


def _packed_qkv_all_to_all(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    runtime: PackedUlyssesRuntime,
) -> torch.Tensor:
    rows_local, heads, head_dim = q.shape
    heads_local = heads // runtime.world
    counts = runtime.row_counts(rows_local)
    rows_full = sum(counts)
    packed = (
        pack_qkv_destination_major(q, k, v, runtime.world)
        if can_pack_qkv(q, k, v)
        else pack_qkv_reference(q, k, v, runtime.world)
    )
    block = heads_local * 3 * head_dim
    exchanged = _all_to_all_varlen(
        packed,
        rows_full * block,
        [rows_local * block] * runtime.world,
        [count * block for count in counts],
        runtime.group,
    )
    return exchanged.reshape(rows_full, heads_local, 3 * head_dim)


def _packed_output_all_to_all(
    output: torch.Tensor,
    rows_local: int,
    runtime: PackedUlyssesRuntime,
) -> torch.Tensor:
    _, heads_local, head_dim = output.shape
    counts = runtime.row_counts(rows_local)
    block = heads_local * head_dim
    exchanged = _all_to_all_varlen(
        output,
        rows_local * runtime.world * block,
        [count * block for count in counts],
        [rows_local * block] * runtime.world,
        runtime.group,
    )
    return merge_heads(
        exchanged.reshape(runtime.world, rows_local, heads_local, head_dim)
    )


def install_packed_ulysses(
    transformer: Any,
    *,
    group: Any = None,
    fused_qknorm_rope: bool = True,
) -> PackedUlyssesRuntime:
    """Replace native CP attention layout/dispatch while retaining dense SDPA."""

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("packed Ulysses requires an initialized process group")
    group = group or dist.group.WORLD
    runtime = PackedUlyssesRuntime(group)
    if runtime.world < 2:
        raise RuntimeError("packed Ulysses degree must be at least two")

    def apply() -> None:
        for block in transformer.transformer_blocks:
            attention = block.attn

            def forward(
                hidden_states,
                rotary_emb=None,
                attention_mask=None,
                *,
                attention=attention,
            ):
                if attention_mask is not None:
                    raise RuntimeError(
                        "packed Ulysses requires padless dense attention"
                    )
                batch, rows_local, _ = hidden_states.shape
                if batch != 1:
                    raise RuntimeError(
                        "packed Ulysses currently requires batch size one"
                    )
                heads, head_dim = attention.heads, attention.head_dim
                query = attention.to_q(hidden_states).reshape(-1, heads, head_dim)
                key = attention.to_k(hidden_states).reshape(-1, heads, head_dim)
                value = attention.to_v(hidden_states).reshape(-1, heads, head_dim)
                if rotary_emb is None:
                    query, key = attention.norm_q(query), attention.norm_k(key)
                elif fused_qknorm_rope:
                    from h3_flash.runtime.kernels.fusions import (
                        fused_qknorm_rope as apply_fused_qknorm_rope,
                    )

                    cos, sin = rotary_emb
                    query = apply_fused_qknorm_rope(
                        query.unsqueeze(0),
                        attention.norm_q.weight,
                        cos,
                        sin,
                        attention.norm_q.eps,
                    ).squeeze(0)
                    key = apply_fused_qknorm_rope(
                        key.unsqueeze(0),
                        attention.norm_k.weight,
                        cos,
                        sin,
                        attention.norm_k.eps,
                    ).squeeze(0)
                else:
                    from diffusers.models.transformers.transformer_minimax_h3 import (
                        _apply_rotary_emb,
                    )

                    cos, sin = rotary_emb
                    query = _apply_rotary_emb(attention.norm_q(query), cos, sin)
                    key = _apply_rotary_emb(attention.norm_k(key), cos, sin)
                packed = _packed_qkv_all_to_all(query, key, value, runtime)
                query, key, value = packed.split(head_dim, dim=-1)
                output = torch.nn.functional.scaled_dot_product_attention(
                    query.transpose(0, 1).unsqueeze(0),
                    key.transpose(0, 1).unsqueeze(0),
                    value.transpose(0, 1).unsqueeze(0),
                    dropout_p=0.0,
                    is_causal=False,
                )
                output = output.squeeze(0).transpose(0, 1).contiguous()
                output = _packed_output_all_to_all(output, rows_local, runtime)
                output = output.reshape(batch, rows_local, heads * head_dim).type_as(
                    hidden_states
                )
                runtime.attention_calls += 1
                return attention.to_out[1](attention.to_out[0](output))

            attention.forward = forward

    with_context_parallel_reapplied(transformer, apply)
    return runtime
