"""Install independently switchable, dense-preserving H3 block fusions."""

from __future__ import annotations

from typing import Any

from .fusions import (
    HAVE_TRITON,
    fused_qknorm_rope,
    fused_residual_gate_rmsnorm_modulate,
    fused_rmsnorm_modulate,
    fused_swiglu,
)


def _error_metrics(actual: Any, reference: Any) -> dict[str, float]:
    from torch.nn import functional

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


def _patch_blocks(
    transformer: Any,
    modulate: bool,
    swiglu: bool,
    runtime: dict[str, Any],
) -> None:
    for block_index, block in enumerate(transformer.transformer_blocks):

        def forward(
            hidden_states,
            temb,
            adaln_indices,
            rotary_emb,
            attention_mask=None,
            *,
            block=block,
            gate_this_block=block_index == 0,
        ):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                block.adaln_proj(temb)
            )
            if modulate:
                normed = fused_rmsnorm_modulate(
                    hidden_states,
                    block.norm1.weight,
                    scale_msa,
                    shift_msa,
                    adaln_indices,
                    block.norm1.eps,
                )
                if gate_this_block and "rmsnorm_modulate" not in runtime["gate"]:
                    reference = block.norm1(hidden_states)
                    reference = reference * (
                        1.0 + scale_msa.index_select(0, adaln_indices)
                    ) + shift_msa.index_select(0, adaln_indices)
                    runtime["gate"]["rmsnorm_modulate"] = _error_metrics(
                        normed, reference
                    )
            else:
                normed = block.norm1(hidden_states)
                normed = normed * (
                    1.0 + scale_msa.index_select(0, adaln_indices)
                ) + shift_msa.index_select(0, adaln_indices)
            attention_output = block.attn(normed, rotary_emb, attention_mask)
            residual_input = hidden_states
            if modulate:
                hidden_states, normed = fused_residual_gate_rmsnorm_modulate(
                    residual_input,
                    attention_output,
                    gate_msa,
                    block.norm2.weight,
                    scale_mlp,
                    shift_mlp,
                    adaln_indices,
                    block.norm2.eps,
                )
                if gate_this_block and "residual_chain" not in runtime["gate"]:
                    reference_hidden = (
                        residual_input
                        + gate_msa.index_select(0, adaln_indices) * attention_output
                    )
                    reference_normed = block.norm2(reference_hidden)
                    reference_normed = reference_normed * (
                        1.0 + scale_mlp.index_select(0, adaln_indices)
                    ) + shift_mlp.index_select(0, adaln_indices)
                    runtime["gate"]["residual_output"] = _error_metrics(
                        hidden_states, reference_hidden
                    )
                    runtime["gate"]["residual_chain"] = _error_metrics(
                        normed, reference_normed
                    )
            else:
                hidden_states = (
                    hidden_states
                    + gate_msa.index_select(0, adaln_indices) * attention_output
                )
                normed = block.norm2(hidden_states)
                normed = normed * (
                    1.0 + scale_mlp.index_select(0, adaln_indices)
                ) + shift_mlp.index_select(0, adaln_indices)
            if swiglu:
                swiglu_layer, _, output_projection = block.ff.net
                projected = swiglu_layer.proj(normed)
                activated = fused_swiglu(projected)
                if gate_this_block and "swiglu" not in runtime["gate"]:
                    value, gate = projected.chunk(2, -1)
                    reference = value * swiglu_layer.activation(gate)
                    runtime["gate"]["swiglu"] = _error_metrics(activated, reference)
                feed_forward_output = output_projection(activated)
            else:
                feed_forward_output = block.ff(normed)
            return (
                hidden_states
                + gate_mlp.index_select(0, adaln_indices) * feed_forward_output
            )

        block.forward = forward


def _patch_attention(transformer: Any, runtime: dict[str, Any]) -> None:
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    for block_index, block in enumerate(transformer.transformer_blocks):
        attention = block.attn

        def forward(
            hidden_states,
            rotary_emb=None,
            attention_mask=None,
            *,
            attention=attention,
            gate_this_block=block_index == 0,
        ):
            processor = attention.processor
            if attention.fused_projections:
                query, key, value = attention.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                query = attention.to_q(hidden_states)
                key = attention.to_k(hidden_states)
                value = attention.to_v(hidden_states)
            query = query.unflatten(-1, (attention.heads, -1))
            key = key.unflatten(-1, (attention.heads, -1))
            value = value.unflatten(-1, (attention.heads, -1))
            if rotary_emb is None:
                query = attention.norm_q(query)
                key = attention.norm_k(key)
            else:
                cos, sin = rotary_emb
                query = fused_qknorm_rope(
                    query,
                    attention.norm_q.weight,
                    cos,
                    sin,
                    attention.norm_q.eps,
                )
                key = fused_qknorm_rope(
                    key,
                    attention.norm_k.weight,
                    cos,
                    sin,
                    attention.norm_k.eps,
                )
                if gate_this_block and "qknorm_partial_rope" not in runtime["gate"]:
                    from diffusers.models.transformers.transformer_minimax_h3 import (
                        _apply_rotary_emb,
                    )

                    reference_query = _apply_rotary_emb(
                        attention.norm_q(
                            attention.to_q(hidden_states).unflatten(
                                -1, (attention.heads, -1)
                            )
                        ),
                        cos,
                        sin,
                    )
                    runtime["gate"]["qknorm_partial_rope"] = _error_metrics(
                        query, reference_query
                    )
            output = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=processor._attention_backend,
                parallel_config=processor._parallel_config,
            )
            output = output.flatten(2, 3).type_as(query)
            return attention.to_out[1](attention.to_out[0](output))

        attention.forward = forward


def install_transformer_fusions(
    transformer: Any,
    *,
    modulate: bool = True,
    swiglu: bool = True,
    qknorm_rope: bool = True,
) -> dict[str, Any]:
    if not HAVE_TRITON:
        raise RuntimeError("H3 transformer fusions require Triton")
    runtime = {
        "enabled": True,
        "modulate": modulate,
        "swiglu": swiglu,
        "qknorm_rope": qknorm_rope,
        "origin": "NVIDIA Sana GB200 architecture adapted by FluxMatrix",
        "gate": {},
    }
    if modulate or swiglu:
        _patch_blocks(transformer, modulate, swiglu, runtime)
    if qknorm_rope:
        _patch_attention(transformer, runtime)
    return runtime
