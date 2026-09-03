"""Padless Ulysses context parallelism for the Diffusers MiniMax H3 model.

MiniMax H3 receives separate text, video, and audio streams and packs them with
global indices inside ``forward``.  The safe sharding boundary is therefore the
transformer block stack: split after packing, keep the dense attention graph,
and gather before the global modality indices select the two output streams.
"""

from __future__ import annotations

from typing import Any


def h3_context_parallel_plan() -> dict[str, Any]:
    """Build the Diffusers hook plan without importing Diffusers at CLI import."""

    from diffusers.models._modeling_parallel import (
        ContextParallelInput,
        ContextParallelOutput,
    )

    return {
        # RoPE is computed once per model evaluation from the full position
        # vector.  Split its two outputs once, then reuse the local tables in
        # every block.
        "rope": {
            0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
            1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        # The outer Python loop passes the full index vector to every block.
        "transformer_blocks.*": {
            "adaln_indices": ContextParallelInput(
                split_dim=0, expected_dims=1, split_output=False
            ),
        },
        # Hidden states are threaded block-to-block and need splitting only at
        # the entry to the stack.
        "transformer_blocks.0": {
            "hidden_states": ContextParallelInput(
                split_dim=1, expected_dims=3, split_output=False
            ),
        },
        "norm_out": {
            "timestep_indices": ContextParallelInput(
                split_dim=0, expected_dims=1, split_output=False
            ),
        },
        # Gather both heads before the model applies global video/audio row
        # indices.  No row or attention edge is discarded.
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "audio_proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }


def _guard_padless(transformer: Any) -> None:
    if getattr(transformer, "_h3_flash_padless_guard", False):
        return
    original = transformer.forward

    def guarded(*args: Any, **kwargs: Any) -> Any:
        token_tags = kwargs.get("token_tags")
        if token_tags is None and len(args) >= 6:
            token_tags = args[5]
        if token_tags is not None and bool((token_tags < 0).any()):
            raise RuntimeError(
                "H3 lossless context parallelism requires a padless packed "
                "sequence; padding rows would require a row-sharded mask"
            )
        return original(*args, **kwargs)

    transformer.forward = guarded
    transformer._h3_flash_padless_guard = True


def enable_h3_ulysses(
    transformer: Any,
    *,
    degree: int,
    ulysses_anything: bool = True,
) -> dict[str, Any]:
    """Enable Diffusers-native dense Ulysses on an initialized process group."""

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("Ulysses requires an initialized torch process group")
    from diffusers.models._modeling_parallel import ContextParallelConfig

    world_size = dist.get_world_size()
    if degree != world_size:
        raise RuntimeError(
            f"Ulysses degree {degree} must equal process-group world size {world_size}"
        )
    if degree < 2:
        raise RuntimeError("Ulysses degree must be at least two")

    _guard_padless(transformer)
    # Pin the registered native backend.  It dispatches to Diffusers' context
    # parallel template and PyTorch SDPA without changing dense semantics.
    transformer.set_attention_backend("native")
    plan = h3_context_parallel_plan()
    transformer.enable_parallelism(
        config=ContextParallelConfig(
            ulysses_degree=degree,
            ulysses_anything=ulysses_anything,
        ),
        cp_plan=plan,
    )
    # Keep the exact plan object so later runtime-only attention adapters can
    # be installed below the input/output sharding hooks and then have those
    # hooks re-applied in the same order.
    transformer._h3_flash_cp_plan = plan
    metadata = {
        "backend": "diffusers_native_ulysses",
        "degree": degree,
        "ulysses_anything": ulysses_anything,
        "semantics": "dense",
    }
    return metadata


def with_context_parallel_reapplied(transformer: Any, action: Any) -> Any:
    """Install an adapter below Diffusers' context-parallel hook layer."""

    from diffusers.hooks.context_parallel import (
        apply_context_parallel,
        remove_context_parallel,
    )

    parallel_config = getattr(transformer, "_parallel_config", None)
    cp_config = (
        getattr(parallel_config, "context_parallel_config", None)
        if parallel_config is not None
        else None
    )
    plan = getattr(transformer, "_h3_flash_cp_plan", None)
    if cp_config is not None and plan is None:
        raise RuntimeError("context parallelism is enabled without its recorded plan")
    if cp_config is not None:
        remove_context_parallel(transformer, plan)
    try:
        return action()
    finally:
        if cp_config is not None:
            apply_context_parallel(transformer, cp_config, plan)
