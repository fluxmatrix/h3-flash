"""Mathematically equivalent rank-local H3 input and output projections."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional


def _split_sizes(total: int, parts: int) -> list[int]:
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]


def _split_bounds(total: int, parts: int, rank: int) -> tuple[int, int]:
    sizes = _split_sizes(total, parts)
    start = sum(sizes[:rank])
    return start, start + sizes[rank]


def _local_modality_indices(
    indices: torch.Tensor, start: int, stop: int
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = (indices >= start) & (indices < stop)
    source = torch.nonzero(selected, as_tuple=False).flatten()
    destination = indices.index_select(0, source) - start
    return source, destination


def _modality_counts(indices: torch.Tensor, row_counts: list[int]) -> list[int]:
    boundaries = torch.tensor(
        [sum(row_counts[: index + 1]) for index in range(len(row_counts))],
        dtype=indices.dtype,
        device=indices.device,
    )
    cumulative = torch.searchsorted(indices, boundaries, right=False)
    cumulative = torch.cat([cumulative.new_zeros(1), cumulative])
    return [int(value) for value in torch.diff(cumulative).tolist()]


def _gather_uneven(value: torch.Tensor, counts: list[int], group: Any) -> torch.Tensor:
    maximum = max(counts)
    if value.shape[1] < maximum:
        value = functional.pad(value, (0, 0, 0, maximum - value.shape[1]))
    gathered = [torch.empty_like(value) for _ in counts]
    dist.all_gather(gathered, value.contiguous(), group=group)
    return torch.cat(
        [part[:, :rows] for part, rows in zip(gathered, counts, strict=True)], dim=1
    )


def _metrics(
    actual: torch.Tensor, reference: torch.Tensor, group: Any
) -> dict[str, float]:
    delta = (actual.float() - reference.float()).abs()
    values = torch.stack(
        [
            delta.max(),
            delta.mean(),
            delta.norm() / reference.float().norm().clamp_min(1e-12),
        ]
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
    return {
        "max_abs": float(values[0]),
        "max_rank_mean_abs": float(values[1]),
        "max_rank_relative_l2": float(values[2]),
    }


def _official_rope(module: Any, position_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    position_ids = position_ids.to(torch.float32)
    frequencies = position_ids.unsqueeze(-1) * module.inv_freq.view(1, 1, -1)
    time, height, width = frequencies.unbind(dim=1)
    frequencies = torch.cat((time, height, width), dim=-1)
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos(), frequencies.sin()


class RankLocalIORuntime:
    def __init__(
        self,
        *,
        group: Any,
        rank_local_inputs: bool,
        compact_output_gather: bool,
    ) -> None:
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.rank_local_inputs = rank_local_inputs
        self.compact_output_gather = compact_output_gather
        self.gate: dict[str, Any] = {}
        self._gate_complete = False

    def provenance(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "rank_local_inputs": self.rank_local_inputs,
            "compact_output_gather": self.compact_output_gather,
            "semantics": "all_rows_projected_exactly_once_then_gathered",
            "gate": self.gate,
            "origin": "FluxMatrix port of the LTX MiniMax-H3 runtime",
        }


def install_rank_local_io(
    transformer: Any,
    *,
    group: Any = None,
    rank_local_inputs: bool = True,
    compact_output_gather: bool = True,
) -> RankLocalIORuntime:
    """Replace Diffusers CP I/O hooks with explicit uneven local execution."""

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("rank-local H3 I/O requires an initialized process group")
    group = group or dist.group.WORLD
    runtime = RankLocalIORuntime(
        group=group,
        rank_local_inputs=rank_local_inputs,
        compact_output_gather=compact_output_gather,
    )
    if runtime.world < 2:
        raise RuntimeError("rank-local H3 I/O requires at least two ranks")

    from diffusers.hooks.context_parallel import remove_context_parallel

    plan = getattr(transformer, "_h3_flash_cp_plan", None)
    if plan is None:
        raise RuntimeError(
            "rank-local H3 I/O must be installed after context parallelism"
        )
    remove_context_parallel(transformer, plan)

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        attention_kwargs=None,
        return_dict=True,
    ):
        if attention_kwargs:
            raise RuntimeError(
                "rank-local H3 I/O does not support runtime LoRA scaling"
            )
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError("position_ids must have shape (sequence, 3)")
        sequence_length = position_ids.shape[0]
        if bool((token_tags < 0).any()):
            raise RuntimeError("rank-local H3 I/O requires a padless packed sequence")
        start, stop = _split_bounds(sequence_length, runtime.world, runtime.rank)
        row_counts = _split_sizes(sequence_length, runtime.world)
        local_timestep_indices = timestep_indices[start:stop].contiguous()
        local_tags = token_tags[start:stop].contiguous()
        adaln_indices = local_timestep_indices * 3 + local_tags.clamp(min=0)

        text_embeds = self.context_embedder(
            encoder_hidden_states.to(self.context_embedder.weight.dtype)
        )
        text_embeds = self.token_refiner(text_embeds)
        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))

        if runtime.rank_local_inputs:
            local = text_embeds.new_zeros(
                (text_embeds.shape[0], stop - start, text_embeds.shape[-1])
            )
            for indices, values, projection in (
                (text_indices, text_embeds, None),
                (video_indices, hidden_states, self.proj_in),
                (audio_indices, audio_hidden_states, self.audio_proj_in),
            ):
                source, destination = _local_modality_indices(indices, start, stop)
                if not source.numel():
                    continue
                selected = values.index_select(1, source)
                if projection is not None:
                    selected = projection(selected.to(projection.weight.dtype))
                local = local.index_copy(1, destination, selected.to(local.dtype))
            rotary_emb = self.rope(position_ids[start:stop])
        else:
            video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
            audio_embeds = self.audio_proj_in(
                audio_hidden_states.to(self.audio_proj_in.weight.dtype)
            )
            full = text_embeds.new_zeros(
                (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
            )
            full = full.index_copy(1, text_indices, text_embeds)
            full = full.index_copy(1, video_indices, video_embeds.to(full.dtype))
            full = full.index_copy(1, audio_indices, audio_embeds.to(full.dtype))
            local = full[:, start:stop].contiguous()
            rotary_full = self.rope(position_ids)
            rotary_emb = tuple(value[start:stop].contiguous() for value in rotary_full)

        gate_this_call = not runtime._gate_complete
        if gate_this_call and runtime.rank_local_inputs:
            video_reference = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
            audio_reference = self.audio_proj_in(
                audio_hidden_states.to(self.audio_proj_in.weight.dtype)
            )
            full_reference = text_embeds.new_zeros(
                (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
            )
            full_reference = full_reference.index_copy(1, text_indices, text_embeds)
            full_reference = full_reference.index_copy(
                1, video_indices, video_reference.to(full_reference.dtype)
            )
            full_reference = full_reference.index_copy(
                1, audio_indices, audio_reference.to(full_reference.dtype)
            )
            input_metrics = _metrics(
                local, full_reference[:, start:stop], runtime.group
            )
            rotary_reference = _official_rope(self.rope, position_ids)
            rotary_metrics = [
                _metrics(actual, reference[start:stop], runtime.group)
                for actual, reference in zip(rotary_emb, rotary_reference, strict=True)
            ]
            runtime.gate["rank_local_inputs"] = {
                "hidden_states": input_metrics,
                "rotary": rotary_metrics,
            }

        for block in self.transformer_blocks:
            local = block(local, temb, adaln_indices, rotary_emb, None)

        if runtime.compact_output_gather:

            def project(indices, head):
                _source, destination = _local_modality_indices(indices, start, stop)
                selected = local.index_select(1, destination)
                selected_timesteps = local_timestep_indices.index_select(0, destination)
                selected = self.norm_out(selected, temb, selected_timesteps)
                selected = head(selected.to(head.weight.dtype))
                counts = _modality_counts(indices, row_counts)
                return _gather_uneven(selected, counts, runtime.group)

            video_output = project(video_indices, self.proj_out)
            audio_output = project(audio_indices, self.audio_proj_out)
        else:
            full = _gather_uneven(local, row_counts, runtime.group)
            full = self.norm_out(full, temb, timestep_indices).to(
                self.proj_out.weight.dtype
            )
            video_output = self.proj_out(full).index_select(1, video_indices)
            audio_output = self.audio_proj_out(full).index_select(1, audio_indices)

        if gate_this_call and runtime.compact_output_gather:
            full_reference = _gather_uneven(local, row_counts, runtime.group)
            full_reference = self.norm_out(full_reference, temb, timestep_indices).to(
                self.proj_out.weight.dtype
            )
            video_reference = self.proj_out(full_reference).index_select(
                1, video_indices
            )
            audio_reference = self.audio_proj_out(full_reference).index_select(
                1, audio_indices
            )
            runtime.gate["compact_output_gather"] = {
                "video": _metrics(video_output, video_reference, runtime.group),
                "audio": _metrics(audio_output, audio_reference, runtime.group),
            }
        runtime._gate_complete = True

        from diffusers.models.transformers.transformer_minimax_h3 import (
            MiniMaxH3TransformerOutput,
        )

        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(
            sample=video_output, audio_sample=audio_output
        )

    # The invariant-cache wrapper installed earlier owns the fixed-schedule
    # key. Replacing it means the explicit forward must preserve that scope.
    cache_runtime = getattr(transformer, "_h3_flash_invariant_cache", None)
    if cache_runtime is not None:
        from h3_flash.runtime.cache.invariants import _timestep_key

        core = forward

        def cached_forward(self, *args, **kwargs):
            timestep = kwargs.get("timestep")
            if timestep is None:
                timestep = args[3]
            cache_runtime.current_timestep_key = _timestep_key(timestep)
            try:
                return core(self, *args, **kwargs)
            finally:
                cache_runtime.current_timestep_key = None

        transformer.forward = MethodType(cached_forward, transformer)
    else:
        transformer.forward = MethodType(forward, transformer)
    return runtime
