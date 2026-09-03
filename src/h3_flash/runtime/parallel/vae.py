"""Lossless clip-parallel decoding for the official MiniMax H3 video VAE."""

from __future__ import annotations

from typing import Any


def install_clip_parallel_video_vae(
    vae: Any,
    *,
    group: Any = None,
    batched: bool = True,
    compile_mode: str | None = None,
) -> dict[str, Any]:
    """Distribute independent official decoder tiles and restore tile order.

    Every original tile is decoded exactly once by the unchanged decoder.  A
    fixed-size all-gather restores the original row-major tile list before the
    official overlap stitcher runs.  Equal-size tiles may be batched because
    the decoder has no operation that reduces across its batch dimension.
    """

    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("clip-parallel VAE requires an initialized process group")
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if world_size < 2:
        raise RuntimeError("clip-parallel VAE requires at least two ranks")
    if getattr(vae, "_h3_flash_clip_parallel", None) is not None:
        return vae._h3_flash_clip_parallel

    original = vae._decode_clip
    eager_decoder = vae.decoder
    if compile_mode is not None:
        vae.decoder = torch.compile(eager_decoder, mode=compile_mode, dynamic=False)
    numerical_gate: dict[str, float] = {}

    def sharded_decode_clip(z: torch.Tensor) -> torch.Tensor:
        if not vae.use_tiling:
            return original(z)

        height = z.shape[-2] * vae.spatial_compression_ratio
        width = z.shape[-1] * vae.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = vae._split_tiles(
            height,
            vae.tile_sample_min_height,
            vae.tile_sample_min_overlap_height,
        )
        x_indices, x_lengths, x_overlaps = vae._split_tiles(
            width,
            vae.tile_sample_min_width,
            vae.tile_sample_min_overlap_width,
        )
        ratio = vae.spatial_compression_ratio
        coordinates = [
            (y_indices[y], y_lengths[y], x_indices[x], x_lengths[x])
            for y in range(len(y_indices))
            for x in range(len(x_indices))
        ]
        per_rank = (len(coordinates) + world_size - 1) // world_size
        padded = coordinates + [coordinates[-1]] * (
            per_rank * world_size - len(coordinates)
        )
        mine = padded[rank * per_rank : (rank + 1) * per_rank]
        slices = [
            z[
                ...,
                y // ratio : y // ratio + y_length // ratio,
                x // ratio : x // ratio + x_length // ratio,
            ]
            for y, y_length, x, x_length in mine
        ]
        if batched:
            decoder_input = vae.post_quant_conv(torch.cat(slices, dim=0))
            local_tiles = vae.decoder(decoder_input)
            if compile_mode is not None and not numerical_gate:
                eager_tiles = eager_decoder(decoder_input)
                delta = (local_tiles.float() - eager_tiles.float()).abs()
                values = torch.stack(
                    [
                        delta.max(),
                        delta.mean(),
                        delta.norm() / eager_tiles.float().norm().clamp_min(1e-12),
                    ]
                )
                dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
                numerical_gate.update(
                    max_abs=float(values[0]),
                    max_rank_mean_abs=float(values[1]),
                    max_rank_relative_l2=float(values[2]),
                )
        else:
            local_tiles = torch.stack(
                [vae.decoder(vae.post_quant_conv(value)) for value in slices], dim=0
            ).squeeze(1)

        gathered = [torch.empty_like(local_tiles) for _ in range(world_size)]
        dist.all_gather(gathered, local_tiles, group=group)
        ordered = torch.cat(gathered, dim=0)[: len(coordinates)]
        rows = []
        offset = 0
        for _ in y_indices:
            row = []
            for _ in x_indices:
                row.append(ordered[offset].unsqueeze(0))
                offset += 1
            rows.append(row)
        return vae._stitch_tiles(rows, y_overlaps, x_overlaps)

    vae._decode_clip = sharded_decode_clip
    metadata = {
        "backend": "official_tile_clip_parallel",
        "world_size": world_size,
        "batched_equal_tiles": batched,
        "compile_mode": compile_mode,
        "compile_contract": (
            "mathematically_equivalent_bounded_numeric_error"
            if compile_mode is not None
            else "eager_bit_exact_tile_execution"
        ),
        "compile_gate": numerical_gate,
    }
    vae._h3_flash_clip_parallel = metadata
    return metadata
