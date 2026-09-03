"""Pinned Diffusers adapter for the official MiniMax H3 BF16 reference."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from h3_flash.locks import sha256_file
from h3_flash.manifest import GenerationRequest


class OfficialBackendError(RuntimeError):
    """Raised when the official reference backend contract is unavailable."""


@dataclass(frozen=True, slots=True)
class OfficialGenerationResult:
    state: Any
    load_seconds: float
    generation_seconds: float
    text_encoding_seconds: float
    diffusion_and_decode_seconds: float
    peak_gpu_memory_bytes: tuple[int, ...]
    api_num_inference_steps: int
    model_evaluations: int
    attention_audit: dict[str, Any]
    cache_audit: dict[str, int] | None


class OfficialDiffusersBackend:
    """Single-B200 BF16 reference with no distributed collectives.

    Qwen3-VL, the transformer, and both VAEs default to cuda:0. A distinct text
    device remains supported for lower-memory GPUs. The class deliberately
    does not select a custom attention backend or compile any module: those
    are later optimization dimensions, not properties of the reference.
    """

    def __init__(
        self,
        model_root: Path,
        *,
        generation_device: str = "cuda:0",
        text_device: str = "cuda:0",
        attention_backend: str = "official",
        fa4_site_packages: Path | None = None,
        fused_qkv: bool = False,
        invariant_caches: bool = False,
        ulysses_degree: int = 1,
        vae_clip_parallel: bool = False,
        transformer_fusions: bool = False,
        packed_ulysses: bool = False,
        rank_local_inputs: bool = False,
        compact_output_gather: bool = False,
        vae_compile_mode: str | None = None,
    ) -> None:
        if attention_backend not in {"official", "flash_attention_4"}:
            raise OfficialBackendError(
                f"unsupported attention backend: {attention_backend}"
            )
        if attention_backend == "flash_attention_4" and fa4_site_packages is None:
            raise OfficialBackendError(
                "flash_attention_4 requires an explicit fa4_site_packages path"
            )
        self.model_root = Path(model_root).expanduser().resolve()
        self.generation_device = generation_device
        self.text_device = text_device
        self.pipeline = None
        self.denoise_pipeline = None
        self._text_encoder_block = None
        self.load_seconds = 0.0
        self.attention_backend = attention_backend
        self.fa4_site_packages = (
            Path(fa4_site_packages).expanduser().resolve()
            if fa4_site_packages is not None
            else None
        )
        self.attention_runtime: dict[str, Any] = {
            "backend": "official",
            "semantics": "dense",
        }
        self.fused_qkv = fused_qkv
        self.invariant_caches = invariant_caches
        self.cache_runtime = None
        self.ulysses_degree = ulysses_degree
        self.vae_clip_parallel = vae_clip_parallel
        self.transformer_fusions = transformer_fusions
        self.packed_ulysses = packed_ulysses
        self.packed_ulysses_runtime = None
        self.rank_local_inputs = rank_local_inputs
        self.compact_output_gather = compact_output_gather
        self.rank_local_io_runtime = None
        self.vae_compile_mode = vae_compile_mode
        self.parallel_runtime: dict[str, Any] = {
            "backend": "none",
            "degree": 1,
        }
        self.vae_runtime: dict[str, Any] = {"backend": "official_replicated"}
        self.fusion_runtime: dict[str, Any] = {"enabled": False}

    def source_provenance(self) -> dict[str, Any]:
        """Identify the exact imported official implementation checkout."""

        self.load()
        import diffusers
        import torch
        import transformers

        module_path = Path(diffusers.__file__).resolve()
        checkout = next(
            (parent for parent in module_path.parents if (parent / ".git").exists()),
            None,
        )
        commit = None
        if checkout is not None:
            completed = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            commit = completed.stdout.strip()
        turbo_manifest_path = self.model_root / "turbo-bake.json"
        derived_model = None
        if turbo_manifest_path.is_file():
            derived_model = {
                "manifest_path": str(turbo_manifest_path),
                "manifest_sha256": sha256_file(turbo_manifest_path),
                "manifest": json.loads(turbo_manifest_path.read_text(encoding="utf-8")),
            }
        return {
            "diffusers": {
                "version": diffusers.__version__,
                "module_path": str(module_path),
                "checkout": str(checkout) if checkout is not None else None,
                "git_commit": commit,
            },
            "torch": {"version": torch.__version__},
            "transformers": {"version": transformers.__version__},
            "attention": self.attention_runtime,
            "optimizations": {
                "fused_qkv": self.fused_qkv,
                "invariant_caches": self.invariant_caches,
                "packed_ulysses": self.packed_ulysses,
                "rank_local_inputs": self.rank_local_inputs,
                "compact_output_gather": self.compact_output_gather,
            },
            "parallel": (
                self.packed_ulysses_runtime.provenance()
                if self.packed_ulysses_runtime is not None
                else self.parallel_runtime
            ),
            "video_vae_execution": self.vae_runtime,
            "transformer_fusions": self.fusion_runtime,
            "rank_local_io": (
                self.rank_local_io_runtime.provenance()
                if self.rank_local_io_runtime is not None
                else {"enabled": False}
            ),
            "derived_model": derived_model,
        }

    def load(self) -> None:
        if self.pipeline is not None:
            return
        if not (self.model_root / "modular_model_index.json").is_file():
            raise OfficialBackendError(
                f"Diffusers-layout MiniMax H3 model not found at {self.model_root}"
            )

        import torch
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines.modular_pipeline import (
            SequentialPipelineBlocks,
        )
        from transformers import Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise OfficialBackendError("official BF16 reference requires a CUDA GPU")

        started = perf_counter()
        manager = ComponentsManager()
        pipeline = ModularPipeline.from_pretrained(
            str(self.model_root), components_manager=manager
        )
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_root),
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            device_map={"": self.text_device},
            local_files_only=True,
        )
        pipeline.update_components(text_encoder=text_encoder)
        # Component specs embedded in the official index retain the Hub repo
        # identifier. Override it for every still-unloaded component so an
        # offline reference run cannot silently fetch a different revision.
        pipeline.load_components(
            dtype=torch.bfloat16,
            pretrained_model_name_or_path=str(self.model_root),
            local_files_only=True,
        )
        pipeline.transformer.to(self.generation_device)
        pipeline.vae.to(self.generation_device)
        pipeline.audio_vae.to(self.generation_device)
        if self.fused_qkv:
            pipeline.transformer.fuse_qkv_projections()
        if self.invariant_caches:
            from h3_flash.runtime.cache import install_h3_invariant_caches

            self.cache_runtime = install_h3_invariant_caches(pipeline.transformer)
        if self.attention_backend == "flash_attention_4":
            from h3_flash.runtime.attention.dense_fa4 import install_dense_fa4

            self.attention_runtime = install_dense_fa4(
                pipeline.transformer, self.fa4_site_packages
            )
        if self.transformer_fusions:
            from h3_flash.runtime.kernels import install_transformer_fusions

            self.fusion_runtime = install_transformer_fusions(pipeline.transformer)
        if self.ulysses_degree > 1:
            from h3_flash.runtime.parallel import enable_h3_ulysses

            self.parallel_runtime = enable_h3_ulysses(
                pipeline.transformer,
                degree=self.ulysses_degree,
                ulysses_anything=True,
            )
            if self.packed_ulysses:
                from h3_flash.runtime.parallel import install_packed_ulysses

                self.packed_ulysses_runtime = install_packed_ulysses(
                    pipeline.transformer,
                    fused_qknorm_rope=self.transformer_fusions,
                )
            if self.rank_local_inputs or self.compact_output_gather:
                from h3_flash.runtime.parallel import install_rank_local_io

                self.rank_local_io_runtime = install_rank_local_io(
                    pipeline.transformer,
                    rank_local_inputs=self.rank_local_inputs,
                    compact_output_gather=self.compact_output_gather,
                )
        elif self.packed_ulysses:
            raise OfficialBackendError("packed Ulysses requires ulysses_degree > 1")
        elif self.rank_local_inputs or self.compact_output_gather:
            raise OfficialBackendError("rank-local H3 I/O requires ulysses_degree > 1")
        if self.vae_clip_parallel:
            from h3_flash.runtime.parallel import install_clip_parallel_video_vae

            self.vae_runtime = install_clip_parallel_video_vae(
                pipeline.vae, compile_mode=self.vae_compile_mode
            )

        # Execute the official text encoder block separately, as Modular
        # Diffusers explicitly supports, then run the unchanged remaining
        # blocks. Besides making the stage timer explicit, this also supports
        # a distinct text device without relying on the pipeline's global
        # _execution_device inference. A device transfer does not alter values.
        self._text_encoder_block = pipeline._blocks.sub_blocks["text_encoder"]
        denoise_blocks = SequentialPipelineBlocks.from_blocks_dict(
            {
                name: block
                for name, block in pipeline._blocks.sub_blocks.items()
                if name != "text_encoder"
            }
        )
        denoise_pipeline = type(pipeline)(blocks=denoise_blocks)
        denoise_pipeline.update_components(
            **{
                name: getattr(pipeline, name)
                for name in denoise_pipeline.component_names
                if hasattr(pipeline, name)
            }
        )
        self.pipeline = pipeline
        self.denoise_pipeline = denoise_pipeline
        torch.cuda.synchronize(self.generation_device)
        torch.cuda.synchronize(self.text_device)
        self.load_seconds = perf_counter() - started

    def generate(
        self,
        request: GenerationRequest,
        *,
        api_num_inference_steps: int = 50,
        output_type: str = "pt",
        first_frame: Any | None = None,
    ) -> OfficialGenerationResult:
        request.validate()
        if (request.mode == "i2va") != (first_frame is not None):
            raise OfficialBackendError(
                "i2va requires a first frame and t2va does not accept one"
            )
        if api_num_inference_steps < 2:
            raise OfficialBackendError(
                "official sigma grid requires at least two points"
            )
        if output_type not in {"latent", "pt", "np", "pil"}:
            raise OfficialBackendError(f"unsupported output_type: {output_type}")
        self.load()

        import torch

        keyframes = None
        if first_frame is not None:
            from PIL import ImageOps

            from diffusers.modular_pipelines.minimax_h3.packing import (
                prepare_keyframe_image,
            )

            first_frame = ImageOps.exif_transpose(first_frame).convert("RGB")
            keyframes = [
                prepare_keyframe_image(
                    first_frame,
                    request.height,
                    request.width,
                    stretch=True,
                )
            ]

        if self.cache_runtime is not None:
            self.cache_runtime.begin_request()
        if self.packed_ulysses_runtime is not None:
            self.packed_ulysses_runtime.begin_request()

        active_indices: list[int] = []
        for device_name in (self.generation_device, self.text_device):
            device = torch.device(device_name)
            if device.type != "cuda":
                continue
            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            if index not in active_indices:
                active_indices.append(index)
        for index in active_indices:
            torch.cuda.reset_peak_memory_stats(index)
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        torch.cuda.synchronize(self.generation_device)
        torch.cuda.synchronize(self.text_device)
        started = perf_counter()
        attention_calls_before = sum(
            processor.calls
            for processor in getattr(
                self.pipeline.transformer, "_h3_flash_dense_fa4_processors", ()
            )
        )
        packed_calls_before = (
            self.packed_ulysses_runtime.attention_calls
            if self.packed_ulysses_runtime is not None
            else 0
        )
        with torch.inference_mode():
            prompt_embeds, text_token_tags = self._text_encoder_block.encode_prompt(
                self.pipeline,
                request.prompt,
                keyframes,
                device=torch.device(self.text_device),
                dtype=torch.bfloat16,
            )
            prompt_embeds = prompt_embeds.to(self.generation_device)
            torch.cuda.synchronize(self.generation_device)
            torch.cuda.synchronize(self.text_device)
            text_encoding_seconds = perf_counter() - started
            diffusion_started = perf_counter()
            state = self.denoise_pipeline(
                prompt_embeds=prompt_embeds,
                text_token_tags=text_token_tags,
                generator=generator,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                num_inference_steps=api_num_inference_steps,
                output_type=output_type,
                image=first_frame,
            )
        torch.cuda.synchronize(self.generation_device)
        torch.cuda.synchronize(self.text_device)
        diffusion_and_decode_seconds = perf_counter() - diffusion_started
        generation_seconds = perf_counter() - started
        processors = getattr(
            self.pipeline.transformer, "_h3_flash_dense_fa4_processors", ()
        )
        attention_calls_after = sum(processor.calls for processor in processors)
        packed_calls_after = (
            self.packed_ulysses_runtime.attention_calls
            if self.packed_ulysses_runtime is not None
            else 0
        )
        # Preserve GPU-index addressing in the report without creating CUDA
        # contexts on devices that this backend does not use.
        peak = [0] * torch.cuda.device_count()
        for index in active_indices:
            peak[index] = torch.cuda.max_memory_allocated(index)
        return OfficialGenerationResult(
            state=state,
            load_seconds=self.load_seconds,
            generation_seconds=generation_seconds,
            text_encoding_seconds=text_encoding_seconds,
            diffusion_and_decode_seconds=diffusion_and_decode_seconds,
            peak_gpu_memory_bytes=tuple(peak),
            api_num_inference_steps=api_num_inference_steps,
            model_evaluations=api_num_inference_steps - 1,
            attention_audit={
                **(
                    self.packed_ulysses_runtime.provenance()
                    if self.packed_ulysses_runtime is not None
                    else self.attention_runtime
                ),
                "calls_this_request": attention_calls_after - attention_calls_before,
                "packed_calls_this_request": packed_calls_after - packed_calls_before,
            },
            cache_audit=(
                self.cache_runtime.audit() if self.cache_runtime is not None else None
            ),
        )

    def prepare_fixed_schedule(
        self,
        request: GenerationRequest,
        *,
        api_num_inference_steps: int = 50,
        first_frame: Any | None = None,
        freeze: bool = True,
        expected_values_per_site: int | None = None,
    ) -> dict[str, Any]:
        """Precompute exact fixed-schedule cache values outside request latency."""

        if self.cache_runtime is None:
            raise OfficialBackendError("invariant caches are not enabled")
        started = perf_counter()
        generation = self.generate(
            request,
            api_num_inference_steps=api_num_inference_steps,
            output_type="latent",
            first_frame=first_frame,
        )
        freed = 0
        if freeze:
            freed = self.cache_runtime.freeze_fixed_schedule(
                expected_values_per_site=(
                    expected_values_per_site or generation.model_evaluations
                )
            )
        import torch

        torch.cuda.empty_cache()
        return {
            "seconds": perf_counter() - started,
            "entries": self.cache_runtime.audit()["schedule_entries"],
            "model_evaluations": generation.model_evaluations,
            "freed_parameter_bytes": freed,
        }
