"""Dense FlashAttention 4 adapter for the pinned Diffusers MiniMax H3 model."""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


def load_fa4(site_packages: Path) -> Callable[..., Any]:
    """Load the pinned Blackwell CuTe kernel without replacing Diffusers."""

    site = Path(site_packages).expanduser().resolve()
    package = site / "flash_attn" / "cute" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(f"FlashAttention 4 CuTe package not found below {site}")
    site_text = str(site)
    if site_text not in sys.path:
        sys.path.append(site_text)
    import flash_attn

    fa4_namespace = str(site / "flash_attn")
    if fa4_namespace not in flash_attn.__path__:
        flash_attn.__path__.append(fa4_namespace)
    from flash_attn.cute import flash_attn_func

    return flash_attn_func


def fa4_provenance(site_packages: Path) -> dict[str, Any]:
    """Describe the isolated FA4 distribution selected by the profile."""

    site = Path(site_packages).expanduser().resolve()
    version = None
    for distribution in distributions(path=[str(site)]):
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        if name == "flash-attn-4":
            version = distribution.version
            break
    if version is None:
        raise RuntimeError(f"flash-attn-4 metadata not found below {site}")
    return {
        "distribution": "flash-attn-4",
        "version": version,
        "site_packages": str(site),
    }


def _fa4_forward_op(
    _ctx: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = False,
    _parallel_config: Any = None,
    *,
    kernel: Callable[..., Any],
) -> Any:
    """Adapt FA4 to Diffusers' forward-only context-parallel template."""

    del _ctx, _save_ctx, _parallel_config
    if attention_mask is not None or dropout_p != 0.0 or enable_gqa or return_lse:
        raise RuntimeError("H3 dense FA4 supports only unmasked inference attention")
    output = kernel(
        query,
        key,
        value,
        softmax_scale=scale,
        causal=is_causal,
    )
    return output[0] if isinstance(output, tuple) else output


def _inference_only_backward(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("H3 dense FA4 context parallelism is inference-only")


class DenseFA4MiniMaxH3Processor:
    """Preserve full attention semantics while changing kernel evaluation order."""

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        site_packages: Path | None = None,
        *,
        kernel: Callable[..., Any] | None = None,
    ) -> None:
        if kernel is None:
            if site_packages is None:
                configured = os.environ.get("H3_FA4_SITE_PACKAGES")
                if not configured:
                    raise RuntimeError(
                        "dense FA4 requires an explicit site-packages path"
                    )
                site_packages = Path(configured)
            kernel = load_fa4(site_packages)
        self.kernel = kernel
        self.calls = 0

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        rotary_emb: tuple[Any, Any] | None = None,
        attention_mask: Any | None = None,
    ) -> Any:
        if attention_mask is not None:
            raise RuntimeError(
                "dense FA4 profile requires the official padless sequence"
            )
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if rotary_emb is not None:
            from diffusers.models.transformers.transformer_minimax_h3 import (
                _apply_rotary_emb,
            )

            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        # FA4's CuTe interface has no dropout argument. Inference is
        # deterministic with respect to dropout because the kernel implements
        # attention without a dropout path.
        if self._parallel_config is None:
            output = self.kernel(query, key, value, causal=False)
        else:
            from diffusers.models.attention_dispatch import (
                _templated_context_parallel_attention,
            )

            output = _templated_context_parallel_attention(
                query,
                key,
                value,
                None,
                0.0,
                False,
                None,
                False,
                False,
                forward_op=functools.partial(_fa4_forward_op, kernel=self.kernel),
                backward_op=_inference_only_backward,
                _parallel_config=self._parallel_config,
            )
        if isinstance(output, tuple):
            output = output[0]
        self.calls += 1
        output = output.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        return attn.to_out[1](output)


def install_dense_fa4(transformer: Any, site_packages: Path) -> dict[str, Any]:
    """Install one shared pinned kernel function on every H3 attention module."""

    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    kernel = load_fa4(site_packages)
    processors = []
    for module in transformer.modules():
        if isinstance(module, MiniMaxH3Attention):
            processor = DenseFA4MiniMaxH3Processor(kernel=kernel)
            processor._parallel_config = getattr(transformer, "_parallel_config", None)
            module.set_processor(processor)
            processors.append(processor)
    if not processors:
        raise RuntimeError("no MiniMax H3 attention modules found")
    transformer._h3_flash_dense_fa4_processors = processors
    return {
        "backend": "flash_attention_4",
        "semantics": "dense",
        "processor_count": len(processors),
        **fa4_provenance(site_packages),
    }
