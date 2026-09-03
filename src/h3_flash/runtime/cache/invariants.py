"""Cache H3 values that are invariant within a request or fixed schedule."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType
from typing import Any


@dataclass
class H3InvariantCacheRuntime:
    """State owned by one resident H3 transformer instance."""

    request_values: dict[str, Any] = field(default_factory=dict)
    schedule_values: dict[str, dict[tuple[Any, ...], Any]] = field(default_factory=dict)
    current_timestep_key: tuple[Any, ...] | None = None
    request_hits: int = 0
    request_misses: int = 0
    schedule_hits: int = 0
    schedule_misses: int = 0
    adaln_modules: list[Any] = field(default_factory=list, repr=False)
    schedule_frozen: bool = False
    freed_parameter_bytes: int = 0

    def begin_request(self) -> None:
        self.request_values.clear()
        self.current_timestep_key = None
        self.request_hits = 0
        self.request_misses = 0
        self.schedule_hits = 0
        self.schedule_misses = 0

    def audit(self) -> dict[str, int]:
        return {
            "request_hits": self.request_hits,
            "request_misses": self.request_misses,
            "schedule_hits": self.schedule_hits,
            "schedule_misses": self.schedule_misses,
            "schedule_entries": sum(
                len(values) for values in self.schedule_values.values()
            ),
            "schedule_frozen": self.schedule_frozen,
            "freed_parameter_bytes": self.freed_parameter_bytes,
        }

    def freeze_fixed_schedule(self, expected_values_per_site: int = 49) -> int:
        """Drop AdaLN weights after every fixed-schedule result is cached."""

        if expected_values_per_site < 1:
            raise ValueError("expected_values_per_site must be positive")
        if self.schedule_frozen:
            return self.freed_parameter_bytes
        if not self.schedule_values or any(
            len(values) < expected_values_per_site
            for values in self.schedule_values.values()
        ):
            raise RuntimeError(
                "cannot freeze AdaLN projections before all expected schedule "
                f"values are cached ({expected_values_per_site} per site)"
            )
        freed = 0
        for module in self.adaln_modules:
            projection = module.linear
            for parameter in projection.parameters():
                freed += parameter.numel() * parameter.element_size()
            module.linear = None
        self.freed_parameter_bytes = freed
        self.schedule_frozen = True
        return freed


def _timestep_key(timestep: Any) -> tuple[Any, ...]:
    values = timestep.detach().float().cpu().reshape(-1).tolist()
    return (
        tuple(timestep.shape),
        str(timestep.dtype),
        *(float(value) for value in values),
    )


def _wrap_request_value(
    module: Any, name: str, runtime: H3InvariantCacheRuntime
) -> None:
    original = module.forward

    def forward(_module: Any, *args: Any, **kwargs: Any) -> Any:
        if name in runtime.request_values:
            runtime.request_hits += 1
            return runtime.request_values[name]
        value = original(*args, **kwargs)
        runtime.request_values[name] = value
        runtime.request_misses += 1
        return value

    module.forward = MethodType(forward, module)


def _wrap_schedule_value(
    module: Any, name: str, runtime: H3InvariantCacheRuntime
) -> None:
    original = module.forward
    values = runtime.schedule_values.setdefault(name, {})

    def forward(_module: Any, *args: Any, **kwargs: Any) -> Any:
        key = runtime.current_timestep_key
        if key is None:
            raise RuntimeError(
                f"schedule cache {name} called outside H3 transformer forward"
            )
        if key in values:
            runtime.schedule_hits += 1
            return values[key]
        if runtime.schedule_frozen:
            raise RuntimeError(
                f"fixed-schedule cache {name} received an unseen timestep after weights were released"
            )
        value = original(*args, **kwargs)
        values[key] = value
        runtime.schedule_misses += 1
        return value

    module.forward = MethodType(forward, module)


def install_h3_invariant_caches(transformer: Any) -> H3InvariantCacheRuntime:
    """Install explicit-lifetime caches without changing model weights or schedule."""

    installed = getattr(transformer, "_h3_flash_invariant_cache", None)
    if installed is not None:
        return installed

    runtime = H3InvariantCacheRuntime()
    _wrap_request_value(transformer.context_embedder, "context_embedder", runtime)
    _wrap_request_value(transformer.token_refiner, "token_refiner", runtime)
    _wrap_request_value(transformer.rope, "rope", runtime)
    for index, block in enumerate(transformer.transformer_blocks):
        _wrap_schedule_value(block.adaln_proj, f"adaln.{index}", runtime)
        runtime.adaln_modules.append(block.adaln_proj)
    _wrap_schedule_value(transformer.norm_out.linear, "norm_out.linear", runtime)

    original_forward = transformer.forward

    def forward(_module: Any, *args: Any, **kwargs: Any) -> Any:
        timestep = kwargs.get("timestep")
        if timestep is None:
            if len(args) < 4:
                raise RuntimeError("cannot resolve H3 timestep for schedule cache")
            timestep = args[3]
        runtime.current_timestep_key = _timestep_key(timestep)
        try:
            return original_forward(*args, **kwargs)
        finally:
            runtime.current_timestep_key = None

    transformer.forward = MethodType(forward, transformer)
    transformer._h3_flash_invariant_cache = runtime
    return runtime
