import pytest

from h3_flash.runtime.cache import H3InvariantCacheRuntime
from h3_flash.runtime.cache.invariants import (
    _timestep_key,
    _wrap_request_value,
    _wrap_schedule_value,
)


class Counter:
    def __init__(self) -> None:
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value + self.calls


def test_request_cache_is_reused_then_cleared() -> None:
    runtime = H3InvariantCacheRuntime()
    module = Counter()
    _wrap_request_value(module, "value", runtime)

    assert module.forward(10) == 11
    assert module.forward(99) == 11
    runtime.begin_request()
    assert module.forward(20) == 22
    assert runtime.audit()["request_misses"] == 1


def test_schedule_cache_reuses_equal_timestep_key_across_requests() -> None:
    runtime = H3InvariantCacheRuntime()
    module = Counter()
    _wrap_schedule_value(module, "adaln", runtime)
    runtime.current_timestep_key = (0.5,)

    assert module.forward(10) == 11
    assert module.forward(99) == 11
    runtime.begin_request()
    runtime.current_timestep_key = (0.5,)
    assert module.forward(50) == 11
    runtime.current_timestep_key = (0.25,)
    assert module.forward(50) == 52


def test_schedule_cache_fails_closed_without_transformer_context() -> None:
    runtime = H3InvariantCacheRuntime()
    module = Counter()
    _wrap_schedule_value(module, "adaln", runtime)
    with pytest.raises(RuntimeError, match="outside H3 transformer"):
        module.forward(1)


def test_timestep_key_includes_values_shape_and_dtype() -> None:
    torch = pytest.importorskip("torch")
    a = torch.tensor([0.5], dtype=torch.float32)
    b = torch.tensor([0.5], dtype=torch.float64)
    c = torch.tensor([0.25], dtype=torch.float32)
    assert _timestep_key(a) != _timestep_key(b)
    assert _timestep_key(a) != _timestep_key(c)


def test_fixed_schedule_freeze_releases_adaln_projection_and_fails_closed() -> None:
    torch = pytest.importorskip("torch")

    class Holder:
        def __init__(self) -> None:
            self.linear = torch.nn.Linear(2, 3)

    runtime = H3InvariantCacheRuntime()
    holder = Holder()
    expected_bytes = sum(
        value.numel() * value.element_size() for value in holder.linear.parameters()
    )
    runtime.adaln_modules.append(holder)
    runtime.schedule_values["adaln"] = {(index,): index for index in range(49)}
    assert runtime.freeze_fixed_schedule() == expected_bytes
    assert holder.linear is None
    assert runtime.audit()["schedule_frozen"] is True

    module = Counter()
    _wrap_schedule_value(module, "new", runtime)
    runtime.current_timestep_key = (100,)
    with pytest.raises(RuntimeError, match="unseen timestep"):
        module.forward(1)


def test_fixed_schedule_freeze_accepts_declared_few_step_count() -> None:
    torch = pytest.importorskip("torch")

    class Holder:
        def __init__(self) -> None:
            self.linear = torch.nn.Linear(2, 3)

    runtime = H3InvariantCacheRuntime()
    holder = Holder()
    expected_bytes = sum(
        value.numel() * value.element_size() for value in holder.linear.parameters()
    )
    runtime.adaln_modules.append(holder)
    runtime.schedule_values["adaln"] = {(index,): index for index in range(4)}

    assert runtime.freeze_fixed_schedule(expected_values_per_site=4) == expected_bytes
    assert holder.linear is None
