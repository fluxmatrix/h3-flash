import pytest

from h3_flash.runtime.media import HostTransferPool


def test_cpu_values_preserve_identity_and_non_tensors_pass_through() -> None:
    torch = pytest.importorskip("torch")
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    pool = HostTransferPool()

    result = pool.copy({"video": source, "sampling_rate": 32000})

    assert torch.equal(result["video"], source)
    assert result["sampling_rate"] == 32000
    assert pool.buffer_count == 0


def test_pool_signature_includes_shape_stride_and_dtype() -> None:
    torch = pytest.importorskip("torch")
    pool = HostTransferPool()
    contiguous = torch.empty((2, 3), dtype=torch.float32)
    transposed = torch.empty((3, 2), dtype=torch.float32).t()

    assert pool._signature(contiguous) != pool._signature(transposed)
