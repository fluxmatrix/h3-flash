from pathlib import Path

import pytest

from h3_flash.runtime.attention.dense_fa4 import (
    DenseFA4MiniMaxH3Processor,
    load_fa4,
)


def test_missing_fa4_package_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="CuTe package"):
        load_fa4(tmp_path)


def test_dense_processor_preserves_shape_and_full_sequence() -> None:
    torch = pytest.importorskip("torch")

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = 2
            self.fused_projections = False
            self.to_q = torch.nn.Linear(8, 8, bias=False)
            self.to_k = torch.nn.Linear(8, 8, bias=False)
            self.to_v = torch.nn.Linear(8, 8, bias=False)
            self.norm_q = torch.nn.Identity()
            self.norm_k = torch.nn.Identity()
            self.to_out = torch.nn.ModuleList(
                [torch.nn.Linear(8, 8, bias=False), torch.nn.Identity()]
            )

    seen = {}

    def kernel(q, k, v, **kwargs):
        seen["shapes"] = (q.shape, k.shape, v.shape)
        seen["kwargs"] = kwargs
        return v

    processor = DenseFA4MiniMaxH3Processor(kernel=kernel)
    result = processor(Attention(), torch.randn(1, 7, 8))

    assert result.shape == (1, 7, 8)
    assert seen["shapes"] == ((1, 7, 2, 4),) * 3
    assert seen["kwargs"] == {"causal": False}
    assert processor.calls == 1


def test_dense_processor_rejects_masked_sequence() -> None:
    processor = DenseFA4MiniMaxH3Processor(kernel=lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="padless"):
        processor(None, None, attention_mask=object())
