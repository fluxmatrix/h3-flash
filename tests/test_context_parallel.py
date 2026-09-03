from unittest.mock import Mock, patch

import pytest

from h3_flash.runtime.parallel.context import enable_h3_ulysses


def test_ulysses_requires_initialized_process_group() -> None:
    transformer = Mock()
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        pytest.raises(RuntimeError, match="initialized torch process group"),
    ):
        enable_h3_ulysses(transformer, degree=8)
