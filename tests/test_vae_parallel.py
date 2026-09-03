from unittest.mock import patch

import pytest

from h3_flash.runtime.parallel.vae import install_clip_parallel_video_vae


def test_clip_parallel_requires_initialized_process_group() -> None:
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        pytest.raises(RuntimeError, match="initialized process group"),
    ):
        install_clip_parallel_video_vae(object())
