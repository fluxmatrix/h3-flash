import torch

from h3_flash.run import _prepare_video_on_device, _video_to_uint8


def test_prepare_video_selects_batch_and_matches_reference_cpu_layout() -> None:
    video = torch.tensor([[[[[0.0, 0.5], [1.0, 0.25]]]]], dtype=torch.float32)
    expected = _video_to_uint8(video[0])
    actual = _prepare_video_on_device(video)
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.uint8
