"""Bit-preserving persistent pinned GPU-to-host transfers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class HostTransferPool:
    """Reuse pinned host tensors for repeated fixed-shape outputs.

    Returned tensors are owned by the pool and may be overwritten by the next
    call using the same logical key. Consumers must finish encoding before that
    key is reused. Shape, stride and dtype are part of the cache signature.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, Any] = {}

    @staticmethod
    def _signature(tensor: Any) -> tuple[Any, ...]:
        return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)

    def _buffer(self, key: str, source: Any) -> Any:
        import torch

        current = self._buffers.get(key)
        if current is None or self._signature(current) != self._signature(source):
            current = torch.empty_strided(
                tuple(source.shape),
                tuple(source.stride()),
                dtype=source.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._buffers[key] = current
        return current

    def copy(self, tensors: Mapping[str, Any]) -> dict[str, Any]:
        """Copy tensors to host and synchronize once per CUDA device."""

        import torch

        outputs: dict[str, Any] = {}
        devices: set[Any] = set()
        for key, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                outputs[key] = value
                continue
            source = value.detach()
            if source.device.type != "cuda":
                outputs[key] = source.cpu()
                continue
            destination = self._buffer(key, source)
            destination.copy_(source, non_blocking=True)
            devices.add(source.device)
            outputs[key] = destination
        for device in devices:
            torch.cuda.current_stream(device).synchronize()
        return outputs

    def clear(self) -> None:
        self._buffers.clear()

    @property
    def buffer_count(self) -> int:
        return len(self._buffers)
