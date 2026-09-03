"""Low-overhead MP4 output through an FFmpeg raw-video pipe."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any


def _stereo_pcm16(samples: Any) -> Any:
    """Return contiguous CPU ``[samples, 2]`` signed PCM16 audio."""

    import torch

    if not isinstance(samples, torch.Tensor):
        samples = torch.as_tensor(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2:
        raise ValueError(f"expected one- or two-dimensional audio, got {samples.shape}")
    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T
    if samples.shape[1] != 2:
        raise ValueError(f"expected stereo audio, got {samples.shape}")
    if samples.dtype != torch.int16:
        samples = (samples.float().clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
    return samples.detach().cpu().contiguous()


def _write_wave(path: Path, samples: Any, sample_rate: int) -> None:
    pcm = _stereo_pcm16(samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(memoryview(pcm.numpy()).cast("B"))


def encode_video_ffmpeg(
    video: Any,
    *,
    fps: int,
    output_path: Path,
    audio: Any,
    audio_sample_rate: int,
    ffmpeg_path: str | None = None,
    preset: str = "veryfast",
    crf: int = 20,
    video_codec: str = "libx264",
    pixel_format: str = "yuv420p",
    audio_codec: str = "aac",
) -> dict[str, Any]:
    """Encode ``[T,H,W,C]`` uint8 RGB plus stereo audio with one raw pipe."""

    import torch

    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise TypeError("FFmpeg output expects a [T,H,W,C] torch.Tensor")
    if video.shape[-1] != 3 or video.dtype != torch.uint8:
        raise TypeError(f"expected uint8 RGB video, got {video.dtype} {video.shape}")
    video = video.detach().cpu().contiguous()
    frames, height, width, _channels = video.shape
    if video_codec != "libx264" or pixel_format != "yuv420p" or audio_codec != "aac":
        raise ValueError(
            "the pinned H3-Flash output contract requires "
            "video_codec=libx264, pixel_format=yuv420p, audio_codec=aac"
        )

    configured = ffmpeg_path or os.environ.get("H3_FLASH_FFMPEG_BIN", "ffmpeg")
    resolved = shutil.which(configured)
    if resolved is None:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            resolved = str(candidate.resolve())
        else:
            raise FileNotFoundError(f"FFmpeg binary not found: {configured}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.audio.",
        suffix=".wav",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_audio = Path(temporary_name)
    try:
        _write_wave(temporary_audio, audio, int(audio_sample_rate))
        command = [
            resolved,
            "-n",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(int(fps)),
            "-i",
            "pipe:0",
            "-i",
            str(temporary_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-pix_fmt",
            pixel_format,
            "-crf",
            str(int(crf)),
            "-c:a",
            audio_codec,
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        environment = os.environ.copy()
        ffmpeg_library = Path(resolved).parent.parent / "lib"
        if ffmpeg_library.is_dir():
            old_path = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = (
                str(ffmpeg_library) if not old_path else f"{ffmpeg_library}:{old_path}"
            )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=environment,
        )
        try:
            assert process.stdin is not None
            for chunk in torch.tensor_split(video, min(16, int(frames)), dim=0):
                process.stdin.write(memoryview(chunk.numpy()).cast("B"))
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
            if return_code != 0:
                detail = stderr.decode(errors="replace")[-2000:]
                raise RuntimeError(
                    f"FFmpeg video encode failed with code {return_code}: {detail}"
                )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    finally:
        temporary_audio.unlink(missing_ok=True)

    return {
        "backend": "ffmpeg_raw_pipe",
        "binary": resolved,
        "video_codec": video_codec,
        "preset": preset,
        "crf": int(crf),
        "pixel_format": pixel_format,
        "audio_codec": audio_codec,
        "audio_termination": "full_source_with_encoder_padding",
        "streaming": "faststart",
    }


__all__ = ["encode_video_ffmpeg"]
