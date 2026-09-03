"""Small resident web service for the H3-Flash browser demo."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import queue
import secrets
import signal
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any
from urllib.parse import parse_qs, urlparse

from .backends.official_diffusers import OfficialDiffusersBackend
from .distributed_benchmark import _maximum_across_ranks, _peaks_across_ranks
from .locks import LockRepository, sha256_file
from .manifest import GenerationRequest, build_manifest
from .models import require_profile_model_verification
from .profiles import ProfileRepository
from .prompting import (
    OpenAICompatiblePromptEnhancer,
    PromptEnhancementError,
    make_i2va_prompt,
)
from .run import _prepare_video_on_device, _write_json_atomic
from .runtime.media import HostTransferPool, encode_video_ffmpeg


MAX_PROMPT_BYTES = 32_768
MAX_FIRST_FRAME_BYTES = 12 * 1024 * 1024
MAX_REQUEST_BYTES = 17 * 1024 * 1024
MAX_STORY_HISTORY = 8
MAX_STORY_CONTEXT_BYTES = 32_768
STORY_CONTEXT_FRAME_COUNT = 4
STORY_CONTEXT_WINDOW_SECONDS = 2
FIRST_FRAME_DATA_HEADERS = {
    "data:image/jpeg;base64",
    "data:image/png;base64",
    "data:image/webp;base64",
}
FIRST_FRAME_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
WEB_RESOLUTION_PRESETS = {
    "480p-landscape": (832, 480),
    "480p-portrait": (480, 832),
    "768p-landscape": (1344, 768),
    "768p-portrait": (768, 1344),
}
WEB_DURATION_PRESETS = {5: 124, 10: 243, 15: 362}


class ServiceError(ValueError):
    """Raised when a browser request violates the service contract."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def validate_job_payload(
    payload: Any, *, has_first_frame: bool | None = None
) -> GenerationRequest:
    if not isinstance(payload, dict):
        raise ServiceError("request body must be a JSON object")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ServiceError("prompt must be a non-empty string")
    prompt = prompt.strip()
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ServiceError(f"prompt must be at most {MAX_PROMPT_BYTES} UTF-8 bytes")
    seed = payload.get("seed")
    if seed in (None, ""):
        seed = secrets.randbelow(2**31)
    if isinstance(seed, bool):
        raise ServiceError("seed must be an integer")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as error:
        raise ServiceError("seed must be an integer") from error
    if not 0 <= seed < 2**63:
        raise ServiceError("seed must be between 0 and 2^63-1")
    resolution = payload.get("resolution", "768p-landscape")
    if resolution not in WEB_RESOLUTION_PRESETS:
        choices = ", ".join(WEB_RESOLUTION_PRESETS)
        raise ServiceError(f"resolution must be one of: {choices}")
    duration = payload.get("duration", 5)
    if isinstance(duration, bool):
        raise ServiceError("duration must be 5, 10, or 15 seconds")
    try:
        duration = int(duration)
    except (TypeError, ValueError) as error:
        raise ServiceError("duration must be 5, 10, or 15 seconds") from error
    if duration not in WEB_DURATION_PRESETS:
        raise ServiceError("duration must be 5, 10, or 15 seconds")
    width, height = WEB_RESOLUTION_PRESETS[resolution]
    if has_first_frame is None:
        has_first_frame = payload.get("first_frame") not in (None, "")
    if has_first_frame:
        prompt = make_i2va_prompt(prompt)
    return GenerationRequest(
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        num_frames=WEB_DURATION_PRESETS[duration],
        fps=24,
        mode="i2va" if has_first_frame else "t2va",
    )


def validate_first_frame_bytes(
    image_bytes: bytes, *, media_type: str | None = None
) -> bytes:
    if media_type not in (None, "application/octet-stream") and (
        media_type.lower() not in FIRST_FRAME_MIME_TYPES
    ):
        raise ServiceError("first_frame must be a JPEG, PNG, or WebP image")
    if not image_bytes or len(image_bytes) > MAX_FIRST_FRAME_BYTES:
        raise ServiceError("first_frame must be at most 12 MiB")
    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            image_format = image.format
            image.verify()
    except Exception as error:
        raise ServiceError("first_frame is not a valid image") from error
    if image_format not in {"JPEG", "PNG", "WEBP"}:
        raise ServiceError("first_frame must be a JPEG, PNG, or WebP image")
    if width < 32 or height < 32 or width > 8192 or height > 8192:
        raise ServiceError("first_frame dimensions must be between 32 and 8192 pixels")
    if width * height > 50_000_000:
        raise ServiceError("first_frame must contain at most 50 megapixels")
    return image_bytes


def validate_first_frame(payload: Any) -> bytes | None:
    if not isinstance(payload, dict):
        raise ServiceError("request body must be a JSON object")
    value = payload.get("first_frame")
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ServiceError("first_frame must be a JPEG, PNG, or WebP data URL")
    header, separator, encoded = value.partition(",")
    if separator != "," or header.lower() not in FIRST_FRAME_DATA_HEADERS:
        raise ServiceError("first_frame must be a JPEG, PNG, or WebP data URL")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ServiceError("first_frame contains invalid base64 data") from error
    media_type = header.removeprefix("data:").removesuffix(";base64").lower()
    return validate_first_frame_bytes(image_bytes, media_type=media_type)


def parse_job_request(
    content_type: str, body: bytes
) -> tuple[GenerationRequest, bytes | None]:
    """Parse JSON or browser multipart jobs without base64-expanding start frames."""

    if content_type.partition(";")[0].strip().lower() == "application/json":
        payload = json.loads(body)
        return validate_job_payload(payload), validate_first_frame(payload)
    if not content_type.lower().startswith("multipart/form-data;"):
        raise ServiceError(
            "job Content-Type must be application/json or multipart/form-data"
        )

    envelope = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
        + body
    )
    message = BytesParser(policy=policy.default).parsebytes(envelope)
    if not message.is_multipart():
        raise ServiceError("invalid multipart job body")
    request_payload: Any = None
    first_frame: bytes | None = None
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        value = part.get_payload(decode=True) or b""
        if name == "request":
            if request_payload is not None:
                raise ServiceError("multipart job must contain one request field")
            request_payload = json.loads(value)
        elif name == "first_frame":
            if first_frame is not None:
                raise ServiceError("multipart job must contain at most one first frame")
            first_frame = validate_first_frame_bytes(
                value, media_type=part.get_content_type()
            )
    if request_payload is None:
        raise ServiceError("multipart job must contain a JSON request field")
    request = validate_job_payload(
        request_payload, has_first_frame=first_frame is not None
    )
    return request, first_frame


def parse_byte_range(value: str | None, size: int) -> tuple[int, int, bool]:
    """Return an inclusive byte range and whether the response is partial."""

    if size <= 0:
        raise ServiceError("video is empty")
    if value is None:
        return 0, size - 1, False
    unit, separator, requested = value.partition("=")
    if separator != "=" or unit.strip().lower() != "bytes" or "," in requested:
        raise ServiceError("invalid Range header")
    start_text, dash, end_text = requested.strip().partition("-")
    if dash != "-" or (not start_text and not end_text):
        raise ServiceError("invalid Range header")
    try:
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_text)
            end = size - 1 if not end_text else int(end_text)
    except ValueError as error:
        raise ServiceError("invalid Range header") from error
    if start < 0 or start >= size or end < start:
        raise ServiceError("Range is outside the video")
    return start, min(end, size - 1), True


def validate_prompt_enhancement_payload(payload: Any) -> tuple[str, int, str]:
    if not isinstance(payload, dict):
        raise ServiceError("request body must be a JSON object")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ServiceError("prompt must be a non-empty string")
    prompt = prompt.strip()
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ServiceError(f"prompt must be at most {MAX_PROMPT_BYTES} UTF-8 bytes")
    duration = payload.get("duration", 5)
    if isinstance(duration, bool):
        raise ServiceError("duration must be 5, 10, or 15 seconds")
    try:
        duration = int(duration)
    except (TypeError, ValueError) as error:
        raise ServiceError("duration must be 5, 10, or 15 seconds") from error
    if duration not in WEB_DURATION_PRESETS:
        raise ServiceError("duration must be 5, 10, or 15 seconds")
    mode = payload.get("mode", "t2va")
    if mode not in {"t2va", "i2va"}:
        raise ServiceError("mode must be t2va or i2va")
    return prompt, duration, mode


def validate_story_opening_payload(payload: Any) -> tuple[str, int, str]:
    if not isinstance(payload, dict):
        raise ServiceError("request body must be a JSON object")
    premise = payload.get("premise")
    if not isinstance(premise, str) or not premise.strip():
        raise ServiceError("story premise must be a non-empty string")
    premise = premise.strip()
    if len(premise.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ServiceError(
            f"story premise must be at most {MAX_PROMPT_BYTES} UTF-8 bytes"
        )
    duration = payload.get("duration", 10)
    if isinstance(duration, bool):
        raise ServiceError("duration must be 5, 10, or 15 seconds")
    try:
        duration = int(duration)
    except (TypeError, ValueError) as error:
        raise ServiceError("duration must be 5, 10, or 15 seconds") from error
    if duration not in WEB_DURATION_PRESETS:
        raise ServiceError("duration must be 5, 10, or 15 seconds")
    language = payload.get("language", "en")
    display_languages = {"en": "English", "zh-CN": "Simplified Chinese"}
    if language not in display_languages:
        raise ServiceError("story language must be en or zh-CN")
    return premise, duration, display_languages[language]


def validate_story_payload(
    payload: Any,
) -> tuple[
    str,
    str,
    list[str],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    int,
    str,
    str,
]:
    if not isinstance(payload, dict):
        raise ServiceError("request body must be a JSON object")
    premise = payload.get("premise")
    current_prompt = payload.get("current_prompt")
    if not isinstance(premise, str) or not premise.strip():
        raise ServiceError("story premise must be a non-empty string")
    if not isinstance(current_prompt, str) or not current_prompt.strip():
        raise ServiceError("current_prompt must be a non-empty string")
    premise = premise.strip()
    current_prompt = current_prompt.strip()
    history = payload.get("history", [])
    if not isinstance(history, list) or len(history) > MAX_STORY_HISTORY:
        raise ServiceError(
            f"story history must contain at most {MAX_STORY_HISTORY} choices"
        )
    if not all(isinstance(choice, str) and choice.strip() for choice in history):
        raise ServiceError("each story history choice must be a non-empty string")
    history = [choice.strip() for choice in history]
    story_bible = payload.get("story_bible")
    story_arc = payload.get("story_arc")
    story_state = payload.get("story_state")
    if not isinstance(story_bible, dict):
        raise ServiceError("story_bible must be a JSON object")
    if (
        not isinstance(story_arc, list)
        or len(story_arc) != 5
        or not all(isinstance(beat, dict) for beat in story_arc)
    ):
        raise ServiceError("story_arc must contain five chapter objects")
    if not isinstance(story_state, dict):
        raise ServiceError("story_state must be a JSON object")
    duration = payload.get("duration", 10)
    if isinstance(duration, bool):
        raise ServiceError("duration must be 5, 10, or 15 seconds")
    try:
        duration = int(duration)
    except (TypeError, ValueError) as error:
        raise ServiceError("duration must be 5, 10, or 15 seconds") from error
    if duration not in WEB_DURATION_PRESETS:
        raise ServiceError("duration must be 5, 10, or 15 seconds")
    language = payload.get("language", "en")
    display_languages = {"en": "English", "zh-CN": "Simplified Chinese"}
    if language not in display_languages:
        raise ServiceError("story language must be en or zh-CN")
    source_job_id = payload.get("source_job_id")
    if (
        not isinstance(source_job_id, str)
        or len(source_job_id) != 32
        or any(character not in "0123456789abcdef" for character in source_job_id)
    ):
        raise ServiceError("source_job_id must be a 32-character lowercase hex job ID")
    encoded_size = len(
        json.dumps(
            [premise, current_prompt, history, story_bible, story_arc, story_state],
            ensure_ascii=False,
        ).encode("utf-8")
    )
    if encoded_size > MAX_STORY_CONTEXT_BYTES:
        raise ServiceError(
            f"story context must be at most {MAX_STORY_CONTEXT_BYTES} UTF-8 bytes"
        )
    return (
        premise,
        current_prompt,
        history,
        story_bible,
        story_arc,
        story_state,
        duration,
        display_languages[language],
        source_job_id,
    )


def story_context_frame_indices(
    frame_count: int,
    fps: int,
    *,
    sample_count: int = STORY_CONTEXT_FRAME_COUNT,
    window_seconds: int = STORY_CONTEXT_WINDOW_SECONDS,
) -> tuple[int, ...]:
    """Choose chronological samples ending on the exact final decoded frame."""

    if frame_count < sample_count or fps <= 0 or sample_count < 2:
        raise ServiceError("video is too short for story context sampling")
    final_index = frame_count - 1
    first_index = max(0, final_index - fps * window_seconds)
    span = final_index - first_index
    return tuple(
        first_index + round(span * index / (sample_count - 1))
        for index in range(sample_count)
    )


@dataclass(slots=True)
class GenerationJob:
    job_id: str
    prompt: str
    seed: int
    width: int
    height: int
    num_frames: int
    fps: int
    mode: str
    first_frame_path: str | None
    first_frame_sha256: str | None
    first_shape_warmup: bool
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    generation_seconds: float | None = None
    request_seconds: float | None = None
    video_url: str | None = None
    last_frame_url: str | None = None
    error: str | None = None

    def public(self, queue_position: int | None = None) -> dict[str, Any]:
        result = asdict(self)
        result.pop("first_frame_path")
        result["has_first_frame"] = self.first_frame_path is not None
        if queue_position is not None:
            result["queue_position"] = queue_position
        return result


class ServiceState:
    """Thread-safe service readiness and FIFO generation queue."""

    def __init__(
        self,
        *,
        profile: str,
        max_jobs: int = 100,
        prompt_enhancer: OpenAICompatiblePromptEnhancer | None = None,
        job_root: Path | None = None,
    ) -> None:
        self.profile = profile
        self.max_jobs = max_jobs
        self.ready = False
        self.phase = "loading"
        self.startup: dict[str, Any] = {}
        self._jobs: dict[str, GenerationJob] = {}
        self.prompt_enhancer = prompt_enhancer
        self.job_root = job_root
        self._prompt_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
        self._story_opening_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
        self._story_cache: dict[
            tuple[str, str, tuple[str, ...], str, int, str, str], dict[str, Any]
        ] = {}
        self._prompt_lock = threading.Lock()
        # next_job() uses a short timed wait so rank 0 can observe a shutdown
        # flag set by the signal handler without doing lock work in the handler.
        self._pending: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._stopping = False
        self._warmed_shapes: set[tuple[int, int, int, int, str]] = set()

    def set_ready(self, startup: dict[str, Any]) -> None:
        with self._lock:
            self.ready = True
            self.phase = "ready"
            self.startup = startup
            warmed_shapes = startup.get(
                "warmed_shapes",
                [
                    [1344, 768, 124, 24, "t2va"],
                    [1344, 768, 124, 24, "i2va"],
                ],
            )
            self._warmed_shapes.update(tuple(shape) for shape in warmed_shapes)

    def submit(
        self, request: GenerationRequest, first_frame_bytes: bytes | None = None
    ) -> GenerationJob:
        with self._lock:
            if not self.ready:
                raise ServiceError("model is still loading")
            if len(self._jobs) >= self.max_jobs:
                removable = next(
                    (
                        key
                        for key, job in self._jobs.items()
                        if job.status in {"succeeded", "failed"}
                    ),
                    None,
                )
                if removable is None:
                    raise ServiceError("job queue is full")
                self._jobs.pop(removable)
            if (request.mode == "i2va") != (first_frame_bytes is not None):
                raise ServiceError("i2va requires exactly one first frame")
            job_id = uuid.uuid4().hex
            first_frame_path = None
            first_frame_sha256 = None
            if first_frame_bytes is not None:
                if self.job_root is None:
                    raise ServiceError("first-frame storage is not configured")
                input_path = self.job_root / "jobs" / job_id / "first-frame.image"
                input_path.parent.mkdir(parents=True, exist_ok=True)
                input_path.write_bytes(first_frame_bytes)
                first_frame_path = str(input_path)
                first_frame_sha256 = hashlib.sha256(first_frame_bytes).hexdigest()
            job = GenerationJob(
                job_id=job_id,
                prompt=request.prompt,
                seed=request.seed,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                fps=request.fps,
                mode=request.mode,
                first_frame_path=first_frame_path,
                first_frame_sha256=first_frame_sha256,
                first_shape_warmup=(
                    request.width,
                    request.height,
                    request.num_frames,
                    request.fps,
                    request.mode,
                )
                not in self._warmed_shapes,
                status="queued",
                created_at=_now(),
            )
            self._jobs[job.job_id] = job
            self._pending.put(job.job_id)
            return job

    def enhance_prompt(
        self, prompt: str, duration_seconds: int, mode: str = "t2va"
    ) -> dict[str, Any]:
        cache_key = (prompt, duration_seconds, mode)
        with self._lock:
            if not self.ready:
                raise ServiceError("model is still loading")
            cached = self._prompt_cache.get(cache_key)
            if cached is not None:
                return {**cached, "cached": True}
            enhancer = self.prompt_enhancer
        if enhancer is None:
            raise ServiceError("prompt enhancer is not configured")
        # vLLM is configured for one sequence; serialize callers and check the
        # cache again after waiting so identical requests are never duplicated.
        with self._prompt_lock:
            with self._lock:
                cached = self._prompt_cache.get(cache_key)
                if cached is not None:
                    return {**cached, "cached": True}
            enhanced = enhancer.enhance(
                prompt, duration_seconds=duration_seconds, mode=mode
            )
            result = {
                "prompt": enhanced.prompt,
                "seconds": enhanced.seconds,
                "input_tokens": enhanced.input_tokens,
                "output_tokens": enhanced.output_tokens,
                "model": enhanced.model,
                "cached": False,
            }
            with self._lock:
                if len(self._prompt_cache) >= self.max_jobs:
                    self._prompt_cache.pop(next(iter(self._prompt_cache)))
                self._prompt_cache[cache_key] = result
            return result

    def open_story(
        self,
        premise: str,
        duration_seconds: int,
        display_language: str = "English",
    ) -> dict[str, Any]:
        cache_key = (premise, duration_seconds, display_language)
        with self._lock:
            if not self.ready:
                raise ServiceError("model is still loading")
            cached = self._story_opening_cache.get(cache_key)
            if cached is not None:
                return {**cached, "cached": True}
            enhancer = self.prompt_enhancer
        if enhancer is None:
            raise ServiceError("story director is not configured")
        with self._prompt_lock:
            with self._lock:
                cached = self._story_opening_cache.get(cache_key)
                if cached is not None:
                    return {**cached, "cached": True}
            opening = enhancer.open_story(
                premise,
                duration_seconds=duration_seconds,
                display_language=display_language,
            )
            result = {
                "title": opening.title,
                "opening_summary": opening.opening_summary,
                "story_bible": opening.story_bible,
                "story_arc": opening.story_arc,
                "initial_state": opening.initial_state,
                "prompt": opening.prompt,
                "seconds": opening.seconds,
                "input_tokens": opening.input_tokens,
                "output_tokens": opening.output_tokens,
                "model": opening.model,
                "cached": False,
            }
            with self._lock:
                if len(self._story_opening_cache) >= self.max_jobs:
                    self._story_opening_cache.pop(next(iter(self._story_opening_cache)))
                self._story_opening_cache[cache_key] = result
            return result

    def plan_story(
        self,
        *,
        premise: str,
        current_prompt: str,
        history: list[str],
        story_bible: dict[str, Any],
        story_arc: list[dict[str, Any]],
        story_state: dict[str, Any],
        duration_seconds: int,
        display_language: str = "English",
        source_job_id: str,
    ) -> dict[str, Any]:
        structured_context = json.dumps(
            [story_bible, story_arc, story_state],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        cache_key = (
            premise,
            current_prompt,
            tuple(history),
            structured_context,
            duration_seconds,
            display_language,
            source_job_id,
        )
        with self._lock:
            if not self.ready:
                raise ServiceError("model is still loading")
            cached = self._story_cache.get(cache_key)
            if cached is not None:
                return {**cached, "cached": True}
            enhancer = self.prompt_enhancer
            source_job = self._jobs.get(source_job_id)
        if enhancer is None:
            raise ServiceError("story director is not configured")
        if source_job is None or source_job.status != "succeeded":
            raise ServiceError("story source job is not available")
        if self.job_root is None:
            raise ServiceError("story context storage is not configured")
        context_frame_paths = tuple(
            self.job_root / "jobs" / source_job_id / f"story-frame-{index:02d}.jpg"
            for index in range(1, STORY_CONTEXT_FRAME_COUNT + 1)
        )
        if not all(path.is_file() for path in context_frame_paths):
            raise ServiceError("story source frames are not available")
        with self._prompt_lock:
            with self._lock:
                cached = self._story_cache.get(cache_key)
                if cached is not None:
                    return {**cached, "cached": True}
            planned = enhancer.plan_story(
                premise=premise,
                current_prompt=current_prompt,
                history=history,
                story_bible=story_bible,
                story_arc=story_arc,
                story_state=story_state,
                duration_seconds=duration_seconds,
                display_language=display_language,
                context_frame_paths=context_frame_paths,
            )
            print(
                "[h3-flash web] story plan "
                f"source={source_job_id} observer={planned.observer_seconds:.3f}s "
                f"director={planned.director_seconds:.3f}s "
                f"prompt={planned.prompt_writer_seconds:.3f}s "
                f"total={planned.seconds:.3f}s",
                flush=True,
            )
            result = {
                "visual_state": planned.visual_state,
                "visual_observation": planned.visual_observation,
                "scene_summary": planned.scene_summary,
                "branches": list(planned.branches),
                "seconds": planned.seconds,
                "input_tokens": planned.input_tokens,
                "output_tokens": planned.output_tokens,
                "model": planned.model,
                "observer_seconds": planned.observer_seconds,
                "director_seconds": planned.director_seconds,
                "prompt_writer_seconds": planned.prompt_writer_seconds,
                "writer_seconds": planned.writer_seconds,
                "cached": False,
            }
            with self._lock:
                if len(self._story_cache) >= self.max_jobs:
                    self._story_cache.pop(next(iter(self._story_cache)))
                self._story_cache[cache_key] = result
            return result

    def next_job(self) -> GenerationJob | None:
        while not self._stopping:
            try:
                job_id = self._pending.get(timeout=0.25)
                break
            except queue.Empty:
                continue
        else:
            return None
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = _now()
            self._changed.notify_all()
            return job

    def request_stop(self) -> None:
        # Keep the signal handler lock-free. next_job() polls this flag so the
        # interpreter never stays parked indefinitely inside a blocking get().
        self._stopping = True

    def succeed(
        self, job_id: str, *, generation_seconds: float, request_seconds: float
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "succeeded"
            job.finished_at = _now()
            job.generation_seconds = generation_seconds
            job.request_seconds = request_seconds
            job.video_url = f"/outputs/{job_id}.mp4"
            job.last_frame_url = f"/outputs/{job_id}.last-frame.jpg"
            self._warmed_shapes.add(
                (job.width, job.height, job.num_frames, job.fps, job.mode)
            )
            self._changed.notify_all()

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.finished_at = _now()
            job.error = error
            self._changed.notify_all()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            position = None
            if job.status == "queued":
                queued = [
                    item.job_id
                    for item in self._jobs.values()
                    if item.status == "queued"
                ]
                position = queued.index(job_id) + 1
            return job.public(position)

    def wait_for_change(
        self, job_id: str, *, since: str, timeout: float = 15.0
    ) -> dict[str, Any] | None:
        """Wait until a job changes state, avoiding browser polling latency."""

        deadline = monotonic() + timeout
        with self._changed:
            while True:
                job = self._jobs.get(job_id)
                if job is None:
                    return None
                if job.status != since:
                    return job.public()
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return job.public()
                self._changed.wait(remaining)

    def status(self) -> dict[str, Any]:
        with self._lock:
            counts = {
                status: sum(job.status == status for job in self._jobs.values())
                for status in ("queued", "running", "succeeded", "failed")
            }
            return {
                "ready": self.ready,
                "phase": self.phase,
                "profile": self.profile,
                "jobs": counts,
                "startup": self.startup,
                "capabilities": {
                    "resolutions": list(WEB_RESOLUTION_PRESETS),
                    "durations": list(WEB_DURATION_PRESETS),
                    "reference_frames": {"first": True, "last": False},
                    "prompt_enhancer": (
                        self.prompt_enhancer.describe()
                        if self.prompt_enhancer is not None
                        else {"enabled": False}
                    ),
                    "story": {
                        "enabled": self.prompt_enhancer is not None,
                        "branch_count": 2,
                        "continuation": "tail-frame-i2va",
                        "visual_context": "four-frames-final-two-seconds",
                    },
                },
            }


def build_handler(
    state: ServiceState, *, web_root: Path, story_root: Path, output_root: Path
) -> type[BaseHTTPRequestHandler]:
    index_path = web_root / "index.html"
    story_index_path = story_root / "index.html"

    class Handler(BaseHTTPRequestHandler):
        server_version = "H3FlashDemo/1"
        protocol_version = "HTTP/1.1"

        def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, status: HTTPStatus, message: str) -> None:
            self._json(status, {"error": message})

        def _video_path(self, path: str) -> Path | None:
            if not path.startswith("/outputs/") or not path.endswith(".mp4"):
                return None
            job_id = path.removeprefix("/outputs/").removesuffix(".mp4")
            job = state.get(job_id)
            if job is None or job["status"] != "succeeded":
                return None
            video = output_root / "jobs" / job_id / "output.mp4"
            return video if video.is_file() else None

        def _last_frame_path(self, path: str) -> Path | None:
            suffix = ".last-frame.jpg"
            if not path.startswith("/outputs/") or not path.endswith(suffix):
                return None
            job_id = path.removeprefix("/outputs/").removesuffix(suffix)
            job = state.get(job_id)
            if job is None or job["status"] != "succeeded":
                return None
            image = output_root / "jobs" / job_id / "last-frame.jpg"
            return image if image.is_file() else None

        def _serve_last_frame(self, image: Path, *, send_body: bool) -> None:
            size = image.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "private, max-age=3600, immutable")
            self.end_headers()
            if send_body:
                try:
                    self.wfile.write(image.read_bytes())
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _serve_video(self, video: Path, *, send_body: bool) -> None:
            size = video.stat().st_size
            try:
                start, end, partial = parse_byte_range(self.headers.get("Range"), size)
            except ServiceError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Disposition", "inline")
            self.send_header("Cache-Control", "private, max-age=3600, immutable")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not send_body:
                return
            try:
                with video.open("rb") as handle:
                    handle.seek(start)
                    remaining = length
                    while remaining:
                        chunk = handle.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_HEAD(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            last_frame = self._last_frame_path(path)
            if last_frame is not None:
                self._serve_last_frame(last_frame, send_body=False)
                return
            video = self._video_path(path)
            if video is None:
                self._error(HTTPStatus.NOT_FOUND, "video is not available")
                return
            self._serve_video(video, send_body=False)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path in {"/", "/index.html"}:
                payload = index_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            if path in {"/story", "/story/", "/story/index.html"}:
                payload = story_index_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/api/status":
                self._json(HTTPStatus.OK, state.status())
                return
            if path.startswith("/api/jobs/"):
                job_id = path.removeprefix("/api/jobs/")
                query = parse_qs(parsed.query)
                since = query.get("since", [None])[0]
                wait = query.get("wait", ["0"])[0] == "1"
                job = (
                    state.wait_for_change(job_id, since=since)
                    if wait and since in {"queued", "running"}
                    else state.get(job_id)
                )
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "unknown job")
                else:
                    self._json(HTTPStatus.OK, job)
                return
            if path.startswith("/outputs/") and path.endswith(".mp4"):
                video = self._video_path(path)
                if video is None:
                    self._error(HTTPStatus.NOT_FOUND, "video is not available")
                    return
                self._serve_video(video, send_body=True)
                return
            if path.startswith("/outputs/") and path.endswith(".last-frame.jpg"):
                last_frame = self._last_frame_path(path)
                if last_frame is None:
                    self._error(HTTPStatus.NOT_FOUND, "last frame is not available")
                    return
                self._serve_last_frame(last_frame, send_body=True)
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path not in {
                "/api/jobs",
                "/api/prompts/enhance",
                "/api/story/opening",
                "/api/story/branches",
            }:
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
                return
            if size <= 0 or size > MAX_REQUEST_BYTES:
                self._error(HTTPStatus.BAD_REQUEST, "invalid request size")
                return
            try:
                if path == "/api/jobs":
                    request, first_frame_bytes = parse_job_request(
                        self.headers.get("Content-Type", ""), self.rfile.read(size)
                    )
                    job = state.submit(request, first_frame_bytes)
                elif path == "/api/prompts/enhance":
                    payload = json.loads(self.rfile.read(size))
                    prompt, duration, mode = validate_prompt_enhancement_payload(
                        payload
                    )
                    enhancement = state.enhance_prompt(prompt, duration, mode)
                elif path == "/api/story/opening":
                    payload = json.loads(self.rfile.read(size))
                    premise, duration, display_language = (
                        validate_story_opening_payload(payload)
                    )
                    story_opening = state.open_story(
                        premise,
                        duration,
                        display_language,
                    )
                else:
                    payload = json.loads(self.rfile.read(size))
                    (
                        premise,
                        current_prompt,
                        history,
                        story_bible,
                        story_arc,
                        story_state,
                        duration,
                        display_language,
                        source_job_id,
                    ) = validate_story_payload(payload)
                    story_plan = state.plan_story(
                        premise=premise,
                        current_prompt=current_prompt,
                        history=history,
                        story_bible=story_bible,
                        story_arc=story_arc,
                        story_state=story_state,
                        duration_seconds=duration,
                        display_language=display_language,
                        source_job_id=source_job_id,
                    )
            except (json.JSONDecodeError, UnicodeDecodeError, ServiceError) as error:
                status = (
                    HTTPStatus.SERVICE_UNAVAILABLE
                    if str(error) == "model is still loading"
                    else HTTPStatus.BAD_REQUEST
                )
                self._error(status, str(error))
                return
            except PromptEnhancementError as error:
                print(
                    f"[h3-flash web] prompt pipeline error on {path}: {error}",
                    flush=True,
                )
                self._error(HTTPStatus.BAD_GATEWAY, str(error))
                return
            if path == "/api/jobs":
                self._json(HTTPStatus.ACCEPTED, job.public(queue_position=1))
                return
            if path == "/api/prompts/enhance":
                response = enhancement
            elif path == "/api/story/opening":
                response = story_opening
            else:
                response = story_plan
            self._json(HTTPStatus.OK, response)

        def log_message(self, message: str, *args: Any) -> None:
            print(f"[h3-flash web] {self.address_string()} {message % args}")

    return Handler


def _start_http(
    state: ServiceState,
    *,
    host: str,
    port: int,
    web_root: Path,
    story_root: Path,
    output_root: Path,
) -> ThreadingHTTPServer:
    handler = build_handler(
        state, web_root=web_root, story_root=story_root, output_root=output_root
    )
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[h3-flash] web demo: http://{host}:{port}", flush=True)
    return server


def run_service(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    profiles = ProfileRepository(args.profiles_dir)
    locks = LockRepository(args.locks_dir)
    profile = profiles.resolve(args.profile)
    expected_world = int(profile["parallel"]["world_size"])
    if world_size != expected_world:
        raise RuntimeError(
            f"profile requires {expected_world} ranks, launcher provided {world_size}"
        )

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(device),
    )
    model_root = args.model_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    require_profile_model_verification(locks, profile, model_root)
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        prompt_enhancer = (
            OpenAICompatiblePromptEnhancer(
                endpoint=args.prompt_enhancer_url,
                model=args.prompt_enhancer_model,
                timeout_seconds=args.prompt_enhancer_timeout,
                story_endpoint=args.story_director_url,
                story_model=args.story_director_model,
            )
            if args.prompt_enhancer_url
            else None
        )
        state = ServiceState(
            profile=args.profile,
            prompt_enhancer=prompt_enhancer,
            job_root=output_root,
        )
        server = _start_http(
            state,
            host=args.host,
            port=args.port,
            web_root=args.web_root.expanduser().resolve(),
            story_root=args.story_root.expanduser().resolve(),
            output_root=output_root,
        )
    else:
        state = None
        server = None

    def request_stop(_signum: int, _frame: Any) -> None:
        # torchrun forwards signals to every rank. Rank 0 wakes the queue and
        # broadcasts the shutdown command; the other ranks remain available
        # to receive it instead of interrupting their current collective.
        if state is not None:
            state.request_stop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)

    optimizations = profile.get("optimizations", {})
    backend = OfficialDiffusersBackend(
        model_root,
        generation_device=device,
        text_device=device,
        attention_backend=profile["attention"]["backend"],
        fused_qkv=optimizations.get("fused_qkv") is True,
        invariant_caches=optimizations.get("invariant_caches") is True,
        transformer_fusions=optimizations.get("transformer_fusions") is True,
        packed_ulysses=optimizations.get("packed_ulysses") is True,
        rank_local_inputs=optimizations.get("rank_local_inputs") is True,
        compact_output_gather=optimizations.get("compact_output_gather") is True,
        vae_compile_mode=profile.get("vae", {}).get("compile_mode"),
        ulysses_degree=world_size,
        vae_clip_parallel=profile.get("vae", {}).get("video_decoder")
        == "official_clip_parallel",
    )
    backend.load()
    dist.barrier()
    warmup_shape = (
        {"width": 832, "height": 480, "num_frames": 243}
        if args.warm_story
        else {"width": 1344, "height": 768, "num_frames": 124}
    )
    warmup_request = GenerationRequest(
        prompt="A calm lake at sunrise.", seed=0, **warmup_shape
    )
    warmup_started = perf_counter()
    schedule = backend.prepare_fixed_schedule(
        warmup_request,
        api_num_inference_steps=profile["sampling"]["api_num_inference_steps"],
        freeze=False,
    )
    from PIL import Image

    i2va_warmup_request = GenerationRequest(
        prompt=(
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            "integrated_multimodal_description: [Shot 1] The calm lake in "
            "<Picture 1> develops gentle ripples.\n\n"
            "overall_soundscape: Light water movement.\n\n"
            "non_diegetic_music: N/A"
        ),
        seed=0,
        mode="i2va",
        **warmup_shape,
    )
    i2va_schedule = backend.prepare_fixed_schedule(
        i2va_warmup_request,
        api_num_inference_steps=profile["sampling"]["api_num_inference_steps"],
        first_frame=Image.new("RGB", (warmup_shape["width"], warmup_shape["height"])),
        expected_values_per_site=schedule["model_evaluations"] * 2,
    )
    dist.barrier()
    schedule_seconds = perf_counter() - warmup_started
    vae_started = perf_counter()
    warmup = backend.generate(
        warmup_request,
        api_num_inference_steps=profile["sampling"]["api_num_inference_steps"],
        output_type="pt",
    )
    dist.barrier()
    vae_seconds = perf_counter() - vae_started
    del warmup
    runtime_source = backend.source_provenance() if rank == 0 else None
    if rank == 0:
        assert state is not None
        state.set_ready(
            {
                "model_load_seconds": backend.load_seconds,
                "schedule_seconds": schedule_seconds,
                "schedule_entries": i2va_schedule["entries"],
                "schedule_variants": ["t2va", "i2va"],
                "vae_warmup_seconds": vae_seconds,
                "warmed_shapes": [
                    [
                        warmup_request.width,
                        warmup_request.height,
                        warmup_request.num_frames,
                        warmup_request.fps,
                        mode,
                    ]
                    for mode in ("t2va", "i2va")
                ],
            }
        )
        print("[h3-flash] web demo ready", flush=True)

    pool = HostTransferPool() if rank == 0 else None
    try:
        while True:
            if rank == 0:
                assert state is not None
                worker_job = state.next_job()
                command: dict[str, Any] | None
                if worker_job is None:
                    command = {"shutdown": True}
                else:
                    command = {
                        "job_id": worker_job.job_id,
                        "prompt": worker_job.prompt,
                        "seed": worker_job.seed,
                        "width": worker_job.width,
                        "height": worker_job.height,
                        "num_frames": worker_job.num_frames,
                        "fps": worker_job.fps,
                        "mode": worker_job.mode,
                        "first_frame_path": worker_job.first_frame_path,
                        "first_frame_sha256": worker_job.first_frame_sha256,
                    }
            else:
                command = None
            payload = [command]
            dist.broadcast_object_list(payload, src=0)
            command = payload[0]
            assert command is not None
            if command.get("shutdown"):
                break
            request = GenerationRequest(
                prompt=command["prompt"],
                seed=command["seed"],
                width=command["width"],
                height=command["height"],
                num_frames=command["num_frames"],
                fps=command["fps"],
                mode=command["mode"],
            )
            first_frame = None
            if command["first_frame_path"] is not None:
                from PIL import Image

                with Image.open(command["first_frame_path"]) as image:
                    first_frame = image.copy()
            job_id = command["job_id"]
            case_dir = output_root / "jobs" / job_id
            if rank == 0:
                assert runtime_source is not None
                manifest = build_manifest(
                    profiles,
                    args.profile,
                    request,
                    backend="official-diffusers-resident-web",
                    locks=locks,
                    runtime_source=runtime_source,
                    execution={
                        "artifact_format": "mp4",
                        "world_size": world_size,
                        "first_frame": (
                            {
                                "sha256": command["first_frame_sha256"],
                                "path": "first-frame.image",
                            }
                            if command["first_frame_path"] is not None
                            else None
                        ),
                    },
                )
                _write_json_atomic(case_dir / "manifest.json", manifest)
            else:
                manifest = None

            dist.barrier()
            request_started = perf_counter()
            generation = backend.generate(
                request,
                api_num_inference_steps=profile["sampling"]["api_num_inference_steps"],
                output_type="pt",
                first_frame=first_frame,
            )
            generation_seconds = _maximum_across_ranks(
                [generation.generation_seconds], device
            )[0]
            peaks = _peaks_across_ranks(max(generation.peak_gpu_memory_bytes), device)
            if rank == 0:
                assert state is not None and manifest is not None and pool is not None
                try:
                    transfer_started = perf_counter()
                    original_video = generation.state.get("videos")
                    host = pool.copy(
                        {
                            "video": _prepare_video_on_device(original_video),
                            "audio": generation.state.get("audio"),
                        }
                    )
                    sampling_rate = generation.state.get("sampling_rate")
                    # Release the large device outputs before CPU encoding. This hides
                    # allocator cleanup under the encode and makes the worker ready for
                    # a back-to-back request as soon as the current video is published.
                    generation.state.values.clear()
                    transfer_seconds = perf_counter() - transfer_started
                    artifact_started = perf_counter()
                    artifact_path = case_dir / "output.mp4"
                    last_frame_path = case_dir / "last-frame.jpg"
                    Image.fromarray(host["video"][-1].contiguous().numpy()).save(
                        last_frame_path, format="JPEG", quality=92
                    )
                    story_frames = []
                    for ordinal, frame_index in enumerate(
                        story_context_frame_indices(len(host["video"]), request.fps),
                        start=1,
                    ):
                        story_frame_path = case_dir / f"story-frame-{ordinal:02d}.jpg"
                        story_frame = Image.fromarray(
                            host["video"][frame_index].contiguous().numpy()
                        )
                        story_frame.thumbnail((448, 448), Image.Resampling.LANCZOS)
                        story_frame.save(story_frame_path, format="JPEG", quality=85)
                        story_frames.append(
                            {
                                "path": story_frame_path.name,
                                "source_frame_index": frame_index,
                                "bytes": story_frame_path.stat().st_size,
                                "sha256": sha256_file(story_frame_path),
                            }
                        )
                    output_config = profile["output"]
                    encoder = encode_video_ffmpeg(
                        host["video"],
                        fps=request.fps,
                        output_path=artifact_path,
                        audio=host["audio"][0],
                        audio_sample_rate=sampling_rate,
                        preset=str(output_config["preset"]),
                        crf=int(output_config["crf"]),
                        video_codec=str(output_config["video_codec"]),
                        pixel_format=str(output_config["pixel_format"]),
                        audio_codec=str(output_config["audio_codec"]),
                    )
                    artifact_seconds = perf_counter() - artifact_started
                    request_seconds = perf_counter() - request_started
                    result = {
                        "schema_version": 1,
                        "created_at": _now(),
                        "profile": args.profile,
                        "request": asdict(request),
                        "timing_seconds": {
                            "generation": generation_seconds,
                            "device_to_host": transfer_seconds,
                            "artifact_write": artifact_seconds,
                            "request_process": request_seconds,
                        },
                        "peak_gpu_memory_bytes": peaks,
                        "outputs": {
                            "path": artifact_path.name,
                            "bytes": artifact_path.stat().st_size,
                            "sha256": sha256_file(artifact_path),
                            "last_frame": {
                                "path": last_frame_path.name,
                                "bytes": last_frame_path.stat().st_size,
                                "sha256": sha256_file(last_frame_path),
                            },
                            "story_context_frames": story_frames,
                            "encoder": encoder,
                        },
                        "manifest_sha256": manifest["manifest_sha256"],
                    }
                    _write_json_atomic(case_dir / "result.json", result)
                    state.succeed(
                        job_id,
                        generation_seconds=generation_seconds,
                        request_seconds=request_seconds,
                    )
                    print(
                        f"[h3-flash] web job {job_id}: "
                        f"generation={generation_seconds:.3f}s "
                        f"e2e={request_seconds:.3f}s",
                        flush=True,
                    )
                except Exception as error:  # keep the resident worker available
                    state.fail(job_id, f"{type(error).__name__}: {error}")
                    print(f"[h3-flash] web job {job_id} failed: {error}", flush=True)
            del generation
            dist.barrier()
    finally:
        if server is not None:
            server.shutdown()
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(prog="h3-flash-web")
    parser.add_argument("--profile", choices=("lossless", "flash"), default="flash")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--web-root", type=Path, default=project_root / "apps" / "web")
    parser.add_argument(
        "--story-root", type=Path, default=project_root / "apps" / "story"
    )
    parser.add_argument(
        "--warm-story",
        action="store_true",
        help="warm the fixed 480p/10s story shape before accepting requests",
    )
    parser.add_argument(
        "--prompt-enhancer-url",
        default=os.environ.get("H3_FLASH_PROMPT_ENHANCER_URL"),
    )
    parser.add_argument(
        "--prompt-enhancer-model",
        default=os.environ.get("H3_FLASH_PROMPT_ENHANCER_MODEL", "qwen3-8b-prompt"),
    )
    parser.add_argument(
        "--story-director-url",
        default=os.environ.get("H3_FLASH_STORY_DIRECTOR_URL"),
    )
    parser.add_argument(
        "--story-director-model",
        default=os.environ.get("H3_FLASH_STORY_DIRECTOR_MODEL"),
    )
    parser.add_argument("--prompt-enhancer-timeout", type=float, default=30)
    parser.add_argument("--profiles-dir", type=Path)
    parser.add_argument("--locks-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    run_service(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
