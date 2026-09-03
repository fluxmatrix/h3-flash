#!/usr/bin/env python3
"""Warm the visual observer and text director used by the story demo."""

from __future__ import annotations

import argparse
import json
import urllib.request
from base64 import b64encode
from io import BytesIO
from time import perf_counter

from PIL import Image

from h3_flash.prompting import build_story_observer_messages


def post(endpoint: str, payload: dict[str, object]) -> float:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()
    return perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--director-endpoint")
    parser.add_argument("--director-model")
    args = parser.parse_args()

    buffer = BytesIO()
    Image.new("RGB", (448, 258), (32, 36, 44)).save(buffer, format="JPEG", quality=85)
    frame = f"data:image/jpeg;base64,{b64encode(buffer.getvalue()).decode('ascii')}"
    payload = {
        "model": args.model,
        "messages": build_story_observer_messages(
            (frame, frame, frame, frame), display_language="Simplified Chinese"
        ),
        "max_tokens": 96,
        "temperature": 0,
        "seed": 20260903,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    print(f"[h3-flash] story VLM warmup: {post(args.endpoint, payload):.3f}s")
    if args.director_endpoint and args.director_model:
        director_payload = {
            "model": args.director_model,
            "messages": [
                {"role": "system", "content": "Reply with one word."},
                {"role": "user", "content": "ready"},
            ],
            "max_tokens": 2,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        elapsed = post(args.director_endpoint, director_payload)
        print(f"[h3-flash] story text warmup: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
