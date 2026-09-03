import base64
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from h3_flash.manifest import GenerationRequest
from h3_flash.service import (
    ServiceError,
    ServiceState,
    parse_byte_range,
    parse_job_request,
    validate_first_frame,
    validate_job_payload,
    validate_prompt_enhancement_payload,
    validate_story_opening_payload,
    validate_story_payload,
    story_context_frame_indices,
)
from h3_flash.prompting import (
    PromptEnhancementResult,
    StoryOpeningResult,
    StoryPlanResult,
)


def story_context() -> tuple[
    dict[str, object], list[dict[str, object]], dict[str, object]
]:
    bible: dict[str, object] = {
        "protagonist_identity": "A courier",
        "visual_anchor": "Short black hair and a dark jacket",
        "personality": "Resourceful and protective",
        "decision_style": "Uses the environment",
        "core_goal": "Deliver the key",
        "weakness": "Will not abandon others",
        "starting_inventory": ["glass key"],
        "starting_relationships": [],
        "non_negotiables": ["Keep the key"],
        "world_setting": "A metro station",
        "genre_tone": "Tense pursuit",
        "visual_style": "Cinematic realism",
        "world_rules": ["The key is an ordinary physical key"],
        "central_mystery": "Who sealed the exit",
        "opposing_force": "A drone",
        "stakes": "The exit will close",
        "emotional_thread": "Returning the key turns suspicion into trust",
        "ending_direction": "Deliver the key and expose the pursuer",
    }
    arc = [
        {
            "chapter": index,
            "purpose": f"beat {index}",
            "story_change": f"change {index}",
            "milestone": f"step {index}",
        }
        for index in range(1, 6)
    ]
    state: dict[str, object] = {
        "chapter": 1,
        "location": "Station fork",
        "current_goal": "Reach the exit",
        "inventory": ["glass key"],
        "relationships": [],
        "established_facts": ["A drone follows the courier"],
        "open_threads": [
            {"id": "thread-1", "question": "Who closed the gate", "urgency": "now"}
        ],
        "last_decision": "Enter the station",
        "last_consequence": "The gate begins to close",
        "character_condition": "Running while holding the key",
        "emotional_progress": "The courier is still working alone",
    }
    return bible, arc, state


class FakePromptEnhancer:
    calls = 0

    def describe(self) -> dict[str, object]:
        return {"enabled": True, "model": "fake"}

    def enhance(
        self, prompt: str, *, duration_seconds: int, mode: str = "t2va"
    ) -> PromptEnhancementResult:
        del mode
        self.calls += 1
        return PromptEnhancementResult(
            prompt=(
                f"integrated_multimodal_description: [Shot 1] {prompt}\n\n"
                "overall_soundscape: Pawsteps.\n\n"
                "non_diegetic_music: N/A"
            ),
            seconds=0.4,
            input_tokens=100,
            output_tokens=30,
            model="fake",
        )

    def open_story(
        self,
        premise: str,
        *,
        duration_seconds: int,
        display_language: str = "English",
    ) -> StoryOpeningResult:
        del duration_seconds, display_language
        self.calls += 1
        bible, arc, state = story_context()
        return StoryOpeningResult(
            title="The Glass Key",
            opening_summary="The courier reaches a closing gate.",
            story_bible=bible,
            story_arc=arc,
            initial_state=state,
            prompt=(
                f"integrated_multimodal_description: [Shot 1] {premise}\n\n"
                "overall_soundscape: Footsteps.\n\n"
                "non_diegetic_music: N/A"
            ),
            seconds=0.5,
            input_tokens=100,
            output_tokens=200,
            model="fake",
        )

    def plan_story(
        self,
        *,
        premise: str,
        current_prompt: str,
        history: list[str],
        story_bible: dict[str, object],
        story_arc: list[dict[str, object]],
        story_state: dict[str, object],
        duration_seconds: int,
        display_language: str = "English",
        context_frame_paths: tuple[Path, ...] = (),
    ) -> StoryPlanResult:
        del premise, current_prompt, history, duration_seconds, display_language
        assert story_bible["core_goal"] == "Deliver the key"
        assert len(story_arc) == 5
        assert story_state["chapter"] == 1
        assert len(context_frame_paths) == 4
        self.calls += 1
        branches = (
            {
                "id": "a",
                "label": "Take the bridge",
                "hook": "The bridge begins to move.",
                "story_memory": "The courier takes the bridge before it moves.",
                "next_state": {**story_state, "chapter": 2},
                "prompt": "first prompt",
            },
            {
                "id": "b",
                "label": "Open the door",
                "hook": "A hidden room waits.",
                "story_memory": "The courier opens the door to a hidden room.",
                "next_state": {**story_state, "chapter": 2},
                "prompt": "second prompt",
            },
        )
        return StoryPlanResult(
            visual_state="The courier slows at a fork.",
            visual_observation={
                "visual_state": "信使在岔路口减速。",
                "visible_motion": "The courier slows.",
                "last_frame_state": "The courier reaches a fork.",
                "continuity_constraints": ["One courier", "Two passages"],
            },
            scene_summary="The courier reaches a choice.",
            branches=branches,
            seconds=0.6,
            input_tokens=200,
            output_tokens=120,
            model="fake",
            observer_seconds=0.2,
            director_seconds=0.3,
            prompt_writer_seconds=0.1,
            writer_seconds=0.4,
        )


def test_validate_job_payload_normalizes_prompt_and_seed() -> None:
    assert validate_job_payload(
        {"prompt": "  A fox runs.  ", "seed": "7"}
    ) == GenerationRequest(
        prompt="A fox runs.",
        seed=7,
        width=1344,
        height=768,
        num_frames=124,
        fps=24,
    )


def test_validate_job_payload_accepts_web_presets() -> None:
    request = validate_job_payload(
        {
            "prompt": "A fox runs.",
            "seed": 7,
            "resolution": "480p-portrait",
            "duration": 15,
        }
    )
    assert (request.width, request.height) == (480, 832)
    assert request.num_frames == 362


def test_validate_prompt_enhancement_payload() -> None:
    assert validate_prompt_enhancement_payload(
        {"prompt": "  A fox runs. ", "duration": "10"}
    ) == ("A fox runs.", 10, "t2va")


def test_validate_story_opening_payload() -> None:
    assert validate_story_opening_payload(
        {"premise": "  A courier carries a key. ", "duration": 10, "language": "zh-CN"}
    ) == ("A courier carries a key.", 10, "Simplified Chinese")


def test_validate_story_payload() -> None:
    bible, arc, state = story_context()
    assert validate_story_payload(
        {
            "premise": "  A courier carries a mysterious key. ",
            "current_prompt": " The courier reaches a station. ",
            "history": [" Board the train "],
            "story_bible": bible,
            "story_arc": arc,
            "story_state": state,
            "duration": "10",
            "language": "zh-CN",
            "source_job_id": "a" * 32,
        }
    ) == (
        "A courier carries a mysterious key.",
        "The courier reaches a station.",
        ["Board the train"],
        bible,
        arc,
        state,
        10,
        "Simplified Chinese",
        "a" * 32,
    )


def test_story_context_frame_indices_cover_final_two_seconds() -> None:
    assert story_context_frame_indices(243, 24) == (194, 210, 226, 242)
    assert story_context_frame_indices(124, 24) == (75, 91, 107, 123)


def test_validate_first_frame_and_build_i2va_job(tmp_path: Path) -> None:
    buffer = BytesIO()
    Image.new("RGB", (64, 96), "blue").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    payload = {
        "prompt": "A bird takes flight.",
        "first_frame": f"data:image/png;base64,{encoded}",
    }
    request = validate_job_payload(payload)
    image_bytes = validate_first_frame(payload)
    assert request.mode == "i2va"
    assert request.prompt.startswith("For the target video, at 0.00 seconds")

    state = ServiceState(profile="flash", job_root=tmp_path)
    state.set_ready({"model_load_seconds": 1.0})
    job = state.submit(request, image_bytes)
    public = job.public()
    assert public["has_first_frame"] is True
    assert public["first_frame_sha256"]
    assert "first_frame_path" not in public
    assert (tmp_path / "jobs" / job.job_id / "first-frame.image").is_file()


def test_parse_multipart_job_keeps_first_frame_binary() -> None:
    buffer = BytesIO()
    Image.new("RGB", (64, 96), "blue").save(buffer, format="PNG")
    request = json.dumps({"prompt": "A bird takes flight."}).encode()
    boundary = "h3-flash-test"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="request"\r\n'
        "Content-Type: application/json\r\n\r\n"
    ).encode() + request
    body += (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="first_frame"; filename="frame.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + buffer.getvalue()
    body += f"\r\n--{boundary}--\r\n".encode()

    parsed, image_bytes = parse_job_request(
        f"multipart/form-data; boundary={boundary}", body
    )
    assert parsed.mode == "i2va"
    assert parsed.prompt.startswith("For the target video, at 0.00 seconds")
    assert image_bytes == buffer.getvalue()


@pytest.mark.parametrize(
    ("header", "size", "expected"),
    [
        (None, 100, (0, 99, False)),
        ("bytes=0-9", 100, (0, 9, True)),
        ("bytes=90-", 100, (90, 99, True)),
        ("bytes=-10", 100, (90, 99, True)),
        ("bytes=0-999", 100, (0, 99, True)),
    ],
)
def test_parse_byte_range(
    header: str | None, size: int, expected: tuple[int, int, bool]
) -> None:
    assert parse_byte_range(header, size) == expected


@pytest.mark.parametrize("header", ["items=0-1", "bytes=", "bytes=100-", "bytes=4-2"])
def test_parse_byte_range_rejects_invalid_ranges(header: str) -> None:
    with pytest.raises(ServiceError, match="Range"):
        parse_byte_range(header, 100)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "prompt"),
        ({"prompt": "   "}, "prompt"),
        ({"prompt": "ok", "seed": True}, "seed"),
        ({"prompt": "ok", "seed": -1}, "seed"),
        ({"prompt": "ok", "resolution": "4k"}, "resolution"),
        ({"prompt": "ok", "duration": 7}, "duration"),
    ],
)
def test_validate_job_payload_rejects_bad_inputs(payload: dict, message: str) -> None:
    with pytest.raises(ServiceError, match=message):
        validate_job_payload(payload)


def test_service_state_tracks_one_job(tmp_path: Path) -> None:
    del tmp_path
    state = ServiceState(profile="flash")
    with pytest.raises(ServiceError, match="still loading"):
        state.submit(GenerationRequest(prompt="A fox runs.", seed=7))
    state.set_ready({"model_load_seconds": 1.0})
    submitted = state.submit(GenerationRequest(prompt="A fox runs.", seed=7))
    assert state.get(submitted.job_id)["queue_position"] == 1
    running = state.next_job()
    assert running.job_id == submitted.job_id
    assert state.get(submitted.job_id)["status"] == "running"
    state.succeed(submitted.job_id, generation_seconds=2.0, request_seconds=2.5)
    result = state.get(submitted.job_id)
    assert result["status"] == "succeeded"
    assert result["video_url"] == f"/outputs/{submitted.job_id}.mp4"
    assert result["last_frame_url"] == f"/outputs/{submitted.job_id}.last-frame.jpg"
    assert result["first_shape_warmup"] is False


def test_service_state_marks_new_shapes_for_first_use_warmup() -> None:
    state = ServiceState(profile="flash")
    state.set_ready({"model_load_seconds": 1.0})
    request = GenerationRequest(
        prompt="A fox runs.", seed=7, width=832, height=480, num_frames=243
    )
    submitted = state.submit(request)
    assert submitted.first_shape_warmup is True
    state.next_job()
    state.succeed(submitted.job_id, generation_seconds=1.0, request_seconds=2.0)
    repeated = state.submit(request)
    assert repeated.first_shape_warmup is False


def test_service_state_can_wake_worker_for_shutdown() -> None:
    state = ServiceState(profile="flash")
    state.request_stop()
    assert state.next_job() is None


def test_service_state_runs_and_caches_prompt_enhancement() -> None:
    enhancer = FakePromptEnhancer()
    state = ServiceState(profile="flash", prompt_enhancer=enhancer)
    state.set_ready({"model_load_seconds": 1.0})
    generated = state.enhance_prompt("A fox runs.", 5)
    assert generated["cached"] is False
    assert generated["model"] == "fake"
    cached = state.enhance_prompt("A fox runs.", 5)
    assert cached["cached"] is True
    assert enhancer.calls == 1


def test_service_state_runs_and_caches_story_opening() -> None:
    enhancer = FakePromptEnhancer()
    state = ServiceState(profile="flash", prompt_enhancer=enhancer)
    state.set_ready({"model_load_seconds": 1.0})
    opening = state.open_story("A courier carries a key.", 10)
    assert opening["cached"] is False
    assert opening["initial_state"]["chapter"] == 1
    cached = state.open_story("A courier carries a key.", 10)
    assert cached["cached"] is True
    assert enhancer.calls == 1


def test_service_state_runs_and_caches_story_planning(tmp_path: Path) -> None:
    enhancer = FakePromptEnhancer()
    state = ServiceState(profile="flash", prompt_enhancer=enhancer, job_root=tmp_path)
    state.set_ready({"model_load_seconds": 1.0})
    source = state.submit(GenerationRequest(prompt="A courier runs.", seed=7))
    state.next_job()
    source_dir = tmp_path / "jobs" / source.job_id
    source_dir.mkdir(parents=True)
    for index in range(1, 5):
        (source_dir / f"story-frame-{index:02d}.jpg").write_bytes(b"frame")
    state.succeed(source.job_id, generation_seconds=1.0, request_seconds=2.0)
    bible, arc, story_state = story_context()
    request = {
        "premise": "A courier carries a key.",
        "current_prompt": "The courier reaches a station.",
        "history": ["Run toward the station"],
        "story_bible": bible,
        "story_arc": arc,
        "story_state": story_state,
        "duration_seconds": 10,
        "source_job_id": source.job_id,
    }
    planned = state.plan_story(**request)
    assert planned["cached"] is False
    assert len(planned["branches"]) == 2
    cached = state.plan_story(**request)
    assert cached["cached"] is True
    assert enhancer.calls == 1


def test_web_demo_keeps_output_above_prompt_and_exposes_prompt_composer() -> None:
    web = (Path(__file__).parents[1] / "apps" / "web" / "index.html").read_text()
    assert web.index('id="job"') < web.index('id="form"')
    assert 'id="quickEnhance"' in web
    assert 'id="aiEnhance"' in web
    assert "/api/prompts/enhance" in web
    assert 'id="videoStage"' in web
    assert 'preload="auto"' in web
    assert 'id="firstFrame"' in web
    assert "new FormData()" in web
    assert "?wait=1&since=" in web
    assert "click → playable" in web
    assert ".video-stage.portrait" in web
    assert "Visual look<select" not in web
    assert "Camera<select" not in web
    assert "Music<select" not in web
    assert "Sound direction<input" not in web
    assert "integrated_multimodal_description:" in web
    assert "overall_soundscape:" in web
    assert "non_diegetic_music:" in web
    assert "MiniMax prompt guide" in web


def test_story_demo_pregenerates_two_tail_frame_continuations() -> None:
    story = (Path(__file__).parents[1] / "apps" / "story" / "index.html").read_text()
    assert "/api/story/opening" in story
    assert "/api/story/branches" in story
    assert "job.last_frame_url" in story
    assert "new FormData()" in story
    assert "Promise.all(node.branches.map" in story
    assert "bufferVideo(job.video_url)" in story
    assert 'id="language"' in story
    assert "language:node.language" in story
    assert "source_job_id:node.job.job_id" in story
    assert "story_bible:app.opening.story_bible" in story
    assert "story_state:node.storyState" in story
    assert "storyState:branch.choice.next_state" in story
    assert "每幕 10 秒" in story
    assert "基于真实尾帧提前生成" in story
