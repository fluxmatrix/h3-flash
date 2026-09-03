"""Prompt-engineering helpers for the resident H3 demo."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


PROMPT_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)

_EXPLICIT_EFFECT_MARKERS = (
    "visual effect",
    "special effect",
    "particles",
    "energy wave",
    "magic",
    "magical",
    "levitat",
    "transform",
    "light burst",
    "特效",
    "粒子",
    "能量波",
    "魔法",
    "悬浮",
    "变形",
    "光爆",
)

_UNSUPPORTED_EFFECT_MARKERS = (
    "particle",
    "energy wave",
    "light burst",
    "burst of light",
    "magical aura",
    "glowing trail",
    "levitat",
    "transform",
    "pulsing light",
    "pulsing glow",
    "glow intensifies",
    "glow brightens",
    "粒子",
    "能量波",
    "光爆",
    "光环",
    "悬浮",
    "变形",
    "光芒变亮",
    "微光变亮",
    "光芒增强",
    "光芒闪烁",
    "微光闪烁",
    "光芒脉动",
    "微光脉动",
)

_SYSTEM_PROMPT = """You write production-ready prompts for MiniMax H3 joint video and audio generation.

Turn the user's idea into one vivid English prompt for a {duration}-second video. First identify the user's facts silently, then preserve every identity, attribute, object, relationship, direction, dialogue, and visible text exactly. An unspecified fact must remain unspecified: never invent clothing, colors, materials, props, locations, weather, time of day, or secondary characters. Never infer music or cultural motifs from nationality, ethnicity, gender, or age. For example, "a Chinese dancer" does not imply a red costume, traditional music, or Chinese architecture; "a woman under an umbrella" does not imply the umbrella's color or a streetlight. Add only camera framing and motion, lighting behavior, chronological motion, and synchronized sound that do not alter the scene.

Translate animal identity precisely. In Chinese, 水豚 means capybara, 豚鼠 means guinea pig, and 龙猫 means chinchilla; never substitute one species for another. A capybara has no cheek pouches or visible tail, so never invent either. Unless the user explicitly asks for them, do not add visual effects, particles, energy waves, light bursts, levitation, slow motion, magical reactions, transformations, or stylized transitions. A glowing object may glow steadily; it does not pulse, chime, float, crackle, explode, or affect nearby objects unless the user says so.

Return exactly these three fields, in this order, with no markdown, preamble, explanation, or analysis:
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...

For integrated_multimodal_description, start with [Shot 1] and use no more than 90 words. Define the initial composition, subject, environment, chronological action, and concrete camera movement. Prefer one coherent continuous shot unless the user explicitly requests cuts. Keep identity, anatomy, objects, directions, and scene geometry stable. Fit the amount of action to {duration} seconds. For dialogue, assign only the speaker a stable ID such as (S1), preserve the user's words and punctuation verbatim, and format them as <d>[Language] exact words</d>. Never assign a speaker ID to a character who does not speak, sing, or vocalize.

For overall_soundscape, use no more than 25 words for concise location ambience and synchronized physical or non-verbal human sounds. For non_diegetic_music, return N/A unless the user explicitly requests background music; if requested, use no more than 15 words describing instrumentation, tempo, rhythm, and dynamics. Keep the entire answer under 130 words. Prefer concrete audiovisual details over abstract praise. Avoid negative-prompt lists, invented logos, captions, punctuation or graphic-symbol overlays, and unsupported story details."""

_STORY_OBSERVER_SYSTEM_PROMPT = """You are a strict visual continuity observer. You receive exactly four chronological frames sampled from the final two seconds of a generated video. Frame 4 is the exact handoff frame for the next clip.

Return one compact JSON object and nothing else with exactly these fields:
"visual_state": a faithful one-sentence {display_language} rendering of visible_motion and last_frame_state with no added facts;
"visible_motion": one sentence describing only the primary visible subject's motion across the frames, ignoring camera motion, background motion, and overlays;
"last_frame_state": one sentence describing the subjects, objects, positions, facing directions, and action phase visible in frame 4;
"identity_observation": one sentence listing only the protagonist traits actually visible in frame 4;
"identity_drift": "none", "uncertain", or one short sentence explaining a visible conflict with canonical_visual_anchor;
"continuity_constraints": exactly three short facts that the next clip must preserve.

Report only visible evidence. canonical_visual_anchor is a comparison target, not permission to claim an invisible trait. Do not infer identity, occupation, intention, emotion, sound, weather, offscreen events, or object function. Do not copy or guess unreadable text. Say "no clear motion" when change is ambiguous. Do not say a subject stops when its position changes across frames. Do not describe a surrounding medium as rising or falling without a visible boundary. A platform barrier or glass wall is not a train unless a vehicle body is clearly visible."""

_STORY_BIBLE_SYSTEM_PROMPT = """You are the continuity editor for a short interactive story. Extract one immutable story bible from the user's premise; do not plot scenes yet.

The premise is canonical. Preserve every stated identity, species, nationality, age, appearance, relationship, object, property, location, goal, and visual style exactly. Never rename or change the protagonist. Distinguish an explicit story goal from a scene seed. If the premise only says that someone walks, sees, watches, notices, approaches, or encounters something, it is a scene seed rather than a goal: create one modest, warm, achievable goal that changes the situation while preserving all facts. The goal must involve discovering a concrete reason, keeping someone safe, returning something, fulfilling a small promise, or solving an ordinary problem; "observe," "approach," "keep watching," and "continue walking" are never sufficient core goals. For example, seeing a dog in the road may become understanding why it remains there and helping it reach a safe place. For an explicit goal, do not invent owners, friends, loved ones, or backstory. For a scene seed only, the later plot may add one ordinary clue and one warm relationship if they are visibly seeded before use; never add injury, tragedy, apocalypse, organizations, magic, technology, or object capabilities. In Chinese, 水豚 means capybara, never guinea pig, chinchilla, or mouse; capybaras have no cheek pouches or visible tail. Do not invent clothing, anatomy, containers, or tools in the immutable bible. A glowing object only glows. visual_anchor includes only explicitly stated appearance; otherwise repeat the identity and size.

Default to a warmhearted cinematic adventure with playful surprise and light suspense. The emotional thread must support the stated core goal. If no relationship exists, use growing courage, persistence, gentle humor, curiosity, or care for the place instead of inventing someone to love, rescue, reward, praise, or recognize the protagonist. Never force romance. Unless explicitly requested, exclude horror, dread, death, injury, imprisonment, cruelty, transformation, grotesque imagery, jump scares, and ominous horror sound.

Return exactly {"story_bible": {...}} in {display_language}. Inside story_bible use every field in this exact order: protagonist_identity, visual_anchor, personality, decision_style, core_goal, weakness, starting_inventory, starting_relationships, non_negotiables, world_setting, genre_tone, visual_style, world_rules, central_mystery, opposing_force, stakes, emotional_thread, ending_direction. starting_inventory contains only objects the premise explicitly says the protagonist already possesses; starting_relationships contains only relationships explicitly present in the premise; otherwise each is an empty array. protagonist_identity closely copies the user's noun phrase. visual_anchor repeats only stated visible traits; if none are stated, repeat protagonist_identity without adding appearance. core_goal copies the user's objective without adding a destination or beneficiary. weakness uses only an explicit or physically unavoidable limitation. world_setting stays inside the premise. world_rules preserve stated properties and otherwise say that ordinary physics applies; never invent activation conditions. stakes are simply failure or delay of core_goal unless the premise states another consequence; a wedding cake does not imply an ongoing wedding or that a wedding depends on the target. central_mystery is a modest question implied by the goal, not a new backstory. opposing_force is an ordinary pressure already implied by the location, scale, timing, or goal—not a new villain. ending_direction achieves the original goal and gives the emotional thread a warm payoff without inventing an event beyond the premise. If the goal is to retrieve or recover an object, the protagonist possesses that exact object at the end and does not give it to someone else."""

_STORY_OPENING_SYSTEM_PROMPT = """You are the plot architect for a five-chapter warmhearted interactive short film. The premise and immutable_story_bible are authoritative.

Build one causal story, not a list of poses or generic helpful actions. Every chapter must change at least one of: a known fact, a relationship, a usable resource, the route, or the cost of the goal. Each change causes the next chapter. Keep suspense gentle and grounded; never use injury, horror, tragedy, magic, visual effects, unexplained object powers, or an instant reward as a plot shortcut.

If the premise is only a scene seed, Chapter 1 may reveal one ordinary visible clue that explains why the encounter matters. The named encounter remains the central counterpart: if a girl sees a dog, develop the girl's relationship and mystery with that dog instead of replacing the story with a passerby. Do not introduce another living character before Chapter 5 when the premise already contains two. Chapter 2 acts on the clue and changes trust or access. Chapter 3 reveals a fact that changes the meaning or route of the goal. Chapter 4 creates a genuine choice between two different future consequences, not watching versus approaching. Chapter 5 resolves the original core goal using only seeded facts and gives a warm visible payoff. A milestone describes an outcome and leaves room for multiple choices.

Return one JSON object in {display_language}. It contains title, opening_summary, and exactly five story_arc objects with chapter, purpose, story_change, and milestone. opening_summary is one feasible {duration}-second continuous shot with two or three connected actions. It must end on a concrete unresolved clue or predicament, never on merely observing, waiting, thinking, or approaching."""

_STORY_SYSTEM_PROMPT = """You are writing the next beat of a warm, lightly suspenseful interactive short film. Design narrative choices only; another model writes the video prompts.

Treat immutable_story_bible and current_story_state as canonical story memory, and visual_observation as canonical for the rendered handoff. Preserve identity, established facts, inventory, ordinary physics, visible geometry, and current motion. Use target_arc_milestone as the required outcome, not as a script to copy. Do not replay an action from recent_choice_summaries.

Continue for 0.5-1 second from the observed pose or motion, then introduce one grounded shared event that grows directly from an established clue, open question, or current goal. The event must make a decision necessary. Write two high-contrast choices with opposite intent, attitude, or risk: engage versus leave, trust versus doubt, reveal versus conceal, take the risk versus play safe, accept versus refuse. Vary the dramatic opposition to fit the actual story instead of repeating one formula. A choice may reject, leave, or walk away, but the world must respond immediately and open a new storyline; it can never simply end the story. Never punish that choice with injury, danger, a crash, guilt, or moralizing. In a leave branch, the encountered subject may safely follow, overtake, redirect, or leave a clue, but must not run into traffic, become trapped, or need rescue. Use a harmless surprise, pursuit, dropped object, contradiction, or new encounter instead. For example, after a girl sees a dog, one branch may have her pet it and discover a collar clue, while the opposite branch has her leave—then the dog follows and drops an object that creates a different mystery. Do not merely vary the tool, direction, body movement, or wording used for the same intention. If both branches must clear the same immediate obstacle, clear it in the shared event and branch only afterward. Bad choices are "push it by hand" versus "push it by foot"; good choices pursue two different consequences or questions. Watching, waiting, looking again, continuing the same action, testing the environment, and a failed attempt are not story choices. Do not base a clue on readable writing, an address, a logo, or tiny symbols that the video model cannot render reliably.

Within each {duration}-second branch, the chosen action must cause a visible result and reveal one concrete new fact. The two facts and ending hooks must be materially different. Each branch must advance the current goal and reach the target milestone while preserving the protagonist's personality. Labels name the consequential in-world decision in at most five words; never label a branch with an abstract strategy. Keep the action physically reachable and limited to two or three connected actions. At most one newly introduced person, animal, or major prop is allowed, and it must be causally connected to an existing clue. Never add horror, injury, cruelty, romance, magic, transformations, particles, light bursts, or other unrequested effects. A practical cost may be a delay, detour, lost position, dirt, water, or mild embarrassment.

Return one JSON object in {display_language} with scene_summary, shared_story_event, dilemma, and exactly two branches. Each branch contains label, hook, story_turn, cost, relationship_change, fact_learned, next_goal, inventory_add, inventory_remove, and new_question. story_turn concisely states the decision, physical cause, and resulting plot change. fact_learned is one specific discovery, not an emotion. next_goal states what this choice makes the protagonist pursue in the following clip; the two next goals must be different and neither may be to keep observing, approach, wait, or retry. The label names that new objective or consequence, not the physical technique used in this clip. hook is the visible ending consequence in at most 18 words. Empty arrays mean no inventory change; use "none" only when appropriate. Return JSON only, without camera directions or video-generation prose."""

_STORY_PROMPT_WRITER_SYSTEM_PROMPT = """You compile two approved interactive-story branches into MiniMax H3 image-to-video prompts. Do not redesign, simplify, merge, or add to either branch.

For each branch, begin exactly at visual_observation.last_frame_state. Use the first 0.5-1 second as a natural continuity bridge from visible_motion, then show the supplied shared_story_event, decision, narrative_turn, visible consequence, realized cost, and cliffhanger in chronological order. Preserve protagonist_identity, canonical_visual_anchor, the last-frame geometry, every continuity constraint, and all established objects. The previous clip is context, not footage to replay: never repeat an approach, touch, recoil, turn, or other action that already happened before last_frame_state. Do not invent anatomy, characters, props, dialogue, object abilities, or story events. A capybara has no visible tail or cheek pouches. Unless explicitly required by the premise, do not add injury, visual effects, particles, energy waves, light bursts, levitation, slow motion, magic, transformations, or stylized transitions. A glowing object only glows steadily and makes no sound.

Return one JSON object with exactly two fields, "a" and "b". Each contains exactly integrated_multimodal_description, overall_soundscape, and non_diegetic_music, all in English. In each integrated_multimodal_description, explicitly continue from <Picture 1>, use one continuous chronological shot, and preserve the full approved story turn; use 35-60 words for 5 seconds and 65-100 words for 10 or 15 seconds. Describe only what the camera sees. Never write meta phrases such as "shared story event," "visible consequence," "realized cost," "cliffhanger," "the branch," or "the story." Keep overall_soundscape under 25 words and describe only established ambience and synchronized physical sounds. Copy current_non_diegetic_music unchanged. Do not add captions, logos, text overlays, negative prompts, or unsupported details."""


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_TEXT_SCHEMA: dict[str, Any] = {"type": "string", "minLength": 1}
_TEXT_ARRAY_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": _TEXT_SCHEMA,
    "maxItems": 12,
}
_H3_PROMPT_SCHEMA = _object_schema({field: _TEXT_SCHEMA for field in PROMPT_FIELDS})
_STORY_BIBLE_SCHEMA = _object_schema(
    {
        "protagonist_identity": _TEXT_SCHEMA,
        "visual_anchor": _TEXT_SCHEMA,
        "personality": _TEXT_SCHEMA,
        "decision_style": _TEXT_SCHEMA,
        "core_goal": _TEXT_SCHEMA,
        "weakness": _TEXT_SCHEMA,
        "starting_inventory": _TEXT_ARRAY_SCHEMA,
        "starting_relationships": _TEXT_ARRAY_SCHEMA,
        "non_negotiables": _TEXT_ARRAY_SCHEMA,
        "world_setting": _TEXT_SCHEMA,
        "genre_tone": _TEXT_SCHEMA,
        "visual_style": _TEXT_SCHEMA,
        "world_rules": _TEXT_ARRAY_SCHEMA,
        "central_mystery": _TEXT_SCHEMA,
        "opposing_force": _TEXT_SCHEMA,
        "stakes": _TEXT_SCHEMA,
        "emotional_thread": _TEXT_SCHEMA,
        "ending_direction": _TEXT_SCHEMA,
    }
)
_STORY_BIBLE_RESPONSE_SCHEMA = _object_schema({"story_bible": _STORY_BIBLE_SCHEMA})
_STORY_ARC_BEAT_SCHEMA = _object_schema(
    {
        "chapter": {"type": "integer", "minimum": 1, "maximum": 5},
        "purpose": _TEXT_SCHEMA,
        "story_change": _TEXT_SCHEMA,
        "milestone": _TEXT_SCHEMA,
    }
)
_STORY_OPENING_SCHEMA = _object_schema(
    {
        "title": _TEXT_SCHEMA,
        "opening_summary": _TEXT_SCHEMA,
        "story_arc": {
            "type": "array",
            "items": _STORY_ARC_BEAT_SCHEMA,
            "minItems": 5,
            "maxItems": 5,
        },
    }
)
_VISUAL_OBSERVATION_SCHEMA = _object_schema(
    {
        "visual_state": _TEXT_SCHEMA,
        "visible_motion": _TEXT_SCHEMA,
        "last_frame_state": _TEXT_SCHEMA,
        "identity_observation": _TEXT_SCHEMA,
        "identity_drift": _TEXT_SCHEMA,
        "continuity_constraints": {
            "type": "array",
            "items": _TEXT_SCHEMA,
            "minItems": 2,
            "maxItems": 6,
        },
    }
)
_STORY_BRANCH_SCHEMA = _object_schema(
    {
        "label": _TEXT_SCHEMA,
        "hook": _TEXT_SCHEMA,
        "story_turn": _TEXT_SCHEMA,
        "cost": _TEXT_SCHEMA,
        "relationship_change": _TEXT_SCHEMA,
        "fact_learned": _TEXT_SCHEMA,
        "next_goal": _TEXT_SCHEMA,
        "inventory_add": _TEXT_ARRAY_SCHEMA,
        "inventory_remove": _TEXT_ARRAY_SCHEMA,
        "new_question": _TEXT_SCHEMA,
    }
)
_STORY_PLAN_SCHEMA = _object_schema(
    {
        "scene_summary": _TEXT_SCHEMA,
        "shared_story_event": _TEXT_SCHEMA,
        "dilemma": _TEXT_SCHEMA,
        "branches": {
            "type": "array",
            "items": _STORY_BRANCH_SCHEMA,
            "minItems": 2,
            "maxItems": 2,
        },
    }
)
_STORY_PROMPTS_SCHEMA = _object_schema({"a": _H3_PROMPT_SCHEMA, "b": _H3_PROMPT_SCHEMA})


def _response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


class PromptEnhancementError(ValueError):
    """Raised when a prompt cannot be enhanced or normalized safely."""


@dataclass(frozen=True, slots=True)
class PromptEnhancementResult:
    prompt: str
    seconds: float
    input_tokens: int
    output_tokens: int
    model: str


@dataclass(frozen=True, slots=True)
class StoryOpeningResult:
    title: str
    opening_summary: str
    story_bible: dict[str, Any]
    story_arc: list[dict[str, Any]]
    initial_state: dict[str, Any]
    prompt: str
    seconds: float
    input_tokens: int
    output_tokens: int
    model: str


@dataclass(frozen=True, slots=True)
class StoryPlanResult:
    visual_state: str
    visual_observation: dict[str, Any]
    scene_summary: str
    branches: tuple[dict[str, Any], dict[str, Any]]
    seconds: float
    input_tokens: int
    output_tokens: int
    model: str
    observer_seconds: float
    director_seconds: float
    prompt_writer_seconds: float
    writer_seconds: float


class OpenAICompatiblePromptEnhancer:
    """Small client for a local OpenAI-compatible prompt-writing model."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_seconds: float = 30,
        story_endpoint: str | None = None,
        story_model: str | None = None,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise PromptEnhancementError("prompt enhancer endpoint must be HTTP(S)")
        if not model.strip():
            raise PromptEnhancementError("prompt enhancer model must not be empty")
        if timeout_seconds <= 0:
            raise PromptEnhancementError("prompt enhancer timeout must be positive")
        if story_endpoint is not None and not story_endpoint.startswith(
            ("http://", "https://")
        ):
            raise PromptEnhancementError("story director endpoint must be HTTP(S)")
        if story_model is not None and not story_model.strip():
            raise PromptEnhancementError("story director model must not be empty")
        self.endpoint = endpoint
        self.model = model.strip()
        self.story_endpoint = story_endpoint or endpoint
        self.story_model = story_model.strip() if story_model else self.model
        self.timeout_seconds = timeout_seconds

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "backend": "openai-compatible",
            "model": self.model,
            "story_model": self.story_model,
        }

    def enhance(
        self, brief: str, *, duration_seconds: int, mode: str = "t2va"
    ) -> PromptEnhancementResult:
        if mode not in {"t2va", "i2va"}:
            raise PromptEnhancementError("prompt enhancement mode must be t2va or i2va")
        payload = {
            "model": self.model,
            "messages": build_h3_prompt_messages(brief, duration_seconds),
            "max_tokens": 224,
            "temperature": 0,
            "seed": 20260903,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        raw, seconds, input_tokens, output_tokens = self._complete(payload)
        prompt = normalize_h3_prompt(raw)
        if mode == "i2va":
            prompt = make_i2va_prompt(prompt)
        return PromptEnhancementResult(
            prompt=prompt,
            seconds=seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )

    def open_story(
        self,
        premise: str,
        *,
        duration_seconds: int,
        display_language: str = "English",
    ) -> StoryOpeningResult:
        """Create the immutable story foundation and first H3-ready chapter."""

        bible_payload = {
            "model": self.story_model,
            "messages": build_story_bible_messages(
                premise, display_language=display_language
            ),
            "max_tokens": 700,
            "temperature": 0,
            "seed": 20260903,
            "response_format": _response_format(
                "h3_flash_story_bible", _STORY_BIBLE_RESPONSE_SCHEMA
            ),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        bible_raw, bible_seconds, bible_input, bible_output = self._complete(
            bible_payload, endpoint=self.story_endpoint
        )
        story_bible = normalize_story_bible(bible_raw, premise=premise)
        if _looks_like_scene_seed(premise):
            title, opening_summary, story_arc, initial_state = _scene_seed_opening(
                premise,
                story_bible=story_bible,
                display_language=display_language,
            )
            opening_seconds = 0.0
            opening_input = 0
            opening_output = 0
        else:
            opening_payload = {
                "model": self.story_model,
                "messages": build_story_opening_messages(
                    premise,
                    story_bible,
                    duration_seconds=duration_seconds,
                    display_language=display_language,
                ),
                "max_tokens": 1000,
                "temperature": 0,
                "seed": 20260903,
                "response_format": _response_format(
                    "h3_flash_story_opening", _STORY_OPENING_SCHEMA
                ),
                "chat_template_kwargs": {"enable_thinking": False},
            }
            raw, opening_seconds, opening_input, opening_output = self._complete(
                opening_payload, endpoint=self.story_endpoint
            )
            (
                title,
                opening_summary,
                story_arc,
                initial_state,
            ) = normalize_story_opening(
                raw,
                story_bible=story_bible,
                allow_visual_effects=_requests_visual_effects(premise),
            )
        opening_brief = (
            f"{premise.strip()}\n\n"
            f"Opening action and cliffhanger: {opening_summary}\n"
            f"Show these stable visible traits: {story_bible['visual_anchor']}.\n"
            f"Express this personality through physical action: {story_bible['personality']}.\n"
            "The original premise is the only authority for identity and appearance. "
            "Ignore any unsupported anatomy, clothing, container, or capability in the "
            "proposed direction. Treat metaphorical wording as mood, never as a physical "
            "effect. A glowing object is steadily lit and silent; do not give it a hum, "
            "chime, pulse, motion, or influence. Do not add any other identity, backstory, "
            "object capability, or plot event."
        )
        enhanced = self.enhance(opening_brief, duration_seconds=duration_seconds)
        prompt = _sanitize_story_h3_prompt(
            enhanced.prompt,
            allow_visual_effects=_requests_visual_effects(premise),
        )
        story_bible["canonical_h3_prompt"] = prompt
        return StoryOpeningResult(
            title=title,
            opening_summary=opening_summary,
            story_bible=story_bible,
            story_arc=story_arc,
            initial_state=initial_state,
            prompt=prompt,
            seconds=bible_seconds + opening_seconds + enhanced.seconds,
            input_tokens=bible_input + opening_input + enhanced.input_tokens,
            output_tokens=bible_output + opening_output + enhanced.output_tokens,
            model=f"{self.story_model} + {self.model}",
        )

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
        context_frame_paths: tuple[Path, ...] = (),
    ) -> StoryPlanResult:
        """Plan two H3-ready continuations for an actual generated tail frame."""

        if len(context_frame_paths) != 4:
            raise PromptEnhancementError(
                "story director requires exactly four context frames"
            )
        frame_data_urls = tuple(_jpeg_data_url(path) for path in context_frame_paths)
        observer_payload = {
            "model": self.model,
            "messages": build_story_observer_messages(
                frame_data_urls,
                display_language=display_language,
                canonical_visual_anchor=(
                    f"{story_bible.get('protagonist_identity', '')}; "
                    f"{story_bible.get('visual_anchor', '')}"
                ),
            ),
            "max_tokens": 384,
            "temperature": 0,
            "seed": 20260903,
            "response_format": _response_format(
                "h3_flash_visual_observation", _VISUAL_OBSERVATION_SCHEMA
            ),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        (
            observer_raw,
            observer_seconds,
            observer_input_tokens,
            observer_output_tokens,
        ) = self._complete(observer_payload)
        visual_observation = normalize_visual_observation(observer_raw)
        writer_payload = {
            "model": self.story_model,
            "messages": build_story_messages(
                premise=premise,
                current_prompt=current_prompt,
                history=history,
                story_bible=story_bible,
                story_arc=story_arc,
                story_state=story_state,
                duration_seconds=duration_seconds,
                display_language=display_language,
                visual_observation=visual_observation,
            ),
            "max_tokens": 720,
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": 20260903 + len(history),
            "response_format": _response_format(
                "h3_flash_story_plan", _STORY_PLAN_SCHEMA
            ),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        raw, director_seconds, director_input_tokens, director_output_tokens = (
            self._complete(writer_payload, endpoint=self.story_endpoint)
        )
        scene_summary, outline_branches = normalize_story_outline(
            raw,
            story_state=story_state,
            story_bible=story_bible,
        )
        violations = story_outline_violations(
            outline_branches,
            target_milestone=_target_story_milestone(story_arc, story_state),
            story_bible=story_bible,
        )
        if violations:
            print(
                "[h3-flash] story quality warnings after deterministic safeguards: "
                + " | ".join(violations),
                flush=True,
            )
        current_music = _h3_prompt_values(current_prompt)["non_diegetic_music"]
        prompt_payload = {
            "model": self.model,
            "messages": build_story_prompt_messages(
                premise=premise,
                current_prompt=current_prompt,
                visual_observation=visual_observation,
                story_bible=story_bible,
                branches=outline_branches,
                duration_seconds=duration_seconds,
            ),
            "max_tokens": 640,
            "temperature": 0,
            "seed": 20260903 + len(history),
            "response_format": _response_format(
                "h3_flash_story_prompts", _STORY_PROMPTS_SCHEMA
            ),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        (
            prompt_raw,
            prompt_writer_seconds,
            prompt_input_tokens,
            prompt_output_tokens,
        ) = self._complete(prompt_payload)
        prompts = normalize_story_prompts(
            prompt_raw,
            non_diegetic_music=current_music,
            allow_visual_effects=_requests_visual_effects(premise),
        )
        branches = attach_story_prompts(outline_branches, prompts)
        writer_seconds = director_seconds + prompt_writer_seconds
        return StoryPlanResult(
            visual_state=visual_observation["visual_state"],
            visual_observation=visual_observation,
            scene_summary=scene_summary,
            branches=branches,
            seconds=observer_seconds + writer_seconds,
            input_tokens=(
                observer_input_tokens + director_input_tokens + prompt_input_tokens
            ),
            output_tokens=(
                observer_output_tokens + director_output_tokens + prompt_output_tokens
            ),
            model=f"{self.model} + {self.story_model}",
            observer_seconds=observer_seconds,
            director_seconds=director_seconds,
            prompt_writer_seconds=prompt_writer_seconds,
            writer_seconds=writer_seconds,
        )

    def _complete(
        self, payload: dict[str, Any], *, endpoint: str | None = None
    ) -> tuple[str, float, int, int]:
        request = urllib.request.Request(
            endpoint or self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = perf_counter()
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                body = json.load(response)
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
            raise PromptEnhancementError(
                f"prompt enhancer request failed: {error}"
            ) from error
        seconds = perf_counter() - started
        try:
            raw = body["choices"][0]["message"]["content"]
            usage = body["usage"]
            input_tokens = int(usage["prompt_tokens"])
            output_tokens = int(usage["completion_tokens"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise PromptEnhancementError(
                "prompt enhancer returned an invalid response"
            ) from error
        if not isinstance(raw, str):
            raise PromptEnhancementError("prompt enhancer returned non-text content")
        return raw, seconds, input_tokens, output_tokens


def build_h3_prompt_messages(brief: str, duration_seconds: int) -> list[dict[str, str]]:
    """Build the deterministic chat request used by the loaded H3 Qwen model."""

    normalized = brief.strip()
    if not normalized:
        raise PromptEnhancementError("prompt must be a non-empty string")
    if duration_seconds not in {5, 10, 15}:
        raise PromptEnhancementError("duration must be 5, 10, or 15 seconds")
    return [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT.format(duration=duration_seconds),
        },
        {
            "role": "user",
            "content": 'A Chinese dancer in a blue tracksuit says "开始吧" and spins once.',
        },
        {
            "role": "assistant",
            "content": (
                "integrated_multimodal_description: [Shot 1] A medium shot "
                "frames a Chinese dancer in a blue tracksuit (S1). The camera "
                "makes a small, steady arc as the dancer says: <d>[Chinese] "
                "开始吧</d> and spins once.\n\n"
                "overall_soundscape: Soft footfalls, clothing movement, and "
                "natural breathing remain synchronized with the spin.\n\n"
                "non_diegetic_music: N/A"
            ),
        },
        {"role": "user", "content": normalized},
    ]


def build_story_bible_messages(
    premise: str, *, display_language: str = "English"
) -> list[dict[str, str]]:
    """Build the focused immutable-character extraction request."""

    normalized = premise.strip()
    if not normalized:
        raise PromptEnhancementError("story premise must not be empty")
    return [
        {
            "role": "system",
            "content": _STORY_BIBLE_SYSTEM_PROMPT.replace(
                "{display_language}", display_language
            ),
        },
        {"role": "user", "content": normalized},
    ]


def build_story_opening_messages(
    premise: str,
    story_bible: dict[str, Any],
    *,
    duration_seconds: int,
    display_language: str = "English",
) -> list[dict[str, str]]:
    """Build the plot request after the immutable story bible is fixed."""

    normalized = premise.strip()
    if not normalized:
        raise PromptEnhancementError("story premise must not be empty")
    if duration_seconds not in {5, 10, 15}:
        raise PromptEnhancementError("duration must be 5, 10, or 15 seconds")
    context = {
        "premise": normalized,
        "scene_seed": _looks_like_scene_seed(normalized),
        "immutable_story_bible": story_bible,
        "opening_contract": {
            "scene_seed_counterpart": (
                "When scene_seed is true, the person, animal, or object already being "
                "observed in the premise is the sole story counterpart. Use that exact "
                "subject in every chapter. Do not add any other person or animal."
            ),
            "chapter_1": (
                "Use only entities explicitly present in the premise; establish one "
                "ordinary physical obstacle and do not touch or acquire an inaccessible "
                "retrieval target. If the premise is only a scene seed, turn the observed "
                "situation into a concrete unanswered question and an achievable warm goal; "
                "do not make watching or approaching the goal. A scene seed may introduce "
                "one ordinary visible clue that causes the story, and that clue must recur."
            ),
            "chapter_2": (
                "If the premise already has a second living subject, keep it as the sole "
                "supporting character and introduce nobody else. Otherwise introduce exactly "
                "one natural supporting character with a concrete different task. It must "
                "not want, own, guard, or retrieve the protagonist's target."
            ),
            "chapters_3_and_4": (
                "Grow reciprocal trust and change the route, clue, or cost without "
                "completing the protagonist's core goal."
            ),
            "chapter_5": (
                "Complete the exact core goal using only seeded characters, facts, and objects."
            ),
        },
    }
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": _STORY_OPENING_SYSTEM_PROMPT.replace(
                "{duration}", str(duration_seconds)
            ).replace("{display_language}", display_language),
        },
    ]
    if display_language == "Simplified Chinese" and _looks_like_scene_seed(normalized):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "premise": "一名男孩路过雨后的公园，看见一只小狗守着路中央的一只黄色雨靴。",
                            "scene_seed": True,
                            "immutable_story_bible": {
                                "protagonist_identity": "路过雨后公园的男孩",
                                "personality": "好奇、耐心、愿意帮助小动物",
                                "core_goal": "弄清小狗守着雨靴的原因，并帮助它安全离开湿滑车道",
                                "weakness": "不能直接知道小狗想表达什么",
                                "starting_inventory": [],
                                "starting_relationships": [],
                                "world_setting": "雨后的公园道路",
                                "world_rules": ["普通物理规律适用"],
                                "emotional_thread": "陌生人与小狗通过线索逐渐建立信任",
                            },
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "title": "雨靴指向的路",
                            "opening_summary": "男孩停在湿滑路边，小狗把黄色雨靴推向他，又急切地望向积水边；一截同色鞋带正被排水栅夹住。",
                            "story_arc": [
                                {
                                    "chapter": 1,
                                    "purpose": "让偶遇变成具体谜题",
                                    "story_change": "男孩发现小狗不是守着雨靴不走，而是在求助取出被排水栅夹住的另一截鞋带。",
                                    "milestone": "男孩确认小狗需要他的帮助，并找到第一个可行动的线索。",
                                },
                                {
                                    "chapter": 2,
                                    "purpose": "行动建立信任",
                                    "story_change": "男孩利用雨靴垫稳栅角取出鞋带，小狗不再后退，主动把雨靴推回他脚边。",
                                    "milestone": "男孩与小狗完成第一次相互配合，并让小狗离开车道边缘。",
                                },
                                {
                                    "chapter": 3,
                                    "purpose": "新事实改变去向",
                                    "story_change": "翻转的雨靴底沾着鲜亮黄漆，与通向公园侧门的一串小脚印一致，小狗随即叼起鞋带领路。",
                                    "milestone": "男孩明白雨靴来自公园侧门方向，故事从解困转为寻找它的来处。",
                                },
                                {
                                    "chapter": 4,
                                    "purpose": "让线索产生真正分支",
                                    "story_change": "黄脚印在水洼前分成通往侧门与长廊的两路；小狗认一条，雨靴上的新泥点指向另一条。",
                                    "milestone": "男孩必须在相信小狗的方向与相信雨靴的新线索之间作出选择，两条路将揭示不同事实。",
                                },
                                {
                                    "chapter": 5,
                                    "purpose": "回收线索并温暖收束",
                                    "story_change": "选择的线索把男孩和小狗带到公园门内，一个穿着另一只黄色雨靴的孩子正焦急等待。",
                                    "milestone": "小狗、成对雨靴和等待的孩子重聚，男孩确认小狗安全后温暖道别。",
                                },
                            ],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
        }
    )
    return messages


def build_story_messages(
    *,
    premise: str,
    current_prompt: str,
    history: list[str],
    story_bible: dict[str, Any] | None = None,
    story_arc: list[dict[str, Any]] | None = None,
    story_state: dict[str, Any] | None = None,
    duration_seconds: int,
    display_language: str = "English",
    visual_observation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the compact story-director request sent to the local Qwen model."""

    normalized_premise = premise.strip()
    normalized_prompt = current_prompt.strip()
    if not normalized_premise or not normalized_prompt:
        raise PromptEnhancementError(
            "story premise and current prompt must not be empty"
        )
    if duration_seconds not in {5, 10, 15}:
        raise PromptEnhancementError("duration must be 5, 10, or 15 seconds")
    if visual_observation is None:
        visual_observation = {
            "visual_state": "No visual observation was supplied.",
            "visible_motion": "No visual observation was supplied.",
            "last_frame_state": "Use the intended prompt cautiously.",
            "continuity_constraints": [],
        }
    story_bible = story_bible or {
        "protagonist_identity": "The protagonist from the premise.",
        "visual_anchor": "Preserve the protagonist visible in the handoff frame.",
        "personality": "Act consistently with the premise.",
        "decision_style": "Make active, physically grounded decisions.",
        "core_goal": normalized_premise,
        "weakness": "Unspecified.",
        "starting_inventory": [],
        "starting_relationships": [],
        "non_negotiables": ["Preserve identity", "Pursue the core goal"],
        "world_setting": "The established setting.",
        "genre_tone": "Coherent cinematic adventure.",
        "visual_style": "Match the rendered clip.",
        "world_rules": ["Ordinary objects stay ordinary"],
        "central_mystery": normalized_premise,
        "opposing_force": "The established obstacle.",
        "stakes": "The protagonist may fail the core goal.",
        "emotional_thread": "A small act of care grows into earned trust.",
        "ending_direction": "Resolve the core goal.",
    }
    story_arc = story_arc or [
        {
            "chapter": index,
            "purpose": "advance the story",
            "story_change": "The protagonist makes meaningful progress toward the goal.",
            "milestone": normalized_premise,
        }
        for index in range(1, 6)
    ]
    story_state = story_state or {
        "chapter": 1,
        "location": "The rendered location.",
        "current_goal": normalized_premise,
        "inventory": [],
        "relationships": [],
        "established_facts": [],
        "open_threads": [],
        "last_decision": "The opening action.",
        "last_consequence": "The rendered handoff.",
        "character_condition": "As shown in the handoff frame.",
        "emotional_progress": "The protagonist has committed to the core goal.",
    }
    milestone = _target_story_milestone(story_arc, story_state)
    director_bible = {
        key: value for key, value in story_bible.items() if key != "canonical_h3_prompt"
    }
    context = {
        "visual_observation": visual_observation,
        "story_context": {
            "premise": normalized_premise,
            "immutable_story_bible": director_bible,
            "five_chapter_arc": story_arc,
            "target_arc_milestone": milestone,
            "current_story_state": story_state,
            "recent_choice_summaries": history,
        },
    }
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _STORY_SYSTEM_PROMPT.format(
                duration=duration_seconds, display_language=display_language
            ),
        }
    ]
    messages.append(
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
        }
    )
    return messages


def _target_story_milestone(
    story_arc: list[dict[str, Any]], story_state: dict[str, Any]
) -> dict[str, Any]:
    current_chapter = story_state.get("chapter", 1)
    next_chapter = current_chapter + 1 if isinstance(current_chapter, int) else 2
    return next(
        (
            item
            for item in story_arc
            if isinstance(item, dict) and item.get("chapter") == next_chapter
        ),
        story_arc[-1],
    )


def build_story_prompt_messages(
    *,
    premise: str,
    current_prompt: str,
    visual_observation: dict[str, Any],
    story_bible: dict[str, Any],
    branches: tuple[dict[str, Any], dict[str, Any]],
    duration_seconds: int,
) -> list[dict[str, str]]:
    """Compile approved narrative turns into two continuity-safe H3 prompts."""

    if duration_seconds not in {5, 10, 15}:
        raise PromptEnhancementError("duration must be 5, 10, or 15 seconds")
    current_values = _h3_prompt_values(current_prompt)
    branch_fields = (
        "id",
        "label",
        "shared_story_event",
        "decision",
        "narrative_turn",
        "cost",
        "fact_learned",
        "next_goal",
        "hook",
        "new_thread",
    )
    context = {
        "duration_seconds": duration_seconds,
        "premise": premise.strip(),
        "protagonist_identity": story_bible.get("protagonist_identity", ""),
        "canonical_visual_anchor": story_bible.get("visual_anchor", ""),
        "visual_style": story_bible.get("visual_style", ""),
        "visual_observation": visual_observation,
        "current_non_diegetic_music": current_values["non_diegetic_music"],
        "approved_branches": [
            {field: branch[field] for field in branch_fields} for branch in branches
        ],
    }
    return [
        {"role": "system", "content": _STORY_PROMPT_WRITER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def build_story_observer_messages(
    frame_data_urls: tuple[str, ...],
    *,
    display_language: str = "English",
    canonical_visual_anchor: str = "",
) -> list[dict[str, Any]]:
    """Build the image-only observation pass for the story director."""

    if len(frame_data_urls) != 4:
        raise PromptEnhancementError(
            "story observer requires exactly four context frames"
        )
    if any(
        not value.startswith("data:image/jpeg;base64,") for value in frame_data_urls
    ):
        raise PromptEnhancementError("story context frames must be JPEG data URLs")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Frames 1 through 4 follow in chronological order. "
                "Compare visible protagonist traits against this canonical_visual_anchor: "
                f"{canonical_visual_anchor or 'not supplied'}"
            ),
        }
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": data_url}}
        for data_url in frame_data_urls
    )
    return [
        {
            "role": "system",
            "content": _STORY_OBSERVER_SYSTEM_PROMPT.format(
                display_language=display_language
            ),
        },
        {"role": "user", "content": content},
    ]


def _jpeg_data_url(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PromptEnhancementError(
            f"story context frame is not readable: {path.name}"
        ) from error
    if not payload.startswith(b"\xff\xd8"):
        raise PromptEnhancementError(f"story context frame is not JPEG: {path.name}")
    return f"data:image/jpeg;base64,{b64encode(payload).decode('ascii')}"


def make_i2va_prompt(prompt: str) -> str:
    """Add MiniMax's official first-frame alignment contract to a core prompt."""

    instruction = (
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced."
    )
    if prompt.lstrip().startswith(instruction):
        return prompt.strip()
    marker = "integrated_multimodal_description: [Shot 1]"
    anchored = prompt.strip().replace(
        marker,
        f"{marker} <Picture 1> establishes the opening composition and subjects. ",
        1,
    )
    return f"{instruction}\n\n{anchored}"


def normalize_h3_prompt(raw: str) -> str:
    """Extract and validate the three official H3 prompt fields."""

    text = raw.strip()
    text = re.sub(r"```(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<|im_end|>", "").strip()

    matches = list(
        re.finditer(
            r"(?im)^\s*(integrated_multimodal_description|overall_soundscape|non_diegetic_music)\s*:\s*",
            text,
        )
    )
    if [match.group(1).lower() for match in matches] != list(PROMPT_FIELDS):
        raise PromptEnhancementError(
            "H3 text encoder did not return the required three-field prompt"
        )

    values: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end() : end].strip()
        if not value:
            raise PromptEnhancementError(
                f"H3 text encoder returned an empty {PROMPT_FIELDS[index]} field"
            )
        values.append(value)
    return "\n\n".join(
        f"{field}: {value}" for field, value in zip(PROMPT_FIELDS, values, strict=True)
    )


_STORY_BIBLE_FIELDS = (
    "protagonist_identity",
    "visual_anchor",
    "personality",
    "decision_style",
    "core_goal",
    "weakness",
    "world_setting",
    "genre_tone",
    "visual_style",
    "central_mystery",
    "opposing_force",
    "stakes",
    "emotional_thread",
    "ending_direction",
)


def _scene_seed_opening(
    premise: str,
    *,
    story_bible: dict[str, Any],
    display_language: str,
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    """Create an open-ended arc when the premise describes an encounter, not a plot."""

    if display_language == "Simplified Chinese":
        title = "未完的相遇"
        opening_summary = (
            f"{premise.strip().rstrip('。！？!?')}。对方作出一个小而明确的回应，"
            "主角停在主动介入还是转身离开的决定前。"
        )
        story_arc = [
            {
                "chapter": 1,
                "purpose": "建立偶遇与选择",
                "story_change": "眼前对象对主角作出明确回应，这次偶遇变成必须决定是否介入的时刻。",
                "milestone": "主角面对两个意图相反、都会产生后果的选择。",
            },
            {
                "chapter": 2,
                "purpose": "让选择触发故事",
                "story_change": "无论介入还是离开，眼前对象或环境都会立即回应，并揭示不同的具体线索。",
                "milestone": "两个相反选择各自触发新事件，并建立不同的下一目标。",
            },
            {
                "chapter": 3,
                "purpose": "让线索改变理解",
                "story_change": "新线索揭示这次偶遇背后的普通原因，改变主角接下来的路线或关系。",
                "milestone": "主角查明核心谜题的一部分，并承担早先选择带来的后果。",
            },
            {
                "chapter": 4,
                "purpose": "让前因产生新的抉择",
                "story_change": "早先选择的后果回到当前局面，形成一次价值或风险上的新选择。",
                "milestone": "主角用行动决定如何完成核心目标。",
            },
            {
                "chapter": 5,
                "purpose": "温暖收束",
                "story_change": "已经建立的线索和关系共同解决原始问题。",
                "milestone": "主角完成核心目标，故事以可见而温暖的回报结束。",
            },
        ]
    else:
        title = "An Unfinished Encounter"
        opening_summary = (
            f"{premise.strip().rstrip('.!?')}. The other subject makes one small, "
            "clear response, leaving the protagonist poised between engaging and "
            "walking away."
        )
        story_arc = [
            {
                "chapter": 1,
                "purpose": "Establish the encounter and choice",
                "story_change": "The encountered subject responds, turning a passing moment into a consequential decision.",
                "milestone": "The protagonist faces two opposite choices that will both have consequences.",
            },
            {
                "chapter": 2,
                "purpose": "Let the decision start the story",
                "story_change": "Whether the protagonist engages or leaves, the subject or setting responds with a different concrete clue.",
                "milestone": "The opposite choices trigger different events and establish different next goals.",
            },
            {
                "chapter": 3,
                "purpose": "Change the meaning of the encounter",
                "story_change": "The chosen clue reveals an ordinary cause behind the encounter and changes the route or relationship.",
                "milestone": "The protagonist learns part of the central answer and faces the consequence of the earlier choice.",
            },
            {
                "chapter": 4,
                "purpose": "Make prior consequences create a new choice",
                "story_change": "An earlier consequence returns and creates a new decision about values or risk.",
                "milestone": "The protagonist commits to a way of completing the core goal.",
            },
            {
                "chapter": 5,
                "purpose": "Deliver a warm resolution",
                "story_change": "Established clues and relationships resolve the original problem together.",
                "milestone": "The protagonist completes the core goal and receives a visible, warm payoff.",
            },
        ]
    initial_state = _normalize_initial_story_state(
        story_bible=story_bible,
        opening_summary=opening_summary,
    )
    return title, opening_summary, story_arc, initial_state


def normalize_story_bible(raw: str, *, premise: str | None = None) -> dict[str, Any]:
    """Validate the premise-derived facts that remain fixed for every chapter."""

    value = _json_object(raw, source="story bible editor")
    raw_bible = value.get("story_bible")
    if not isinstance(raw_bible, dict):
        raise PromptEnhancementError("story bible editor returned no story bible")
    story_bible = {
        field: _required_text(raw_bible.get(field), f"story bible {field}", maximum=480)
        for field in _STORY_BIBLE_FIELDS
    }
    for field in (
        "starting_inventory",
        "starting_relationships",
        "non_negotiables",
        "world_rules",
    ):
        items = _text_list(raw_bible.get(field), f"story bible {field}", maximum=6)
        if not items and field == "non_negotiables":
            if re.search(r"[\u3400-\u9fff]", str(story_bible["core_goal"])):
                items = ["保持主角身份与核心目标不变"]
            else:
                items = ["Keep the protagonist identity and core goal unchanged"]
        elif not items and field == "world_rules":
            if re.search(r"[\u3400-\u9fff]", str(story_bible["core_goal"])):
                items = ["遵循普通物理规律"]
            else:
                items = ["Ordinary physics applies"]
        story_bible[field] = items
    if premise is not None:
        normalized_premise = premise.strip()
        explicit_stakes = any(
            marker in normalized_premise.casefold()
            for marker in (
                "otherwise",
                "or else",
                "before ",
                "if ",
                "否则",
                "不然",
                "如果",
                "赶在",
                "之前",
                "以免",
            )
        )
        chinese = bool(re.search(r"[\u3400-\u9fff]", normalized_premise))
        if not explicit_stakes:
            story_bible["stakes"] = (
                f"如果未成功，核心目标会被延迟：{story_bible['core_goal']}"
                if chinese
                else f"Failure would delay the core goal: {story_bible['core_goal']}"
            )
        if _looks_like_scene_seed(normalized_premise) and any(
            marker in str(story_bible["core_goal"]).casefold()
            for marker in (
                "observe",
                "watch",
                "look at",
                "approach",
                "keep walking",
                "观察",
                "看着",
                "注视",
                "靠近",
                "继续走",
            )
        ):
            story_bible["core_goal"] = (
                "弄清眼前异常状况的具体原因，并以温和、安全的方式让局面发生有意义的改变"
                if chinese
                else (
                    "Discover the concrete reason behind the unusual encounter and "
                    "bring it to a warm, safe, meaningful change"
                )
            )
            if not explicit_stakes:
                story_bible["stakes"] = (
                    f"如果未成功，核心目标会被延迟：{story_bible['core_goal']}"
                    if chinese
                    else (
                        f"Failure would delay the core goal: {story_bible['core_goal']}"
                    )
                )
        if _looks_like_scene_seed(normalized_premise):
            lowered_premise = normalized_premise.casefold()
            if "狗" in lowered_premise and any(
                marker in lowered_premise for marker in ("马路", "道路", "车道")
            ):
                story_bible["core_goal"] = (
                    "弄清小狗停在马路上的原因，并帮助它安全离开车道"
                )
            elif "dog" in lowered_premise and any(
                marker in lowered_premise for marker in ("road", "street", "traffic")
            ):
                story_bible["core_goal"] = (
                    "Discover why the dog remains in the road and help it leave the "
                    "traffic lane safely"
                )
            if not explicit_stakes:
                story_bible["stakes"] = (
                    f"如果未成功，核心目标会被延迟：{story_bible['core_goal']}"
                    if chinese
                    else (
                        f"Failure would delay the core goal: {story_bible['core_goal']}"
                    )
                )
        if _is_retrieval_goal(str(story_bible["core_goal"])):
            story_bible["central_mystery"] = (
                f"主角将如何完成核心目标：{story_bible['core_goal']}"
                if chinese
                else (
                    "How will the protagonist complete the core goal: "
                    f"{story_bible['core_goal']}"
                )
            )
            story_bible["ending_direction"] = (
                f"主角成功完成并保持：{story_bible['core_goal']}；"
                f"情感回报体现为：{story_bible['emotional_thread']}"
                if chinese
                else (
                    "The protagonist completes and keeps the retrieval goal: "
                    f"{story_bible['core_goal']}; the emotional payoff fulfills: "
                    f"{story_bible['emotional_thread']}"
                )
            )
    return story_bible


def normalize_story_opening(
    raw: str,
    *,
    story_bible: dict[str, Any],
    allow_visual_effects: bool = False,
) -> tuple[
    str,
    str,
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Validate the five-chapter arc after immutable facts are fixed."""

    value = _json_object(raw, source="story opening director")
    title = _required_text(value.get("title"), "story title", maximum=120)
    opening_summary = _required_text(
        value.get("opening_summary"), "story opening summary", maximum=320
    )
    raw_arc = value.get("story_arc")
    if not isinstance(raw_arc, list) or len(raw_arc) != 5:
        raise PromptEnhancementError("story opening must return a five-chapter arc")
    story_arc: list[dict[str, Any]] = []
    for expected_chapter, beat in enumerate(raw_arc, start=1):
        if not isinstance(beat, dict):
            raise PromptEnhancementError("story arc beat must be a JSON object")
        story_arc.append(
            {
                "chapter": expected_chapter,
                "purpose": _required_text(
                    beat.get("purpose"), "story arc purpose", maximum=240
                ),
                "story_change": _required_text(
                    beat.get("story_change"), "story arc change", maximum=360
                ),
                "milestone": _required_text(
                    beat.get("milestone"), "story arc milestone", maximum=360
                ),
            }
        )
    if any(
        marker in str(story_bible.get("protagonist_identity", "")).casefold()
        for marker in ("capybara", "水豚")
    ):
        opening_summary = _sanitize_capybara_anatomy(opening_summary)
        story_arc = _sanitize_capybara_anatomy(story_arc)
    if not allow_visual_effects:
        opening_summary = _sanitize_unrequested_effects(opening_summary)
        story_arc = _sanitize_unrequested_effects(story_arc)
    if _is_retrieval_goal(str(story_bible["core_goal"])):
        if re.search(r"[\u3400-\u9fff]", str(story_bible["core_goal"])):
            story_arc[-1]["story_change"] = (
                f"主角完成并保持原始取回目标：{story_bible['core_goal']}；"
                f"温馨回报兑现：{story_bible['emotional_thread']}。"
            )
            story_arc[-1]["milestone"] = (
                f"{story_bible['protagonist_identity']}在画面中完成并保持："
                f"{story_bible['core_goal']}。"
            )
        else:
            story_arc[-1]["story_change"] = (
                "The protagonist completes and keeps the exact retrieval goal: "
                f"{story_bible['core_goal']}. The warm payoff fulfills: "
                f"{story_bible['emotional_thread']}."
            )
            story_arc[-1]["milestone"] = (
                f"{story_bible['protagonist_identity']} visibly completes and keeps: "
                f"{story_bible['core_goal']}."
            )

    initial_state = _normalize_initial_story_state(
        story_bible=story_bible,
        opening_summary=opening_summary,
    )
    return (
        title,
        opening_summary,
        story_arc,
        initial_state,
    )


def normalize_story_outline(
    raw: str,
    *,
    story_state: dict[str, Any] | None = None,
    story_bible: dict[str, Any] | None = None,
) -> tuple[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Validate two narrative branches before any video prompt is written."""

    value = _json_object(raw, source="story director")
    summary = value.get("scene_summary")
    shared_story_event = _optional_branch_text(
        value, "shared_story_event", "", maximum=480
    )
    dilemma = _optional_branch_text(value, "dilemma", shared_story_event, maximum=360)
    raw_branches = value.get("branches")
    if not isinstance(summary, str) or not summary.strip():
        raise PromptEnhancementError("story director returned no scene summary")
    summary = summary.strip()
    if len(summary) > 320:
        raise PromptEnhancementError("story scene summary is too long")
    if not isinstance(raw_branches, list) or len(raw_branches) != 2:
        raise PromptEnhancementError("story director must return exactly two branches")

    branches: list[dict[str, Any]] = []
    for index, branch in enumerate(raw_branches):
        if not isinstance(branch, dict):
            raise PromptEnhancementError("story branch must be a JSON object")
        branch = _stabilize_leave_branch(branch, story_bible=story_bible)
        branch = _stabilize_dog_traffic_branch(branch, story_bible=story_bible)
        label = branch.get("label")
        hook = branch.get("hook")
        if not all(isinstance(item, str) and item.strip() for item in (label, hook)):
            raise PromptEnhancementError("story branch is missing label or hook")
        label = label.strip()
        hook = hook.strip()
        if len(label) > 64 or len(hook) > 240:
            raise PromptEnhancementError("story branch label or hook is too long")
        decision = _optional_branch_text(branch, "decision", label)
        story_turn = _optional_branch_text(branch, "story_turn", "")
        causal_link = _optional_branch_text(branch, "causal_link", story_turn or hook)
        persona_expression = _optional_branch_text(
            branch, "persona_expression", decision
        )
        visible_consequence = _optional_branch_text(branch, "visible_consequence", hook)
        narrative_turn = _optional_branch_text(
            branch, "narrative_turn", story_turn or visible_consequence
        )
        cost = _optional_branch_text(branch, "cost", "No additional cost established.")
        relationship_change = _optional_branch_text(
            branch, "relationship_change", "none"
        )
        emotional_beat = _optional_branch_text(
            branch,
            "emotional_beat",
            relationship_change
            if relationship_change.casefold() != "none"
            else persona_expression,
        )
        fact_learned = _optional_branch_text(branch, "fact_learned", "none")
        next_goal = _optional_branch_text(branch, "next_goal", "unchanged")
        resolved_thread_id = _optional_branch_text(
            branch, "resolved_thread_id", "none", maximum=80
        )
        new_thread = _optional_branch_text(
            branch,
            "new_thread",
            _optional_branch_text(branch, "new_question", "none", maximum=360),
            maximum=360,
        )
        if re.fullmatch(r"thread[-_ ]?\d+[a-z]?", new_thread, re.IGNORECASE):
            new_thread = dilemma
        thread_urgency = _optional_branch_text(
            branch, "thread_urgency", "later", maximum=120
        )
        if isinstance(branch.get("state_delta"), dict):
            state_delta = _normalize_state_delta(branch["state_delta"])
        else:
            relationships_add = (
                []
                if relationship_change.casefold() in {"none", "unchanged"}
                else [relationship_change]
            )
            facts_add = (
                []
                if fact_learned.casefold() in {"none", "unchanged"}
                else [fact_learned]
            )
            state_delta = _normalize_state_delta(
                {
                    "location": branch.get("next_location", "unchanged"),
                    "current_goal": next_goal,
                    "inventory_add": branch.get("inventory_add", []),
                    "inventory_remove": branch.get("inventory_remove", []),
                    "relationships_add": relationships_add,
                    "facts_add": facts_add,
                    "character_condition": cost,
                    "emotional_progress": emotional_beat,
                }
            )
        story_memory = (
            f"{shared_story_event} {decision} {narrative_turn} "
            f"{visible_consequence} {emotional_beat} Cost: {cost} "
            f"Next: {new_thread}"
        ).strip()
        next_state = (
            _apply_story_state(
                story_state,
                state_delta=state_delta,
                decision=decision,
                consequence=visible_consequence,
                resolved_thread_id=resolved_thread_id,
                new_thread=new_thread,
                thread_urgency=thread_urgency,
                branch_id="a" if index == 0 else "b",
            )
            if story_state is not None
            else None
        )
        branches.append(
            {
                "id": "a" if index == 0 else "b",
                "label": label,
                "hook": hook,
                "story_memory": story_memory,
                "shared_story_event": shared_story_event,
                "dilemma": dilemma,
                "decision": decision,
                "causal_link": causal_link,
                "narrative_turn": narrative_turn,
                "persona_expression": persona_expression,
                "emotional_beat": emotional_beat,
                "visible_consequence": visible_consequence,
                "cost": cost,
                "fact_learned": fact_learned,
                "next_goal": next_goal,
                "resolved_thread_id": resolved_thread_id,
                "new_thread": new_thread,
                "state_delta": state_delta,
            }
        )
        if next_state is not None:
            branches[-1]["next_state"] = next_state
    if story_bible is not None and any(
        marker in str(story_bible.get("protagonist_identity", "")).casefold()
        for marker in ("capybara", "水豚")
    ):
        branches = [_sanitize_capybara_anatomy(branch) for branch in branches]
    if branches[0]["label"].casefold() == branches[1]["label"].casefold():
        branches[0]["label"] = f"A · {branches[0]['label']}"
        branches[1]["label"] = f"B · {branches[1]['label']}"
    return summary, (branches[0], branches[1])


def _stabilize_leave_branch(
    branch: dict[str, Any], *, story_bible: dict[str, Any] | None
) -> dict[str, Any]:
    """Keep an animal-encounter leave choice surprising without punishing it."""

    if story_bible is None:
        return branch
    label = str(branch.get("label", ""))
    lowered_label = label.casefold()
    if not any(
        marker in lowered_label
        for marker in ("leave", "walk away", "turn away", "离开", "转身走")
    ):
        return branch
    canon = json.dumps(story_bible, ensure_ascii=False).casefold()
    if not any(marker in canon for marker in ("dog", "puppy", "小狗", "狗狗")):
        return branch
    stabilized = dict(branch)
    if re.search(r"[\u3400-\u9fff]", label):
        stabilized.update(
            {
                "label": "转身离开",
                "hook": "她刚走几步，小狗安全地跟上，把项圈上的红色小玩具轻轻放在她脚边。",
                "story_turn": "女孩转身沿人行道离开；小狗没有留在车道，而是安全地跟上，并主动把项圈上的红色小玩具放到她脚边。",
                "cost": "女孩原本的行程被这位意外同行者稍稍耽搁。",
                "relationship_change": "小狗选择信任并跟随女孩",
                "fact_learned": "小狗不是要她留下，而是坚持让她收下项圈上的红色小玩具",
                "next_goal": "弄清小狗为什么坚持让她收下红色小玩具",
                "new_question": "这个红色小玩具为什么对小狗如此重要？",
            }
        )
    else:
        stabilized.update(
            {
                "label": "Walk away",
                "hook": "After a few steps, the dog safely follows and places a small red collar toy at her feet.",
                "story_turn": "She walks away along the sidewalk; the dog safely follows and places a small red toy from its collar at her feet.",
                "cost": "The unexpected companion slightly delays her original journey.",
                "relationship_change": "The dog chooses to trust and follow her",
                "fact_learned": "The dog wants her to take the small red toy from its collar",
                "next_goal": "Discover why the dog insists that she take the small red toy",
                "new_question": "Why is the small red toy so important to the dog?",
            }
        )
    return stabilized


def _stabilize_dog_traffic_branch(
    branch: dict[str, Any], *, story_bible: dict[str, Any] | None
) -> dict[str, Any]:
    """Replace a punitive run-into-traffic beat with a safe clue-bearing response."""

    if story_bible is None:
        return branch
    canon = json.dumps(story_bible, ensure_ascii=False).casefold()
    if not any(marker in canon for marker in ("dog", "puppy", "小狗", "狗狗")):
        return branch
    branch_text = json.dumps(branch, ensure_ascii=False).casefold()
    unsafe_motion = (
        "run into traffic",
        "run across the road",
        "runs toward the road",
        "冲向马路",
        "跑向马路",
        "跑到马路",
        "跑向车流",
        "滚向马路",
        "车流方向",
        "紧急刹车",
        "钻入车底",
    )
    if not any(marker in branch_text for marker in unsafe_motion):
        return branch
    stabilized = dict(branch)
    if re.search(r"[\u3400-\u9fff]", str(branch.get("label", ""))):
        stabilized.update(
            {
                "label": "伸手摸摸小狗",
                "hook": "她轻抚小狗，小狗放松下来；转动的项圈露出一只沾着草叶的红色小玩具。",
                "story_turn": "女孩伸手轻抚小狗；小狗放松地留在人行道边，转动的项圈露出一只沾着新鲜草叶的红色小玩具，并望向绿树街角。",
                "cost": "女孩停下原本的脚步，需要为这条新线索绕一点路。",
                "relationship_change": "小狗接受女孩的触碰并开始信任她",
                "fact_learned": "项圈上的红色小玩具沾着来自绿树街角的新鲜草叶",
                "next_goal": "带小狗沿着草叶线索寻找红色小玩具的来处",
                "new_question": "绿树街角有什么让小狗如此在意？",
            }
        )
    else:
        stabilized.update(
            {
                "label": "Pet the dog",
                "hook": "She pets the dog; its turning collar reveals a small red toy flecked with fresh leaves.",
                "story_turn": "She gently pets the dog; it relaxes safely by the sidewalk, revealing a small red collar toy flecked with fresh leaves before looking toward a leafy corner.",
                "cost": "She pauses her journey and must take a small detour for the new clue.",
                "relationship_change": "The dog accepts her touch and begins to trust her",
                "fact_learned": "The small red collar toy carries fresh leaves from the leafy street corner",
                "next_goal": "Follow the leaf clue with the dog to discover where the red toy came from",
                "new_question": "What at the leafy corner matters so much to the dog?",
            }
        )
    return stabilized


def normalize_story_prompts(
    raw: str,
    *,
    non_diegetic_music: str | None = None,
    allow_visual_effects: bool = False,
) -> dict[str, str]:
    """Validate the separate H3 compilation pass for both approved branches."""

    value = _json_object(raw, source="story prompt writer")
    prompts: dict[str, str] = {}
    for branch_id in ("a", "b"):
        normalized = _normalize_story_prompt(value.get(branch_id))
        if non_diegetic_music is not None:
            fields = _h3_prompt_values(normalized)
            fields["non_diegetic_music"] = non_diegetic_music.strip()
            normalized = "\n\n".join(
                f"{field}: {fields[field]}" for field in PROMPT_FIELDS
            )
        normalized = _sanitize_story_h3_prompt(
            normalized, allow_visual_effects=allow_visual_effects
        )
        prompts[branch_id] = make_i2va_prompt(normalized)
    return prompts


def story_outline_violations(
    branches: tuple[dict[str, Any], dict[str, Any]],
    *,
    target_milestone: dict[str, Any],
    story_bible: dict[str, Any],
) -> list[str]:
    """Find common low-quality story patterns before video prompts are compiled."""

    violations: list[str] = []
    target_text = json.dumps(target_milestone, ensure_ascii=False).casefold()
    relationship_required = any(
        marker in target_text
        for marker in ("relationship", "trust", "关系", "信任", "合作", "互助")
    )
    capybara_story = any(
        marker in str(story_bible.get("protagonist_identity", "")).casefold()
        for marker in ("capybara", "水豚")
    )
    dead_end_markers = (
        "no change",
        "miss the chance",
        "仍保持独立",
        "未有明显变化",
        "没有变化",
        "错过建立",
        "故事结束",
        "story ends",
    )
    harm_markers = (
        "injur",
        "pain",
        "wound",
        "mortal danger",
        "guilt",
        "受伤",
        "疼痛",
        "划伤",
        "生命危险",
        "更大的危险",
        "内疚",
        "冲向马路",
        "跑向马路",
        "紧急刹车",
        "钻入车底",
        "被困",
        "困住",
    )
    effects_allowed = _requests_visual_effects(
        json.dumps(story_bible, ensure_ascii=False)
    )
    for branch in branches:
        branch_id = branch["id"]
        label_text = str(branch.get("label", "")).casefold()
        story_fields = " ".join(
            str(branch.get(field, ""))
            for field in (
                "decision",
                "shared_story_event",
                "dilemma",
                "narrative_turn",
                "visible_consequence",
                "hook",
                "cost",
                "emotional_beat",
                "fact_learned",
                "next_goal",
                "new_thread",
            )
        ).casefold()
        if any(marker in story_fields for marker in dead_end_markers):
            violations.append(
                f"Branch {branch_id} is a dead end or unchanged story; make the world "
                "respond to the decision and open a distinct storyline."
            )
        if any(
            marker in label_text
            for marker in (
                "keep watching",
                "continue watching",
                "observe longer",
                "move closer",
                "keep approaching",
                "继续观察",
                "保持观察",
                "再观察",
                "继续靠近",
                "靠近小狗",
                "观察小狗",
                "观察柴犬",
                "观察",
                "提出合作计划",
                "制定合作计划",
                "继续帮助",
                "继续行动",
                "cooperate",
                "make a plan",
            )
        ):
            violations.append(
                f"Branch {branch_id} is only a camera-visible action, not a story choice; "
                "replace it with a decision that reveals a fact, changes a relationship, "
                "or creates a different consequence."
            )
        if any(
            marker in story_fields
            for marker in ("fail", "gets stuck", "失败", "无法成功", "卡住")
        ):
            violations.append(
                f"Branch {branch_id} is only a failed attempt; make it produce positive "
                "story progress through a distinct form of cooperation."
            )
        if any(marker in story_fields for marker in harm_markers):
            violations.append(
                f"Branch {branch_id} uses bodily harm as a cost; replace it with a gentle, "
                "reversible practical cost."
            )
        if not effects_allowed and _has_unsupported_effect(story_fields):
            violations.append(
                f"Branch {branch_id} uses an unsupported visual effect; replace it with a "
                "grounded physical or relational consequence."
            )
        if relationship_required and not branch["state_delta"]["relationships_add"]:
            violations.append(
                f"Branch {branch_id} fails the target relationship change; both branches "
                "must add the newly earned relationship."
            )
        if re.fullmatch(r"thread[-_ ]?\d+[a-z]?", branch["new_thread"], re.IGNORECASE):
            violations.append(
                f"Branch {branch_id} uses a placeholder thread ID; new_thread must be an "
                "actual unanswered story question."
            )
        if capybara_story and any(
            marker in story_fields for marker in ("tail", "尾巴", "尾部", "颊囊")
        ):
            violations.append(
                f"Branch {branch_id} invents capybara anatomy; a capybara has no visible "
                "tail or cheek pouches."
            )
    if (
        branches[0]["narrative_turn"].casefold()
        == branches[1]["narrative_turn"].casefold()
    ):
        violations.append(
            "The two branches have the same story turn; make the outcomes distinct."
        )
    first_fact = str(branches[0].get("fact_learned", "")).casefold()
    second_fact = str(branches[1].get("fact_learned", "")).casefold()
    if first_fact in {"", "none", "unchanged"} or second_fact in {
        "",
        "none",
        "unchanged",
    }:
        violations.append("Each branch must reveal one concrete new fact.")
    elif first_fact == second_fact:
        violations.append("The two branches reveal the same fact; make them diverge.")
    first_goal = str(branches[0].get("next_goal", "")).casefold()
    second_goal = str(branches[1].get("next_goal", "")).casefold()
    passive_goal_markers = (
        "observe",
        "watch",
        "wait",
        "approach",
        "retry",
        "观察",
        "看着",
        "等待",
        "靠近",
        "再试",
        "继续尝试",
    )
    if first_goal in {"", "none", "unchanged"} or second_goal in {
        "",
        "none",
        "unchanged",
    }:
        violations.append("Each branch must establish a concrete next story goal.")
    elif first_goal == second_goal:
        violations.append(
            "The two branches lead to the same next goal; make them diverge."
        )
    for index, goal in enumerate((first_goal, second_goal), start=1):
        if any(marker in goal for marker in passive_goal_markers):
            violations.append(
                f"Branch {index} has a passive next goal; replace it with a new objective."
            )
    return violations


def attach_story_prompts(
    branches: tuple[dict[str, Any], dict[str, Any]], prompts: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach compiled H3 prompts without allowing the compiler to alter story state."""

    result = tuple({**branch, "prompt": prompts[branch["id"]]} for branch in branches)
    return result[0], result[1]


def normalize_story_plan(
    raw: str,
    *,
    non_diegetic_music: str | None = None,
    story_state: dict[str, Any] | None = None,
    story_bible: dict[str, Any] | None = None,
) -> tuple[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Backward-compatible validator for a combined director response."""

    summary, branches = normalize_story_outline(
        raw, story_state=story_state, story_bible=story_bible
    )
    value = _json_object(raw, source="story director")
    raw_branches = value["branches"]
    prompt_payload = json.dumps(
        {
            "a": raw_branches[0].get("prompt"),
            "b": raw_branches[1].get("prompt"),
        },
        ensure_ascii=False,
    )
    prompts = normalize_story_prompts(
        prompt_payload, non_diegetic_music=non_diegetic_music
    )
    return summary, attach_story_prompts(branches, prompts)


def _json_object(raw: str, *, source: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).replace("```", "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise PromptEnhancementError(f"{source} did not return a JSON object")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise PromptEnhancementError(f"{source} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise PromptEnhancementError(f"{source} response must be a JSON object")
    return value


def _is_retrieval_goal(goal: str) -> bool:
    lowered = goal.casefold()
    return any(
        marker in lowered
        for marker in ("retrieve", "recover", "get back", "取回", "找回", "拿回")
    )


def _looks_like_scene_seed(premise: str) -> bool:
    lowered = premise.casefold()
    explicit_goal_markers = (
        " wants ",
        " must ",
        " needs to ",
        " tries to ",
        " attempts to ",
        " has to ",
        " retrieve",
        " recover",
        " rescue",
        " deliver",
        " find ",
        " escape",
        "想要",
        "想把",
        "要把",
        "必须",
        "需要",
        "试图",
        "尝试",
        "取回",
        "找回",
        "寻找",
        "救助",
        "营救",
        "送到",
        "逃离",
        "保护",
        "阻止",
        "完成",
    )
    padded = f" {lowered} "
    return not any(marker in padded for marker in explicit_goal_markers)


def _sanitize_capybara_anatomy(value: Any) -> Any:
    if isinstance(value, str):
        result = value.replace("尾巴", "前爪").replace("尾部", "身体后侧")
        result = re.sub(r"\btail\b", "front paw", result, flags=re.IGNORECASE)
        result = re.sub(
            r"\bcheek pouches?\b", "natural cheeks", result, flags=re.IGNORECASE
        )
        return result
    if isinstance(value, list):
        return [_sanitize_capybara_anatomy(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_capybara_anatomy(item) for key, item in value.items()}
    return value


def _requests_visual_effects(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in _EXPLICIT_EFFECT_MARKERS)


def _has_unsupported_effect(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in _UNSUPPORTED_EFFECT_MARKERS) or bool(
        re.search(
            r"(?:glow|light)\s+(?:suddenly\s+)?(?:pulses?|flashes?|flickers?)",
            lowered,
        )
    )


def _sanitize_unrequested_effects(value: Any) -> Any:
    """Remove common LLM-invented spectacle without adding negative prompt terms."""

    if isinstance(value, str):
        result = value
        replacements = (
            (
                r"(?i)\b(?:the\s+)?(?:glow|light)\s+(?:suddenly\s+)?"
                r"(?:pulses?|flashes?|flickers?|intensifies|brightens)\b",
                "the glow remains steady",
            ),
            (
                r"(?i)\b(?:emits?|releases?|sends? out|bursts? into)\s+"
                r"(?:a\s+|an\s+)?(?:faint\s+|soft\s+|bright\s+|sudden\s+)?"
                r"(?:beam|burst|wave|halo|aura)\s+of\s+(?:light|energy)\b",
                "keeps a steady glow",
            ),
            (
                r"(?i)\b(?:sparkling\s+|glowing\s+|magical\s+|shimmering\s+)?"
                r"(?:particles?|energy waves?|light bursts?|magical auras?|glowing trails?)\b",
                "",
            ),
            (
                r"(?:光芒|微光)(?:突然)?(?:变亮|增强|闪烁|脉动|爆发)",
                "微光保持稳定",
            ),
            (
                r"(?:突然)?(?:发出|迸发出|释放出)(?:一道|一阵|一圈)?"
                r"(?:微弱|柔和|耀眼|明亮)?(?:的)?(?:光芒|光束|光波|能量波)",
                "保持稳定微光",
            ),
            (r"(?:闪烁|脉动)的(?:光芒|微光)", "稳定的微光"),
            (r"(?:魔法)?(?:粒子|能量波|光爆|发光轨迹|魔法光环)", ""),
        )
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)
        result = re.sub(r"\s+([,.;:])", r"\1", result)
        result = re.sub(r"[ \t]{2,}", " ", result)
        result = re.sub(r"，{2,}", "，", result)
        return result.strip()
    if isinstance(value, list):
        return [_sanitize_unrequested_effects(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_unrequested_effects(item) for key, item in value.items()}
    return value


def _sanitize_story_h3_prompt(prompt: str, *, allow_visual_effects: bool) -> str:
    if allow_visual_effects:
        return prompt
    fields = _h3_prompt_values(prompt)
    fields = _sanitize_unrequested_effects(fields)
    return "\n\n".join(f"{field}: {fields[field]}" for field in PROMPT_FIELDS)


def _required_text(value: Any, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromptEnhancementError(f"{name} must be non-empty text")
    result = value.strip()
    if len(result) > maximum:
        raise PromptEnhancementError(f"{name} is too long")
    return result


def _text_list(value: Any, name: str, *, maximum: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > maximum:
        raise PromptEnhancementError(f"{name} must be a short array")
    result: list[str] = []
    for item in value:
        text = _required_text(item, name, maximum=360)
        if text.casefold() in {"none", "n/a", "unknown", "无", "无物品", "未知"}:
            continue
        if text.casefold() not in {existing.casefold() for existing in result}:
            result.append(text)
    return result


def _normalize_initial_story_state(
    *,
    story_bible: dict[str, Any],
    opening_summary: str,
) -> dict[str, Any]:
    return {
        "chapter": 1,
        "location": str(story_bible["world_setting"]),
        "current_goal": str(story_bible["core_goal"]),
        "inventory": list(story_bible.get("starting_inventory", [])),
        "relationships": list(story_bible.get("starting_relationships", [])),
        "established_facts": [opening_summary],
        "open_threads": [
            {
                "id": "thread-1",
                "question": str(story_bible["central_mystery"]),
                "urgency": "central story question",
            }
        ],
        "last_decision": opening_summary,
        "last_consequence": opening_summary,
        "character_condition": opening_summary,
        "emotional_progress": (
            f"{story_bible['personality']}；开始为目标行动：{story_bible['core_goal']}"
        ),
    }


def _optional_branch_text(
    branch: dict[str, Any], key: str, fallback: str, *, maximum: int = 480
) -> str:
    value = branch.get(key)
    if not isinstance(value, str) or not value.strip():
        return fallback
    return _required_text(value, f"story branch {key}", maximum=maximum)


def _normalize_state_delta(value: Any) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    result: dict[str, Any] = {}
    for field in (
        "location",
        "current_goal",
        "character_condition",
        "emotional_progress",
    ):
        candidate = value.get(field, "unchanged")
        result[field] = (
            _required_text(candidate, f"state delta {field}", maximum=360)
            if isinstance(candidate, str) and candidate.strip()
            else "unchanged"
        )
    for field in (
        "inventory_add",
        "inventory_remove",
        "relationships_add",
        "facts_add",
    ):
        result[field] = _text_list(
            value.get(field, []), f"state delta {field}", maximum=8
        )
    return result


def _apply_story_state(
    story_state: dict[str, Any],
    *,
    state_delta: dict[str, Any],
    decision: str,
    consequence: str,
    resolved_thread_id: str,
    new_thread: str,
    thread_urgency: str,
    branch_id: str,
) -> dict[str, Any]:
    current_chapter = story_state.get("chapter", 1)
    chapter = current_chapter + 1 if isinstance(current_chapter, int) else 2

    def updated_list(
        current_key: str, add_key: str, remove_key: str | None = None
    ) -> list[str]:
        current = [
            item
            for item in story_state.get(current_key, [])
            if isinstance(item, str) and item.strip()
        ]
        removals = (
            {item.casefold() for item in state_delta.get(remove_key, [])}
            if remove_key
            else set()
        )
        result = [item for item in current if item.casefold() not in removals]
        existing = {item.casefold() for item in result}
        for item in state_delta.get(add_key, []):
            if item.casefold() not in existing:
                result.append(item)
                existing.add(item.casefold())
        return result[-16:]

    open_threads = [
        dict(thread)
        for thread in story_state.get("open_threads", [])
        if isinstance(thread, dict)
        and isinstance(thread.get("id"), str)
        and isinstance(thread.get("question"), str)
    ]
    if resolved_thread_id.casefold() != "none":
        open_threads = [
            thread for thread in open_threads if thread["id"] != resolved_thread_id
        ]
    if new_thread.casefold() != "none":
        open_threads.append(
            {
                "id": f"thread-{chapter}-{branch_id}",
                "question": new_thread,
                "urgency": thread_urgency,
            }
        )

    def scalar(key: str) -> str:
        changed = state_delta[key]
        if changed.casefold() != "unchanged":
            return changed
        current = story_state.get(key)
        return current if isinstance(current, str) and current.strip() else "Unchanged."

    return {
        "chapter": chapter,
        "location": scalar("location"),
        "current_goal": scalar("current_goal"),
        "inventory": updated_list("inventory", "inventory_add", "inventory_remove"),
        "relationships": updated_list("relationships", "relationships_add"),
        "established_facts": updated_list("established_facts", "facts_add"),
        "open_threads": open_threads[-8:],
        "last_decision": decision,
        "last_consequence": consequence,
        "character_condition": scalar("character_condition"),
        "emotional_progress": scalar("emotional_progress"),
    }


def normalize_visual_observation(raw: str) -> dict[str, Any]:
    """Validate the grounded image-only observation passed to the writer."""

    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).replace("```", "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0:
        raise PromptEnhancementError("story observer did not return a JSON object")
    if end <= start:
        value = _recover_truncated_visual_observation(text[start:])
        if value is None:
            raise PromptEnhancementError("story observer did not return a JSON object")
    else:
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as error:
            value = _recover_truncated_visual_observation(text[start:])
            if value is None:
                raise PromptEnhancementError(
                    "story observer returned invalid JSON"
                ) from error
    if not isinstance(value, dict):
        raise PromptEnhancementError("story observer response must be a JSON object")
    motion = value.get("visible_motion")
    last_frame = value.get("last_frame_state")
    constraints = value.get("continuity_constraints")
    visual_state = value.get("visual_state")
    identity_observation = value.get("identity_observation", last_frame)
    identity_drift = value.get("identity_drift", "uncertain")
    if not isinstance(visual_state, str) or not visual_state.strip():
        raise PromptEnhancementError("story observer returned no display visual state")
    if not isinstance(motion, str) or not motion.strip():
        raise PromptEnhancementError("story observer returned no visible motion")
    if not isinstance(last_frame, str) or not last_frame.strip():
        raise PromptEnhancementError("story observer returned no last-frame state")
    if not isinstance(identity_observation, str) or not identity_observation.strip():
        raise PromptEnhancementError("story observer returned no identity observation")
    if not isinstance(identity_drift, str) or not identity_drift.strip():
        raise PromptEnhancementError("story observer returned invalid identity drift")
    if (
        not isinstance(constraints, list)
        or not 2 <= len(constraints) <= 6
        or not all(isinstance(item, str) and item.strip() for item in constraints)
    ):
        raise PromptEnhancementError("story observer returned invalid constraints")
    result = {
        "visual_state": visual_state.strip(),
        "visible_motion": motion.strip(),
        "last_frame_state": last_frame.strip(),
        "identity_observation": identity_observation.strip(),
        "identity_drift": identity_drift.strip(),
        "continuity_constraints": [item.strip() for item in constraints],
    }
    if len(json.dumps(result, ensure_ascii=False)) > 1600:
        raise PromptEnhancementError("story observer response is too long")
    return result


def _recover_truncated_visual_observation(text: str) -> dict[str, Any] | None:
    """Recover complete observer fields when generation stops inside its final array."""

    fields: dict[str, Any] = {}
    for key in (
        "visual_state",
        "visible_motion",
        "last_frame_state",
        "identity_observation",
        "identity_drift",
    ):
        match = re.search(rf'"{key}"\s*:\s*("(?:\\.|[^"\\])*")', text, flags=re.DOTALL)
        if match is None:
            return None
        try:
            fields[key] = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    array_match = re.search(r'"continuity_constraints"\s*:\s*\[', text)
    if array_match is None:
        return None
    constraint_text = text[array_match.end() :]
    constraints: list[str] = []
    for match in re.finditer(r'"(?:\\.|[^"\\])*"', constraint_text):
        try:
            constraints.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    if len(constraints) < 2:
        return None
    fields["continuity_constraints"] = constraints[:6]
    return fields


def _normalize_story_prompt(value: Any) -> str:
    if isinstance(value, dict):
        if list(value) != list(PROMPT_FIELDS) or not all(
            isinstance(value[field], str) and value[field].strip()
            for field in PROMPT_FIELDS
        ):
            raise PromptEnhancementError(
                "story branch prompt must contain the three H3 fields"
            )
        value = "\n\n".join(
            f"{field}: {value[field].strip()}" for field in PROMPT_FIELDS
        )
    if not isinstance(value, str) or not value.strip():
        raise PromptEnhancementError("story branch is missing label, hook, or prompt")
    prompt = normalize_h3_prompt(value)
    prefix = "integrated_multimodal_description:"
    integrated = prompt.removeprefix(prefix).lstrip()
    if not integrated.startswith("[Shot 1]"):
        prompt = prompt.replace(prefix, f"{prefix} [Shot 1]", 1)
    return prompt


def _h3_prompt_values(prompt: str) -> dict[str, str]:
    """Return normalized H3 fields for deterministic local inheritance checks."""

    normalized = normalize_h3_prompt(prompt)
    matches = list(
        re.finditer(
            r"(?im)^\s*(integrated_multimodal_description|overall_soundscape|non_diegetic_music)\s*:\s*",
            normalized,
        )
    )
    values: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        )
        values[match.group(1).lower()] = normalized[match.end() : end].strip()
    return values
