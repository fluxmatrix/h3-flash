import json

import pytest

from h3_flash.prompting import (
    PromptEnhancementError,
    build_h3_prompt_messages,
    build_story_bible_messages,
    build_story_opening_messages,
    build_story_observer_messages,
    build_story_messages,
    build_story_prompt_messages,
    make_i2va_prompt,
    normalize_h3_prompt,
    normalize_story_bible,
    normalize_story_opening,
    normalize_story_outline,
    normalize_story_plan,
    normalize_story_prompts,
    normalize_visual_observation,
    story_outline_violations,
)


def test_build_h3_prompt_messages_preserves_brief_and_duration() -> None:
    messages = build_h3_prompt_messages("  A fox runs.  ", 10)
    assert messages[-1] == {"role": "user", "content": "A fox runs."}
    assert "10-second video" in messages[0]["content"]


def test_make_i2va_prompt_adds_official_alignment_once() -> None:
    core = (
        "integrated_multimodal_description: [Shot 1] A fox runs.\n\n"
        "overall_soundscape: Pawsteps.\n\n"
        "non_diegetic_music: N/A"
    )
    prompt = make_i2va_prompt(core)
    assert prompt.startswith("For the target video, at 0.00 seconds")
    assert "<Picture 1> establishes the opening composition and subjects" in prompt
    assert make_i2va_prompt(prompt) == prompt


def test_build_story_messages_keeps_premise_history_and_current_scene() -> None:
    messages = build_story_messages(
        premise="A courier must deliver a mysterious key.",
        current_prompt="integrated_multimodal_description: [Shot 1] The courier runs.",
        history=["Enter the station"],
        duration_seconds=10,
        display_language="Simplified Chinese",
        visual_observation={
            "visual_state": "信使正跑向岔路。",
            "visible_motion": "The courier runs toward a fork.",
            "last_frame_state": "The courier reaches the fork.",
            "continuity_constraints": ["One courier", "Two passages"],
        },
    )
    assert "10-second" in messages[0]["content"]
    assert "in Simplified Chinese" in messages[0]["content"]
    context = json.loads(messages[-1]["content"])
    assert context["story_context"]["premise"].startswith("A courier")
    assert context["story_context"]["recent_choice_summaries"] == ["Enter the station"]
    assert "immutable_story_bible" in context["story_context"]
    assert "current_story_state" in context["story_context"]


def test_build_and_normalize_story_opening() -> None:
    bible_messages = build_story_bible_messages(
        "A tiny capybara steals a sugar crystal.",
        display_language="Simplified Chinese",
    )
    assert bible_messages[-1]["content"].startswith("A tiny capybara")
    bible = {
        "protagonist_identity": "拇指大小的水豚盗宝者",
        "visual_anchor": "棕色毛发、小皮包、圆耳朵",
        "personality": "谨慎但贪吃",
        "decision_style": "利用厨房物件制造声东击西",
        "core_goal": "取回糖晶",
        "weakness": "无法抵抗甜食",
        "starting_inventory": [],
        "starting_relationships": [],
        "non_negotiables": ["不伤害厨师"],
        "world_setting": "午夜面包店",
        "genre_tone": "微缩盗宝喜剧",
        "visual_style": "电影级微距实拍",
        "world_rules": ["普通甜点没有魔法"],
        "central_mystery": "糖晶为何被藏进蛋糕",
        "opposing_force": "即将返回的面包师",
        "stakes": "面包店关门后会永久封存蛋糕",
        "emotional_thread": "水豚学会在达成目标时照顾身边的小生命",
        "ending_direction": "带着糖晶离开并发现真相",
    }
    messages = build_story_opening_messages(
        "A tiny capybara steals a sugar crystal.",
        bible,
        duration_seconds=10,
        display_language="Simplified Chinese",
    )
    context = json.loads(messages[-1]["content"])
    assert context["premise"].startswith("A tiny capybara")
    assert context["immutable_story_bible"] == bible
    assert normalize_story_bible(json.dumps({"story_bible": bible})) == bible
    raw = json.dumps(
        {
            "title": "午夜糖晶",
            "opening_summary": "水豚爬上蛋糕，面包师的影子突然出现。",
            "story_arc": [
                {
                    "chapter": index,
                    "purpose": f"beat {index}",
                    "story_change": f"change {index}",
                    "milestone": f"step {index}",
                }
                for index in range(1, 6)
            ],
            "initial_state": {
                "location": "婚礼蛋糕顶部",
                "current_goal": "拿到糖晶",
                "inventory": ["小皮包"],
                "relationships": [],
                "established_facts": ["面包师即将返回"],
                "open_threads": [{"question": "影子是谁", "urgency": "立即"}],
                "last_decision": "爬上蛋糕",
                "last_consequence": "面包师的影子出现",
                "character_condition": "趴在糖霜边缘",
                "emotional_progress": "水豚仍只关心糖晶",
            },
        },
        ensure_ascii=False,
    )
    title, summary, arc, state = normalize_story_opening(raw, story_bible=bible)
    assert title == "午夜糖晶"
    assert summary.startswith("水豚爬上")
    assert len(arc) == 5
    assert state["chapter"] == 1
    assert state["open_threads"][0]["id"] == "thread-1"


def test_normalize_story_bible_fills_safe_defaults_for_empty_constraints() -> None:
    bible = {
        "protagonist_identity": "一只水豚",
        "visual_anchor": "一只水豚",
        "personality": "善良",
        "decision_style": "先观察再行动",
        "core_goal": "把苹果送到河对岸",
        "weakness": "不会游泳",
        "starting_inventory": ["苹果"],
        "starting_relationships": [],
        "non_negotiables": [],
        "world_setting": "河边",
        "genre_tone": "温馨冒险",
        "visual_style": "自然实拍",
        "world_rules": [],
        "central_mystery": "怎样安全过河",
        "opposing_force": "河水",
        "stakes": "送达会被延迟",
        "emotional_thread": "学会接受帮助",
        "ending_direction": "把苹果送达",
    }
    normalized = normalize_story_bible(json.dumps({"story_bible": bible}))
    assert normalized["non_negotiables"] == ["保持主角身份与核心目标不变"]
    assert normalized["world_rules"] == ["遵循普通物理规律"]


def test_story_normalizers_remove_unrequested_effects_and_capybara_tail() -> None:
    bible = {
        "protagonist_identity": "一只水豚",
        "visual_anchor": "一只没有可见尾巴的水豚",
        "personality": "谨慎而善良",
        "decision_style": "先观察再行动",
        "core_goal": "取回糖晶",
        "weakness": "体型很小",
        "starting_inventory": [],
        "starting_relationships": [],
        "non_negotiables": ["糖晶只会稳定发光"],
        "world_setting": "面包店",
        "genre_tone": "温馨冒险",
        "visual_style": "自然实拍",
        "world_rules": ["遵循普通物理规律"],
        "central_mystery": "如何够到糖晶",
        "opposing_force": "蛋糕太高",
        "stakes": "取回会被延迟",
        "emotional_thread": "学会接受帮助",
        "ending_direction": "取回糖晶",
    }
    raw_opening = json.dumps(
        {
            "title": "糖晶",
            "opening_summary": "水豚碰到糖晶后，光芒突然变亮。",
            "story_arc": [
                {
                    "chapter": index,
                    "purpose": "推进故事",
                    "story_change": "水豚认识了一位新伙伴。",
                    "milestone": "水豚用尾巴碰触糖晶，糖晶发出微弱光芒。",
                }
                for index in range(1, 6)
            ],
        },
        ensure_ascii=False,
    )
    _, opening_summary, story_arc, _ = normalize_story_opening(
        raw_opening, story_bible=bible
    )
    serialized = json.dumps(
        {"opening_summary": opening_summary, "story_arc": story_arc},
        ensure_ascii=False,
    )
    assert "尾巴" not in serialized
    assert "发出微弱光芒" not in serialized
    assert "光芒突然变亮" not in serialized
    assert "保持稳定" in serialized

    raw_prompts = json.dumps(
        {
            branch_id: {
                "integrated_multimodal_description": (
                    "[Shot 1] The crystal glow suddenly pulses and releases "
                    "sparkling particles while the capybara waits."
                ),
                "overall_soundscape": "Quiet bakery ambience.",
                "non_diegetic_music": "N/A",
            }
            for branch_id in ("a", "b")
        }
    )
    prompts = normalize_story_prompts(raw_prompts)
    assert all("pulses" not in prompt for prompt in prompts.values())
    assert all("particles" not in prompt for prompt in prompts.values())
    assert all("glow remains steady" in prompt for prompt in prompts.values())


def test_build_and_normalize_story_observer_messages() -> None:
    data_urls = tuple(f"data:image/jpeg;base64,frame{index}" for index in range(4))
    messages = build_story_observer_messages(data_urls)
    assert sum(item["type"] == "image_url" for item in messages[-1]["content"]) == 4
    observation = normalize_visual_observation(
        json.dumps(
            {
                "visual_state": "信使转身面向无人机。",
                "visible_motion": "The courier turns toward the drone.",
                "last_frame_state": "The courier faces away while holding the key.",
                "continuity_constraints": ["One courier", "Drone above right"],
            }
        )
    )
    assert observation["last_frame_state"].startswith("The courier")
    assert "safe_first_actions" not in observation


def test_normalize_story_observer_recovers_a_truncated_constraints_array() -> None:
    raw = """{
      "visual_state": "女孩与老者交换位置。",
      "visible_motion": "女孩向右走，老者向左走。",
      "last_frame_state": "女孩站在右侧，老者站在左侧。",
      "identity_observation": "女孩穿蓝色长袍，老者戴黑帽。",
      "identity_drift": "none",
      "continuity_constraints": [
        "女孩保持蓝色长袍",
        "老者保持黑色帽子",
        "竹篮保持完整",
        "未完成的"""
    observation = normalize_visual_observation(raw)
    assert observation["last_frame_state"].startswith("女孩站在右侧")
    assert observation["continuity_constraints"] == [
        "女孩保持蓝色长袍",
        "老者保持黑色帽子",
        "竹篮保持完整",
    ]


def test_normalize_story_plan_returns_two_tail_anchored_h3_prompts() -> None:
    prompt_a = (
        "integrated_multimodal_description: [Shot 1] The courier enters the train.\n"
        "overall_soundscape: Shoes and doors.\n"
        "non_diegetic_music: N/A"
    )
    prompt_b = (
        "integrated_multimodal_description: [Shot 1] The courier follows a light.\n"
        "overall_soundscape: Shoes and a soft hum.\n"
        "non_diegetic_music: N/A"
    )
    raw = json.dumps(
        {
            "scene_summary": "The courier reaches a fork.",
            "branches": [
                {
                    "label": "Board the train",
                    "hook": "The doors close.",
                    "story_memory": "The courier boards before the doors seal.",
                    "prompt": prompt_a,
                },
                {
                    "label": "Follow the light",
                    "hook": "A signal appears.",
                    "story_memory": "The courier follows a signal into the tunnel.",
                    "prompt": prompt_b,
                },
            ],
        }
    )
    summary, branches = normalize_story_plan(raw)
    assert summary == "The courier reaches a fork."
    assert [branch["id"] for branch in branches] == ["a", "b"]
    assert branches[0]["prompt"].startswith("For the target video, at 0.00 seconds")
    assert "<Picture 1> establishes" in branches[1]["prompt"]


def test_normalize_story_plan_accepts_json_native_prompt_fields() -> None:
    prompt = {
        "integrated_multimodal_description": "The courier enters the train.",
        "overall_soundscape": "Shoes and doors.",
        "non_diegetic_music": "N/A",
    }
    raw = json.dumps(
        {
            "scene_summary": "The courier reaches a fork.",
            "branches": [
                {
                    "label": "Board train",
                    "hook": "Doors close.",
                    "story_memory": "The courier boards before the doors close.",
                    "prompt": prompt,
                },
                {
                    "label": "Stay outside",
                    "hook": "A light moves.",
                    "story_memory": "The courier remains outside and sees a light.",
                    "prompt": prompt,
                },
            ],
        }
    )
    _, branches = normalize_story_plan(raw)
    assert "integrated_multimodal_description: [Shot 1]" in branches[0]["prompt"]


def test_normalize_story_plan_inherits_music_in_code() -> None:
    prompt = {
        "integrated_multimodal_description": "[Shot 1] The courier runs.",
        "overall_soundscape": "Shoes on concrete.",
        "non_diegetic_music": "Wrong generated music",
    }
    raw = json.dumps(
        {
            "scene_summary": "The courier keeps moving.",
            "branches": [
                {
                    "label": "Run ahead",
                    "hook": "A path opens.",
                    "story_memory": "The courier runs into the newly opened path.",
                    "prompt": prompt,
                },
                {
                    "label": "Look behind",
                    "hook": "A light appears.",
                    "story_memory": "The courier turns and discovers a signal.",
                    "prompt": prompt,
                },
            ],
        }
    )
    _, branches = normalize_story_plan(raw, non_diegetic_music="N/A")
    assert all("non_diegetic_music: N/A" in branch["prompt"] for branch in branches)
    assert all("Wrong generated music" not in branch["prompt"] for branch in branches)


def test_normalize_story_plan_applies_only_each_branch_state_delta() -> None:
    prompt = {
        "integrated_multimodal_description": "[Shot 1] The courier runs.",
        "overall_soundscape": "Shoes on concrete.",
        "non_diegetic_music": "N/A",
    }
    state = {
        "chapter": 1,
        "location": "Station fork",
        "current_goal": "Reach the exit",
        "inventory": ["glass key", "jacket"],
        "relationships": [],
        "established_facts": ["A drone follows her"],
        "open_threads": [
            {"id": "thread-1", "question": "Who shut the gate", "urgency": "now"}
        ],
        "last_decision": "Enter the station",
        "last_consequence": "The gate closes",
        "character_condition": "Running",
        "emotional_progress": "The courier still works alone",
    }
    branch = {
        "label": "Throw the jacket",
        "hook": "The drone follows it.",
        "decision": "The courier throws her jacket into the right tunnel.",
        "visible_consequence": "The drone turns away and drops a map.",
        "cost": "She loses her jacket.",
        "resolved_thread_id": "thread-1",
        "new_thread": "Why is the exit erased from the map",
        "thread_urgency": "now",
        "state_delta": {
            "location": "Left tunnel",
            "current_goal": "Read the map",
            "inventory_add": ["exit map"],
            "inventory_remove": ["jacket"],
            "relationships_add": [],
            "facts_add": ["The drone tracks her jacket"],
            "character_condition": "Hidden while holding the key",
        },
        "prompt": prompt,
    }
    raw = json.dumps(
        {
            "scene_summary": "The courier reaches the fork.",
            "branches": [branch, {**branch, "label": "Climb the gate"}],
        }
    )
    _, branches = normalize_story_plan(raw, story_state=state)
    next_state = branches[0]["next_state"]
    assert next_state["chapter"] == 2
    assert next_state["inventory"] == ["glass key", "exit map"]
    assert next_state["location"] == "Left tunnel"
    assert next_state["open_threads"] == [
        {
            "id": "thread-2-a",
            "question": "Why is the exit erased from the map",
            "urgency": "now",
        }
    ]
    assert state["inventory"] == ["glass key", "jacket"]


def test_normalize_story_plan_disambiguates_duplicate_labels() -> None:
    prompt = {
        "integrated_multimodal_description": "[Shot 1] The courier runs.",
        "overall_soundscape": "Shoes on concrete.",
        "non_diegetic_music": "N/A",
    }
    raw = json.dumps(
        {
            "scene_summary": "The courier reaches the tunnel.",
            "branches": [
                {"label": "Open hatch", "hook": "Smoke emerges.", "prompt": prompt},
                {"label": "Open hatch", "hook": "A ladder drops.", "prompt": prompt},
            ],
        }
    )
    _, branches = normalize_story_plan(raw)
    assert [branch["label"] for branch in branches] == [
        "A · Open hatch",
        "B · Open hatch",
    ]
    assert branches[0]["story_memory"].startswith("Open hatch Smoke emerges.")


def test_story_quality_gate_allows_opposite_choices_but_rejects_a_dead_end() -> None:
    raw = json.dumps(
        {
            "scene_summary": "A capybara meets a mouse.",
            "shared_story_event": "The capybara finds a mouse tangled in ribbon.",
            "dilemma": "Help the mouse or continue alone and possibly fail.",
            "branches": [
                {
                    "label": "Help first",
                    "hook": "The mouse points out a bridge.",
                    "decision": "The capybara frees the mouse.",
                    "narrative_turn": "They establish trust.",
                    "visible_consequence": "The mouse reveals a bridge.",
                    "cost": "A short delay.",
                    "state_delta": {"relationships_add": ["mouse"]},
                },
                {
                    "label": "Go alone",
                    "hook": "The capybara gets stuck.",
                    "decision": "The capybara ignores the mouse and goes alone.",
                    "narrative_turn": "Nothing changes.",
                    "visible_consequence": "The attempt fails.",
                    "cost": "It misses the chance to build trust.",
                    "state_delta": {"relationships_add": []},
                },
            ],
        }
    )
    _, branches = normalize_story_outline(raw)
    violations = story_outline_violations(
        branches,
        target_milestone={"story_change": "The capybara establishes trust."},
        story_bible={"protagonist_identity": "A capybara"},
    )
    assert any("failed attempt" in issue for issue in violations)
    assert any("Branch b" in issue for issue in violations)


def test_dog_encounter_keeps_opposite_leave_branch_safe_and_story_bearing() -> None:
    raw = json.dumps(
        {
            "scene_summary": "女孩与小狗对视。",
            "shared_story_event": "小狗靠近女孩。",
            "dilemma": "摸摸小狗，还是转身离开？",
            "branches": [
                {
                    "label": "摸摸小狗",
                    "hook": "小狗突然跑向马路。",
                    "story_turn": "女孩摸摸小狗，小狗跑向车流。",
                    "cost": "女孩被吓到。",
                    "relationship_change": "小狗开始信任女孩",
                    "fact_learned": "小狗戴着红色项圈",
                    "next_goal": "追上小狗",
                    "inventory_add": [],
                    "inventory_remove": [],
                    "new_question": "小狗要去哪里？",
                },
                {
                    "label": "转身离开",
                    "hook": "小狗冲向马路，汽车紧急刹车。",
                    "story_turn": "女孩离开，小狗冲向马路。",
                    "cost": "女孩感到内疚。",
                    "relationship_change": "none",
                    "fact_learned": "小狗想追她",
                    "next_goal": "回去救小狗",
                    "inventory_add": [],
                    "inventory_remove": [],
                    "new_question": "小狗是否安全？",
                },
            ],
        },
        ensure_ascii=False,
    )
    _, branches = normalize_story_outline(
        raw,
        story_bible={
            "protagonist_identity": "女孩",
            "central_mystery": "小狗为何停在马路上",
        },
    )
    assert [branch["label"] for branch in branches] == ["伸手摸摸小狗", "转身离开"]
    assert "马路" not in branches[1]["hook"]
    assert branches[0]["next_goal"] != branches[1]["next_goal"]
    assert all(
        "红色小玩具" in branch["new_thread"] or branch["id"] == "a"
        for branch in branches
    )


def test_separate_story_prompt_writer_preserves_approved_story_and_music() -> None:
    prompt = (
        "integrated_multimodal_description: [Shot 1] A capybara pauses.\n\n"
        "overall_soundscape: Quiet bakery ambience.\n\n"
        "non_diegetic_music: N/A"
    )
    raw_outline = json.dumps(
        {
            "scene_summary": "A capybara meets a mouse.",
            "shared_story_event": "The mouse offers a ribbon.",
            "dilemma": "Help first or make a reciprocal plan.",
            "branches": [
                {
                    "label": "Help first",
                    "hook": "Trust reveals a route.",
                    "decision": "The capybara frees the mouse.",
                    "narrative_turn": "They establish trust.",
                    "visible_consequence": "The mouse points out a route.",
                    "cost": "A short delay.",
                },
                {
                    "label": "Work together",
                    "hook": "The ribbon becomes a shared route.",
                    "decision": "They pull the ribbon together.",
                    "narrative_turn": "They establish a partnership.",
                    "visible_consequence": "The ribbon spans the gap.",
                    "cost": "They give up the closest ledge.",
                },
            ],
        }
    )
    _, branches = normalize_story_outline(raw_outline)
    messages = build_story_prompt_messages(
        premise="A capybara retrieves a crystal.",
        current_prompt=prompt,
        visual_observation={"last_frame_state": "The capybara pauses."},
        story_bible={
            "protagonist_identity": "A capybara",
            "visual_anchor": "A capybara with brown fur",
            "visual_style": "Natural live action",
            "canonical_h3_prompt": prompt,
        },
        branches=branches,
        duration_seconds=10,
    )
    context = json.loads(messages[-1]["content"])
    assert context["approved_branches"][0]["shared_story_event"].startswith("The mouse")
    assert context["protagonist_identity"] == "A capybara"
    assert "canonical_h3_prompt" not in context
    raw_prompts = json.dumps(
        {
            branch_id: {
                "integrated_multimodal_description": "[Shot 1] The story continues.",
                "overall_soundscape": "Pawsteps.",
                "non_diegetic_music": "Wrong music",
            }
            for branch_id in ("a", "b")
        }
    )
    compiled = normalize_story_prompts(raw_prompts, non_diegetic_music="N/A")
    assert all("non_diegetic_music: N/A" in item for item in compiled.values())


def test_normalize_h3_prompt_strips_reasoning_and_markdown() -> None:
    raw = """<think>private analysis</think>
```text
integrated_multimodal_description: A fox runs through snow.
overall_soundscape: Pawsteps and wind.
non_diegetic_music: N/A
```<|im_end|>"""
    assert normalize_h3_prompt(raw) == (
        "integrated_multimodal_description: A fox runs through snow.\n\n"
        "overall_soundscape: Pawsteps and wind.\n\n"
        "non_diegetic_music: N/A"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "A plain paragraph.",
        "overall_soundscape: Wind.\nnon_diegetic_music: N/A",
        "integrated_multimodal_description: A fox.\n"
        "overall_soundscape:\nnon_diegetic_music: N/A",
    ],
)
def test_normalize_h3_prompt_rejects_incomplete_outputs(raw: str) -> None:
    with pytest.raises(PromptEnhancementError):
        normalize_h3_prompt(raw)
