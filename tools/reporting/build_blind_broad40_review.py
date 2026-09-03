#!/usr/bin/env python3
"""Build a static, browser-local blind official-vs-candidate review page."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from h3_flash.evals import EvaluationSuite

AXES = (
    ("semantic", "语义 / Prompt 遵循"),
    ("visual", "画面质量"),
    ("motion", "运动与时序一致性"),
    ("audio", "声音质量与内容"),
    ("sync", "音画同步"),
)


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"refusing to replace non-symlink {destination}")
    destination.symlink_to(source.resolve(strict=True))


def _record(root: Path, case_id: str) -> tuple[Path, float]:
    case_root = root / "cases" / case_id
    result = json.loads((case_root / "result.json").read_text(encoding="utf-8"))
    return case_root / result["outputs"]["path"], float(
        result["timing_seconds"]["request_process"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    suite = EvaluationSuite.load(args.suite)
    destination = args.destination.resolve()
    media = destination.parent / f"{destination.stem}-media"
    cards = []
    answer_key = {}
    for case in suite.cases:
        case_id = case.case_id
        try:
            official_path, official_time = _record(args.official_root, case_id)
            candidate_path, candidate_time = _record(args.candidate_root, case_id)
        except FileNotFoundError:
            continue
        official_link = media / "official" / f"{case_id}.mp4"
        candidate_link = media / "candidate" / f"{case_id}.mp4"
        _link(official_path, official_link)
        _link(candidate_path, candidate_link)
        official_left = hashlib.sha256(case_id.encode()).digest()[0] % 2 == 0
        options = [
            ("official", official_link, official_time),
            ("candidate", candidate_link, candidate_time),
        ]
        if not official_left:
            options.reverse()
        answer_key[case_id] = {"A": options[0][0], "B": options[1][0]}
        rows = "".join(
            f'''<label>{html.escape(label)}<select data-axis="{axis}">
              <option value="">未评分</option><option value="A">A 更好</option>
              <option value="tie">相当</option><option value="B">B 更好</option>
            </select></label>'''
            for axis, label in AXES
        )
        urls = [
            path.relative_to(destination.parent).as_posix() for _, path, _ in options
        ]
        timings = [value for _, _, value in options]
        cards.append(
            f'''<article class="card" data-case="{html.escape(case_id)}">
            <header><h2>{html.escape(case_id)}</h2><span>{html.escape(str(case.metadata.get("category", "uncategorized")))}</span></header>
            <div class="videos"><section><h3>A <b class="identity">{options[0][0]}</b><i>{timings[0]:.3f}s</i></h3><video controls preload="metadata" src="{html.escape(urls[0])}"></video></section>
            <section><h3>B <b class="identity">{options[1][0]}</b><i>{timings[1]:.3f}s</i></h3><video controls preload="metadata" src="{html.escape(urls[1])}"></video></section></div>
            <details><summary>Prompt</summary><pre>{html.escape(case.prompt)}</pre></details>
            <div class="scores">{rows}</div><textarea placeholder="可选备注"></textarea></article>'''
        )
    answer_json = json.dumps(answer_key, sort_keys=True).replace("<", "\\u003c")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3-Flash {html.escape(suite.suite_id)} Blind Review</title><style>
:root{{color-scheme:dark;--bg:#080b10;--panel:#111722;--line:#283548;--text:#edf4ff;--muted:#9aabc0;--accent:#69e0b4}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}}.hero,main{{max-width:1500px;margin:auto;padding:24px}}h1{{font-size:40px;margin:0 0 10px}}p,summary{{color:var(--muted)}}button{{padding:9px 13px;margin:4px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--text)}}.card{{margin:20px 0;padding:18px;border:1px solid var(--line);border-radius:16px;background:var(--panel)}}header{{display:flex;align-items:center}}h2{{font-size:17px}}header span{{margin-left:auto;color:var(--accent)}}.videos{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}h3{{margin:0 0 6px}}h3 i{{float:right;color:var(--muted);font-style:normal}}video{{width:100%;aspect-ratio:1344/768;background:#000;border-radius:9px}}.identity{{display:none;color:var(--accent)}}body.reveal .identity{{display:inline}}.scores{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:12px}}label{{display:grid;gap:4px;color:var(--muted)}}select,textarea{{background:#0a0f16;border:1px solid var(--line);border-radius:7px;color:var(--text);padding:7px}}textarea{{width:100%;margin-top:9px}}pre{{white-space:pre-wrap}}@media(max-width:850px){{.videos,.scores{{grid-template-columns:1fr}}}}
</style></head><body><div class="hero"><h1>{html.escape(suite.suite_id)} 盲评</h1><p>共 {len(cards)} 条已就绪。A/B 顺序按 case 固定随机；先独立评语义、画面、运动、声音、音画同步，再揭晓身份。显示的耗时不参与质量评分。</p><button id="reveal">揭晓 / 隐藏身份</button><button id="export">导出评分 JSON</button><span id="progress"></span></div><main>{"".join(cards)}</main>
<script id="answer" type="application/json">{answer_json}</script><script>
const key='h3-flash-broad40-blind-v1',cards=[...document.querySelectorAll('.card')];let saved=JSON.parse(localStorage.getItem(key)||'{{}}');
function persist(){{for(const c of cards){{let r={{axes:{{}},note:c.querySelector('textarea').value}};for(const s of c.querySelectorAll('select'))r.axes[s.dataset.axis]=s.value;saved[c.dataset.case]=r}}localStorage.setItem(key,JSON.stringify(saved));progress()}}
function progress(){{let n=0,total=cards.length*5;for(const r of Object.values(saved))for(const v of Object.values(r.axes||{{}}))if(v)n++;document.querySelector('#progress').textContent=` ${{n}}/${{total}} 项已评分`}}
for(const c of cards){{let r=saved[c.dataset.case]||{{axes:{{}},note:''}};for(const s of c.querySelectorAll('select'))s.value=r.axes[s.dataset.axis]||'';c.querySelector('textarea').value=r.note||'';c.onchange=persist;c.oninput=persist}}
document.querySelector('#reveal').onclick=()=>document.body.classList.toggle('reveal');document.querySelector('#export').onclick=()=>{{persist();let out={{schema_version:1,answers:JSON.parse(document.querySelector('#answer').textContent),ratings:saved}};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}));a.download='h3-flash-broad40-review.json';a.click()}};progress();
</script></body></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    print(json.dumps({"cases": len(cards), "page": str(destination)}))


if __name__ == "__main__":
    main()
