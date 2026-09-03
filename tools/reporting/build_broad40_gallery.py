#!/usr/bin/env python3
"""Build a local Broad40 video review page across public inference lanes."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from h3_flash.evals import EvaluationSuite


def _link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise ValueError(f"refusing to replace non-symlink media: {link}")
    link.symlink_to(target.resolve(strict=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--lossless-root", type=Path, required=True)
    parser.add_argument("--flash-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    suite = EvaluationSuite.load(args.suite)
    cases = suite.cases
    destination = args.destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    media_root = destination.parent / "media"

    official_records = {}
    for case in cases:
        case_id = case.case_id
        case_root = args.official_root / "cases" / case_id
        result = json.loads((case_root / "result.json").read_text(encoding="utf-8"))
        source = case_root / result["outputs"]["path"]
        link = media_root / "official" / f"{case_id}.mp4"
        _link(source, link)
        official_records[case_id] = {
            "url": link.relative_to(destination.parent).as_posix(),
            "e2e": result["timing_seconds"]["request_process"],
        }

    lane_records = {}
    for lane, root in (("LOSSLESS", args.lossless_root), ("FLASH", args.flash_root)):
        records = {}
        for case in cases:
            case_id = case.case_id
            case_root = root / "cases" / case_id
            result = json.loads((case_root / "result.json").read_text(encoding="utf-8"))
            source = case_root / result["outputs"]["path"]
            link = media_root / lane.lower() / f"{case_id}.mp4"
            _link(source, link)
            gpu_count = len(
                [value for value in result.get("peak_gpu_memory_bytes", []) if value]
            )
            records[case_id] = {
                "url": link.relative_to(destination.parent).as_posix(),
                "e2e": result["timing_seconds"]["request_process"],
                "label": f"{lane} · {gpu_count or 1}×B200",
            }
        lane_records[lane] = records

    cards = []
    categories = sorted(
        {str(case.metadata.get("category", "uncategorized")) for case in cases}
    )
    for case in cases:
        case_id = case.case_id
        official = official_records[case_id]
        lossless = lane_records["LOSSLESS"][case_id]
        flash = lane_records["FLASH"][case_id]
        category = str(case.metadata.get("category", "uncategorized"))
        axes = " · ".join(case.metadata.get("stress_axes", ()))
        cards.append(
            f'''<article class="card" data-category="{html.escape(category)}">
              <header><h2>{html.escape(case_id)}</h2><span>{html.escape(category)}</span></header>
              <div class="videos">
                <section><h3>OFFICIAL · 1×B200 <b>{official["e2e"]:.3f}s</b></h3><video controls preload="metadata" playsinline data-src="{html.escape(official["url"])}"></video></section>
                <section><h3>{html.escape(lossless["label"])} <b>{lossless["e2e"]:.3f}s</b></h3><video controls preload="metadata" playsinline data-src="{html.escape(lossless["url"])}"></video></section>
                <section><h3>{html.escape(flash["label"])} <b>{flash["e2e"]:.3f}s</b></h3><video controls preload="metadata" playsinline data-src="{html.escape(flash["url"])}"></video></section>
              </div>
              <p class="axes">{html.escape(axes)}</p>
              <details><summary>查看 prompt</summary><pre>{html.escape(case.prompt)}</pre></details>
            </article>'''
        )

    category_options = "".join(
        f'<option value="{html.escape(category)}">{html.escape(category)}</option>'
        for category in categories
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3-Flash · {html.escape(suite.suite_id)} Review</title><style>
:root{{color-scheme:dark;--bg:#080b10;--panel:#111722;--line:#283548;--text:#edf4ff;--muted:#9aabc0;--accent:#69e0b4}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 70% -10%,#183b4d 0,transparent 34rem),var(--bg);color:var(--text);font:15px/1.5 Inter,system-ui,sans-serif}}
.hero{{max-width:1600px;margin:auto;padding:52px 26px 24px}} h1{{font-size:clamp(38px,6vw,76px);letter-spacing:-.05em;line-height:.95;margin:8px 0 20px}} .hero p,.axes{{color:var(--muted)}}
.controls{{display:flex;gap:12px;flex-wrap:wrap}} input,select{{background:var(--panel);border:1px solid var(--line);border-radius:10px;color:var(--text);padding:10px 12px;font:inherit}}
main{{max-width:1600px;margin:auto;padding:18px 26px 80px;display:grid;gap:24px}} .card{{border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,#141c28,#0d131c);padding:18px}}
.card>header{{display:flex;gap:12px;align-items:center;margin-bottom:13px}} h2{{font-size:18px;margin:0}} .card>header span{{margin-left:auto;color:var(--accent);font-size:12px}}
.videos{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}} .videos section{{min-width:0}} h3{{font-size:12px;letter-spacing:.05em;color:var(--muted);margin:0 0 7px}} h3 b{{float:right;color:var(--accent);font-size:14px}}
video{{width:100%;aspect-ratio:1344/768;background:#000;border-radius:10px;display:block}} .pending div{{aspect-ratio:1344/768;border:1px dashed var(--line);border-radius:10px;display:grid;place-items:center;color:var(--muted)}}
.axes{{font-size:12px;margin:12px 0}} details{{border-top:1px solid var(--line);padding-top:10px}} summary{{cursor:pointer;color:var(--muted)}} pre{{white-space:pre-wrap;font:12px/1.5 ui-monospace,monospace;color:#c0ccdc}}
@media(max-width:850px){{.videos{{grid-template-columns:1fr}}.hero{{padding-top:30px}} main{{padding:12px}}}}
</style></head><body><div class="hero"><div style="color:var(--accent);font-weight:700">FluxMatrix · H3-Flash</div><h1>三模式视频对照</h1>
<p>OFFICIAL：BF16、官方 50-point sigma grid / 49 次 DiT、完整 dense attention，每请求 1×B200。LOSSLESS：权重、步数、有效 attention 图和官方 Video/Audio VAE 不变；使用 SP8、kernel fusion、packed collective、VAE tile parallel/compile 和完整音频 FFmpeg 输出，每请求 8×B200。FLASH：只在 LOSSLESS 之上加入 LightX2V Turbo4 权重与四次 Transformer 求值；Qwen 仍为官方 BF16，attention 仍为 dense PyTorch SDPA。三个版本的资源和质量合同不同，耗时不可脱离标签比较。</p>
<p><b>{html.escape(suite.suite_id)}：</b>{len(cases)} 个相同 prompt / seed 的 OFFICIAL、LOSSLESS 与 FLASH 输出。</p><div class="controls"><input id="search" placeholder="搜索 case / prompt" size="28"><select id="category"><option value="all">全部分类</option>{category_options}</select></div></div>
<main>{"".join(cards)}</main><script>
const loadVideo=v=>{{if(!v.src&&v.dataset.src)v.src=v.dataset.src}};const observer=new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting){{loadVideo(e.target);observer.unobserve(e.target)}}}}),{{rootMargin:'700px'}});document.querySelectorAll('video').forEach(v=>observer.observe(v));
const search=document.querySelector('#search'),category=document.querySelector('#category'); function filter(){{const q=search.value.toLowerCase();for(const card of document.querySelectorAll('.card')){{card.hidden=(category.value!=='all'&&card.dataset.category!==category.value)||!card.textContent.toLowerCase().includes(q)}}}} search.oninput=filter;category.onchange=filter;
</script></body></html>"""
    destination.write_text(page + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
