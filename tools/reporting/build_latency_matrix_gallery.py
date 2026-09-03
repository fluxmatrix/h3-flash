#!/usr/bin/env python3
"""Build a static review page for all 144 latency-matrix artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from h3_flash.evals import EvaluationSuite
from tools.reporting.summarize_latency_matrix import (
    DURATIONS,
    ORIENTATIONS,
    RESOLUTIONS,
    VERSIONS,
    _suite_name,
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


def _result(root: Path, suite_name: str, case_id: str) -> tuple[dict, Path]:
    case_root = root / suite_name / "cases" / case_id
    result = json.loads((case_root / "result.json").read_text(encoding="utf-8"))
    return result, case_root / result["outputs"]["path"]


def main() -> None:
    parser = argparse.ArgumentParser()
    for version in VERSIONS:
        parser.add_argument(f"--{version.lower()}-root", type=Path, required=True)
    parser.add_argument(
        "--suites-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/evals",
    )
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    roots = {version: getattr(args, f"{version.lower()}_root") for version in VERSIONS}
    destination = args.destination.resolve()
    media_root = destination.parent / f"{destination.stem}-media"
    sections = []
    artifact_count = 0
    for orientation in ORIENTATIONS:
        for resolution in RESOLUTIONS:
            for duration in DURATIONS:
                suite_name = _suite_name(resolution, duration, orientation)
                suite = EvaluationSuite.load(args.suites_dir / f"{suite_name}.json")
                summaries = {
                    version: json.loads(
                        (roots[version] / suite_name / "summary.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    for version in VERSIONS
                }
                headline = " · ".join(
                    f"{version} {summaries[version]['latency_seconds']['request_process']['median']:.3f}s"
                    for version in VERSIONS
                )
                cards = []
                for case in suite.cases:
                    videos = []
                    for version in VERSIONS:
                        result, source = _result(
                            roots[version], suite_name, case.case_id
                        )
                        link = (
                            media_root
                            / version.lower()
                            / suite_name
                            / f"{case.case_id}.mp4"
                        )
                        _link(source, link)
                        artifact_count += 1
                        timing = result["timing_seconds"]
                        videos.append(
                            f'''<section class="lane {version.lower()}"><h4>{version}<span>gen {timing["generation"]:.3f}s · E2E {timing["request_process"]:.3f}s</span></h4>
                            <video controls preload="metadata" playsinline style="aspect-ratio:{suite.width}/{suite.height}" data-src="{html.escape(link.relative_to(destination.parent).as_posix())}"></video></section>'''
                        )
                    cards.append(
                        f"""<article class="case"><h3>{html.escape(case.case_id)}</h3><div class="videos">{"".join(videos)}</div>
                        <details><summary>Prompt</summary><pre>{html.escape(case.prompt)}</pre></details></article>"""
                    )
                sections.append(
                    f'''<section class="condition" data-orientation="{orientation}" data-resolution="{resolution}" data-duration="{duration}">
                    <header><h2>{orientation} · {resolution}p · {duration}s</h2><p>{suite.width}×{suite.height}, {suite.num_frames} frames / {suite.fps} fps ({suite.num_frames / suite.fps:.3f}s encoded)<br>{headline}</p></header>
                    {"".join(cards)}</section>'''
                )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3-Flash · latency and portrait review</title><style>
:root{{color-scheme:dark;--bg:#080b10;--panel:#111722;--line:#283548;--text:#edf4ff;--muted:#9aabc0;--green:#69e0b4;--amber:#ffc96b}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 80% -10%,#183b4d 0,transparent 38rem),var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}.hero,main{{max-width:1600px;margin:auto;padding:26px}}h1{{font-size:clamp(36px,6vw,70px);line-height:.95;letter-spacing:-.05em;margin:10px 0 20px}}p,summary{{color:var(--muted)}}.filters{{display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:0;background:#080b10ee;padding:10px 0;z-index:3}}button{{background:var(--panel);color:var(--text);border:1px solid var(--line);padding:8px 11px;border-radius:8px;cursor:pointer}}button.active{{border-color:var(--green);color:var(--green)}}.condition{{margin:34px 0 58px}}.condition>header{{border-bottom:1px solid var(--line);margin-bottom:15px}}.condition h2{{font-size:28px;margin:0}}.case{{background:linear-gradient(145deg,#141c28,#0d131c);border:1px solid var(--line);border-radius:16px;padding:16px;margin:16px 0}}h3{{margin:0 0 10px}}.videos{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px}}h4{{font-size:12px;letter-spacing:.06em;margin:0 0 6px}}h4 span{{float:right;color:var(--muted);font-weight:400;letter-spacing:0}}.lossless h4{{color:var(--green)}}.flash h4{{color:var(--amber)}}video{{width:100%;max-height:72vh;object-fit:contain;background:#000;border-radius:9px}}details{{margin-top:10px}}pre{{white-space:pre-wrap;font:12px/1.5 ui-monospace,monospace}}@media(max-width:900px){{.videos{{grid-template-columns:1fr}}main,.hero{{padding:14px}}}}
</style></head><body><div class="hero"><div style="color:var(--green);font-weight:700">FluxMatrix · H3-Flash</div><h1>横版 / 竖版效果与时延</h1><p>12 个规格 × 4 个固定 prompt × 3 个版本，共 {artifact_count} 个视频。OFFICIAL 为 1×B200/请求；LOSSLESS、FLASH 为 8×B200/请求。显示的是驻留服务口径，模型加载与 startup warm-up 不计入请求时延。FLASH 改变 Turbo 权重和求解轨迹，不能当作无损结果。</p><div class="filters"><button class="active" data-filter="all">全部</button><button data-filter="landscape">横版</button><button data-filter="portrait">竖版</button><button data-filter="480">480p</button><button data-filter="768">768p</button><button data-filter="5">5s</button><button data-filter="10">10s</button><button data-filter="15">15s</button></div></div><main>{"".join(sections)}</main><script>
const loadVideo=v=>{{if(!v.src&&v.dataset.src)v.src=v.dataset.src}};
const observer=new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting){{loadVideo(e.target);observer.unobserve(e.target)}}}}),{{rootMargin:'700px'}});
document.querySelectorAll('video').forEach(v=>observer.observe(v));
for(const b of document.querySelectorAll('button'))b.onclick=()=>{{for(const x of document.querySelectorAll('button'))x.classList.remove('active');b.classList.add('active');let f=b.dataset.filter;for(const s of document.querySelectorAll('.condition'))s.hidden=f!=='all'&&![s.dataset.orientation,s.dataset.resolution,s.dataset.duration].includes(f)}};
</script></body></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    print(json.dumps({"artifacts": artifact_count, "page": str(destination)}))


if __name__ == "__main__":
    main()
