<div align="center">

# H3-Flash

**Open full-stack inference acceleration for MiniMax H3.**

**English** · [简体中文](README_zh-CN.md)

[Examples](#examples) · [Modes & latency](#inference-modes) ·
[Quick start](#quick-start) · [Web demo](#web-demo) · [How it works](#how-it-works)

</div>

H3-Flash is an open inference stack for
[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3). It generates a
5.167s video with synchronized audio in about **1.1 seconds at 480p** or
**2.5 seconds at 768p** on a warm, resident 8×B200 worker.

H3-Flash combines multi-GPU sequence parallelism, fused kernels, parallel VAE
decoding, a resident runtime, and an optional
[LightX2V Turbo4](https://huggingface.co/lightx2v/Minimax-h3-Turbo) adapter by
@ModelTC. The systems-only and approximate optimizations are kept in separate
modes, making the speed/quality boundary explicit and reproducible.

## Highlights

- **Low latency:** about 2.5 seconds of resident-worker end-to-end latency for
  5.167 seconds of 768p video and stereo audio on 8×B200.
- **Three modes:** OFFICIAL for reference, LOSSLESS for systems-only
  acceleration, and FLASH for the lowest latency.
- **Simple setup:** clone the repository, install the runtime, download the
  weights, and generate a video.

## News

- **2026-09-03** — Released H3-Flash, generating a five-second 768p video with
  synchronized audio in 2.5 seconds on 8×B200, with inference scripts and a
  self-hosted web demo.

## Examples

<table>
  <tr>
    <td width="50%"><video src="https://github.com/user-attachments/assets/e628987f-837d-483b-9cdb-fcf2e94aa090" controls></video><br><b>Glacier escape</b><br>Natural large-format action, resolved ice physics, and synchronized impact</td>
    <td width="50%"><video src="https://github.com/user-attachments/assets/35e9aa59-b9bc-47aa-9fef-942148794fd4" controls></video><br><b>Neon metro runner</b><br>Next-generation 3D game art, clean geometry, parallax, and synchronized action</td>
  </tr>
  <tr>
    <td width="50%"><video src="https://github.com/user-attachments/assets/3f621c4b-0599-41f7-8149-ee528c2965ca" controls></video><br><b>Giant dumpling</b><br>Contemporary character animation, stable anatomy, expressive reaction, and foley</td>
    <td width="50%"><video src="https://github.com/user-attachments/assets/bdc4b589-9db5-437f-80c4-68de7335758a" controls></video><br><b>Studio dance</b><br>Chinese solo performer, clean commercial portraiture, expressive choreography, and rhythmic audio</td>
  </tr>
</table>

These samples were generated with FLASH; their prompts and seeds are in
[`configs/evals/h3-showcase4-10s-v1.json`](configs/evals/h3-showcase4-10s-v1.json).

## Inference modes

- **OFFICIAL** · `1×B200` · **Reference** — official BF16 pipeline.
- **LOSSLESS** · `8×B200` · **No intentional approximation** — Ulysses SP8,
  parallel/compiled VAE, and fused kernels.
- **FLASH** · `8×B200` · **Approximate** — LOSSLESS with
  [LightX2V Turbo4](https://huggingface.co/lightx2v/Minimax-h3-Turbo) by @ModelTC.

**Warm resident-worker end-to-end latency (seconds)**

| Resolution | Nominal duration | OFFICIAL<br><sub>1×B200</sub> | LOSSLESS<br><sub>8×B200</sub> | FLASH<br><sub>8×B200</sub> |
|---:|---:|---:|---:|---:|
| 480p | 5s | 52.917 | 6.563 | **1.097** |
| 480p | 10s | 123.084 | 14.995 | **2.172** |
| 480p | 15s | 206.002 | 23.916 | **3.086** |
| 768p | 5s | 176.452 | 20.807 | **2.546** |
| 768p | 10s | 486.991 | 57.301 | **6.060** |
| 768p | 15s | 852.360 | 103.299 | **10.323** |

Each cell is the median of four fixed prompts after warm-up. E2E starts before
text encoding and ends after the video and complete audio have been written to
MP4. It excludes model loading, first-use compilation, prompt enhancement,
queueing, and network delivery. OFFICIAL uses the reference Diffusers/PyAV
writer; LOSSLESS and FLASH use the optimized FFmpeg writer. The 5s, 10s, and
15s cases contain 124, 243, and 345 frames at 24 FPS (5.167s, 10.125s, and
14.375s of encoded video). The same values and protocol are available as
[`benchmarks/results/b200_e2e.json`](benchmarks/results/b200_e2e.json).

The full 40-prompt evaluation set is available at
[`configs/evals/h3-broad40-v1.1.json`](configs/evals/h3-broad40-v1.1.json).

### Same prompt, three modes

Every row uses the same prompt, seed, resolution, and duration. The displayed
latency is warm resident-worker E2E, using the same boundary as the table above.

<table>
  <tr>
    <th>OFFICIAL · 1×B200</th>
    <th>LOSSLESS · 8×B200</th>
    <th>FLASH · 8×B200</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/2f008fb1-26ff-400a-972f-48cd3548029e" controls></video><br><b>Mandarin interview · 178.512s E2E</b></td>
    <td><video src="https://github.com/user-attachments/assets/da19514e-2aea-46d8-a846-b2666ad6b0ee" controls></video><br><b>Mandarin interview · 20.840s E2E</b></td>
    <td><video src="https://github.com/user-attachments/assets/f8b886e4-9156-4167-8918-e361c94df6cb" controls></video><br><b>Mandarin interview · 2.690s E2E</b></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/3d352a66-90d5-4de3-8013-6f5600eb3370" controls></video><br><b>Rooftop parkour · 176.437s E2E</b></td>
    <td><video src="https://github.com/user-attachments/assets/372533f8-da08-4252-9ec4-975a4d026cf7" controls></video><br><b>Rooftop parkour · 20.915s E2E</b></td>
    <td><video src="https://github.com/user-attachments/assets/16f26759-af27-419c-9ca5-9185ac9e943a" controls></video><br><b>Rooftop parkour · 2.593s E2E</b></td>
  </tr>
</table>

| Mode | Quality and efficiency | Best use |
|---|---|---|
| **OFFICIAL** | The official BF16 reference; slowest | Reference and validation |
| **LOSSLESS** | Same weights, sampling schedule, dense attention, and VAEs; outputs track OFFICIAL closely while floating-point order may differ | Maximum fidelity with lower latency |
| **FLASH** | Changes the weights and sampling trajectory; preserves the main subject, action, and sound, but composition may differ and visual or audio quality may be slightly lower (stronger prompt engineering can improve results) | Fast iteration and latency-sensitive serving |

### Prompting

H3-Flash follows H3's native prompting semantics. Start with the official
[H3 Prompt Writing Skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing)
and its [Base Prompt Guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt)
from [@MiniMax-AI](https://github.com/MiniMax-AI).

## Quick start

### Install

**Requirements:** Linux x86_64, 8×NVIDIA B200, an NVIDIA driver with
CUDA 13.x support, and roughly 300 GiB of free disk space.

```bash
git clone https://github.com/fluxmatrix/h3-flash.git
cd h3-flash

export H3_FLASH_RUNTIME_ROOT=/data/h3-flash-runtime
scripts/setup.sh
```

This installs the pinned dependencies in an isolated environment, downloads and
verifies the weights, prepares FLASH, and leaves System Python untouched. If the
model requires authentication, log in once and rerun `scripts/setup.sh`:

```bash
$H3_FLASH_RUNTIME_ROOT/venv/bin/python -m huggingface_hub.cli.hf auth login
scripts/setup.sh
```

### Generate a video

```bash
scripts/generate.sh \
  --mode FLASH \
  --prompt "A fast FPV flight through an autumn forest reveals a vast waterfall." \
  --output-dir "$H3_FLASH_RUNTIME_ROOT/artifacts/forest-waterfall"
```

The video is written to
`$H3_FLASH_RUNTIME_ROOT/artifacts/forest-waterfall/output.mp4`. Use
`--prompt-file` for long prompts, or select `LOSSLESS` or `OFFICIAL` with
`--mode`. This command starts a one-shot worker and therefore includes model
loading and warm-up; use the resident web service below for the latency reported
in the table.

Runtime files default to a sibling `.h3-flash-runtime` directory.

### Web demo

<video src="https://github.com/user-attachments/assets/7b7c546b-f6ec-49eb-8aa9-c31f834cf959" controls></video>

Enter a prompt, optionally enhance it, and generate and play the result in one page.

```bash
scripts/serve.sh
```

Open `http://127.0.0.1:8000`. The resident FLASH worker loads and warms up once,
then processes browser requests through a queue. Use `scripts/serve.sh LOSSLESS`
to serve the LOSSLESS mode instead. The demo has no authentication and listens
on localhost by default.

<details>
<summary><b>Tested software stack</b></summary>

- Python 3.12.14
- PyTorch 2.13.0+cu130
- Diffusers `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`

The bootstrap installs these pinned versions automatically.

</details>

## How it works

<table>
  <tbody>
    <tr><th colspan="3">LOSSLESS</th></tr>
    <tr>
      <th>Optimization</th>
      <th>Principle</th>
      <th>Median E2E reduction</th>
    </tr>
    <tr><td>Dense Ulysses SP8</td><td>Shard the sequence across eight GPUs</td><td><strong>−144.262 s (−81.3%)</strong>; 1→8 GPUs</td></tr>
    <tr><td>Video VAE tile parallelism</td><td>Decode official VAE tiles across GPUs, then restore their original order</td><td><strong>−4.624 s (−17.3%)</strong></td></tr>
    <tr><td>Triton elementwise/QK fusion</td><td>Fuse elementwise, QK normalization, and RoPE operations</td><td><strong>−3.598 s (−12.7%)</strong></td></tr>
    <tr><td>Packed Ulysses collectives</td><td>Pack Q, K, and V into one destination-major collective</td><td><strong>−2.464 s (−10.0%)</strong></td></tr>
    <tr><td>FFmpeg raw-video output</td><td>Pipe raw RGB frames and PCM audio directly to FFmpeg</td><td><strong>−0.961 s (−27.5%)</strong> in FLASH</td></tr>
    <tr><td>Static Video VAE compile</td><td>Compile the fixed-shape official decoder with PyTorch Inductor</td><td>−0.436 s (−2.0%); +50.289 s startup</td></tr>
    <tr><td>Rank-local input and compact output</td><td>Run projections per rank and gather only compact outputs</td><td>−0.136 s (−0.6%)</td></tr>
    <tr><td>Pinned D2H, invariant caches, allocator reuse</td><td>Reuse pinned buffers and cache request-invariant data</td><td>Small or neutral at whole-request level</td></tr>
    <tr><th colspan="3">FLASH</th></tr>
    <tr>
      <th>Optimization</th>
      <th>Principle</th>
      <th>Median E2E reduction</th>
    </tr>
    <tr><td>LightX2V Turbo4</td><td>Apply the Turbo LoRA and sample in four steps</td><td><strong>−18.215 s (−83.9%, 6.22×)</strong> vs LOSSLESS</td></tr>
  </tbody>
</table>

## Project layout

```text
apps/          Self-hosted web demo
assets/        Small, curated output examples
benchmarks/    Focused kernel and data-movement microbenchmarks
configs/       Evaluation suites and optimization metadata
locks/         Pinned upstream/model revisions and weight hashes
profiles/      Runnable OFFICIAL, LOSSLESS, FLASH, and ablation profiles
scripts/       Setup, inference, and benchmark entry points
src/h3_flash/  Inference runtime
tests/         CPU contract tests
tools/         Maintainer-only report and gallery generators
```

Model weights, virtual environments, upstream checkouts, FFmpeg binaries, and
generated artifacts are kept outside this repository.

## Roadmap / TODO

- [x] Released OFFICIAL, LOSSLESS, and FLASH inference modes
- [x] Resident web service with prompt input, job queue, and MP4 playback
- [ ] Use distillation and reinforcement learning to reach fewer sampling steps
  and higher quality without reducing human preference scores
- [ ] Optimize inference performance on other GPU architectures

## Acknowledgements

We thank the projects that made H3-Flash possible:

- **Model:** [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)
- **Runtime:** [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
  and PyTorch
- **Parallelism and kernels:**
  [DeepSpeed Ulysses](https://github.com/microsoft/DeepSpeed),
  [NVIDIA Sana](https://github.com/NVlabs/Sana),
  [SGLang](https://github.com/sgl-project/sglang), and
  [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- **Acceleration model:** [LightX2V](https://github.com/ModelTC/LightX2V) by
  @ModelTC
- **Community references:** [FastVideo](https://github.com/hao-ai-lab/FastVideo)
  and [fal](https://x.com/fal)
- **Media pipeline:** FFmpeg and libx264

FluxMatrix is responsible for the H3 integration, optimization composition,
validation gates, and measurements. Exact revisions and dependency roles are in
[`locks/upstreams.toml`](locks/upstreams.toml) and
[`THIRD_PARTY.md`](THIRD_PARTY.md).

## License

H3-Flash source code is licensed under [Apache-2.0](LICENSE). MiniMax H3 weights
are governed by the MiniMax H3 Community License Agreement; LightX2V model files
and other third-party components retain their respective terms. Model weights are
downloaded from upstream and are not distributed in this repository.

H3-Flash is under active development.
