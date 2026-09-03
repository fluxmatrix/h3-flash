# Third-party software and models

H3-Flash keeps model weights, Python environments, upstream source checkouts, and
FFmpeg binaries outside this repository. The bootstrap and model-download scripts
retrieve pinned upstream artifacts into `H3_FLASH_RUNTIME_ROOT`; this file records
their role and provenance.

The H3-Flash Apache-2.0 license applies only to H3-Flash source. It does not
replace the terms attached to third-party code, packages, binaries, or model
weights.

## Retrieved at runtime

| Component | Role | License or terms recorded upstream | Distribution |
|---|---|---|---|
| MiniMax H3 | Official BF16 model, processors, and VAEs | MiniMax H3 Community License Agreement | Downloaded from MiniMax; never bundled |
| LightX2V MiniMax-H3 Turbo | Optional four-step LoRA used by FLASH | Apache-2.0 model repository; MiniMax base-model terms still apply | Downloaded from LightX2V; never bundled |
| Qwen3.5-4B | Local prompt enhancer, visual observer, and interactive story writer | Apache-2.0 | Downloaded from Qwen; never bundled |
| Hugging Face Diffusers | Pinned H3 pipeline and context-parallel runtime | Apache-2.0 | Clean pinned checkout outside the repository |
| PyTorch, Triton, and Python dependencies | Execution and kernels | Individual package licenses | Hash-locked packages installed into the external virtual environment |
| FFmpeg with libx264 | H.264/AAC artifact writer | Pinned BtbN GPL build; component terms apply | Immutable archive downloaded outside the repository; binary not redistributed here |

## Implementation and design lineage

| Project | Relationship to H3-Flash | Runtime dependency |
|---|---|---|
| MiniMax H3 | Official architecture, preprocessing, sampler, and VAE behavior | Model and Diffusers implementation |
| Hugging Face Diffusers | H3 pipeline and native context-parallel implementation | Yes |
| DeepSpeed Ulysses | Sequence-parallel architecture | No direct DeepSpeed import |
| NVIDIA Sana / Sol-Attn | GB200 fusion and packed-layout implementation lineage | No Sana checkout required by public profiles |
| SGLang | Packed collective/layout design lineage | No direct SGLang import |
| FlashAttention | Dense FA4 implementation lineage for an excluded ablation | Not used by OFFICIAL, LOSSLESS, or FLASH |
| FastVideo | Blackwell FA4 integration pattern and comparison target | Not used by OFFICIAL, LOSSLESS, or FLASH |
| Qwen3.5-4B | Four-frame story continuity observation and H3 prompt writing | Optional local demo dependency |
| AI Toolkit | Historical first inference host adapter | Not used by the public runtime |

Exact Git commits and Hugging Face revisions are recorded in
[`locks/upstreams.toml`](locks/upstreams.toml). Downloaded weight paths, byte
sizes, and SHA-256 values are recorded in `locks/models.*.toml`. A prepared
FLASH model additionally carries `turbo-bake.json` and a verification marker
covering every derived Transformer shard.

If H3-Flash is distributed as a wheel, container, appliance, or hosted image, the
distributor must review the contents of that artifact and include all notices,
license texts, and source offers required by the components actually shipped.
