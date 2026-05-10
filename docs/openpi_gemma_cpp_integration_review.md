# openpi-Phi and gemma.cpp Integration Review

Date: 2026-05-10

This note reviews how the current `openpi-Phi` codebase works, what the local
`~/gemma.cpp` branch provides, and where the two projects can realistically be
combined on the local AMD `gfx1150` machine.

## Context

Local target machine:

- CPU: AMD Ryzen AI 9 HX PRO 370
- GPU: AMD Radeon 890M / `gfx1150`
- OS: Ubuntu 24.04.3
- ROCm: 7.2.1 on host

Relevant branches:

- `openpi-Phi`: current checkout is `main`; remote branch
  `origin/amd-gfx1150-bringup` documents the Docker JAX ROCm path for this
  machine.
- `~/gemma.cpp`: current branch is `paligemma2-vit-hip-gfx1150`, a fork branch
  focused on PaliGemma2 ViT HIP acceleration on `gfx1150`.

## openpi-Phi Core Mechanism

`openpi-Phi` is a robot policy codebase, not a plain language or VLM inference
runtime. The main user-facing inference path is:

```text
input observation dict
  -> policy transforms
  -> model.Observation
  -> Pi0.sample_actions()
  -> action chunk
  -> output transforms
```

The key files are:

- `src/openpi/policies/policy_config.py`
- `src/openpi/policies/policy.py`
- `src/openpi/models/pi0.py`
- `src/openpi/models/pi0_config.py`
- `src/openpi/training/config.py`
- `src/openpi/transforms.py`

### Policy Construction

`create_trained_policy()` loads either:

- a JAX/Orbax checkpoint from `params/`; or
- a PyTorch checkpoint if `model.safetensors` exists.

It then builds the full transform stack:

```text
optional repack transforms
default prompt injection
robot/data transforms
normalization
model transforms
```

The model transforms include 224 image resize, PaliGemma prompt tokenization,
and state/action padding.

### Inference Flow

`Policy.infer()`:

1. Copies the input observation dictionary.
2. Applies input transforms.
3. Converts inputs to JAX arrays or PyTorch tensors.
4. Builds `openpi.models.model.Observation`.
5. Calls `model.sample_actions(...)`.
6. Applies output transforms and unnormalization.

The output is an action chunk, not text.

### Model Architecture

`Pi0` and `Pi05` are built from:

- a PaliGemma-style image/language prefix expert;
- a separate Gemma action expert;
- action projection layers;
- flow-matching action sampling.

In `Pi0.sample_actions()`:

1. Images are encoded by `self.PaliGemma.img(...)`.
2. Prompt tokens are embedded by `self.PaliGemma.llm(..., method="embed")`.
3. Image and language embeddings form the prefix.
4. The prefix is run once to produce a KV cache.
5. A suffix/action expert loop iteratively denoises actions over `num_steps`
   flow-matching steps.
6. `action_out_proj` maps expert hidden states to robot action dimensions.

For `pi05`, robot state is discretized into the language prompt and the action
expert uses adaRMS time conditioning.

This means openpi cannot be replaced by a normal Gemma/PaliGemma text runtime
without reimplementing the action expert and flow sampling path.

## gemma.cpp Branch Review

The local `~/gemma.cpp` branch is:

```text
paligemma2-vit-hip-gfx1150
```

It is an experimental fork of `google/gemma.cpp` focused on accelerating the
PaliGemma2 ViT image encoder on local ROCm/HIP.

Key files:

- `experimental/paligemma2_vit_hip_backend.h`
- `experimental/paligemma2_vit_hip_backend.cc`
- `experimental/paligemma2_vit_hip_backend_probe.cc`
- `experimental/paligemma2_vit_hip_bench.cc`
- `gemma/gemma.cc`
- `gemma/gemma_args.h`
- `gemma/run.cc`
- `scripts/build_gemma_paligemma2_vit_hip.sh`
- `scripts/smoke_paligemma2_vit_hip_recommended.sh`
- `docs/paligemma2_*`

### What It Adds

The branch adds an experimental PaliGemma2 ViT image-token backend hook:

```text
Gemma::GenerateImageTokens()
  -> optional ImageTokensBackendFunc
  -> fallback CPU ViT path
```

The HIP backend can run the PaliGemma2 image encoder stages on `gfx1150` and
return decoder-ready image tokens when enabled with:

```text
--paligemma_vit_backend hip_probe
--paligemma_vit_hip_return_image_tokens 1
```

The current recommended HIP path includes:

```text
attention=f32_direct_qkv_pack_bf16
qkv=wmma2d8x4
attn_out=wmma2d8x4
mlp_up=wmma2d8x4
mlp_down=wmma12+residual for 448px
```

### Current Maturity

The branch is already useful as a performance and correctness lab:

- local build artifacts exist under `build/`;
- converted PaliGemma2 `.sbs` weights exist under `build/models/`;
- rocBLAS solution caches exist;
- smoke/profile scripts are present;
- docs record HIP event timing, hardware counters, roofline notes, and next
  kernel priorities.

Basic code hygiene checks passed during this review:

```bash
git diff --check origin/main..HEAD
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

Both produced no output.

### Important Boundary

The gemma.cpp branch accelerates PaliGemma2 image-token generation. It does not
implement openpi's full robot policy:

- no openpi action expert;
- no flow-matching denoising loop;
- no robot action normalization/unnormalization;
- no openpi policy transforms;
- no Orbax checkpoint loader.

Therefore, the branch is not a drop-in replacement for `Policy.infer()`.

## Best Integration Point

The best integration target is:

```text
openpi image observations
  -> openpi-compatible ViT HIP image-token backend
  -> image prefix embeddings/tokens
  -> existing openpi PaliGemma/action-expert policy
```

This is attractive because:

- `openpi` spends part of inference in image encoding.
- `openpi` processes multiple camera views.
- `gemma.cpp` already has `gfx1150`-specialized ViT kernels.
- The interface can be isolated around image-token generation before touching
  the rest of the policy.

This is also the safest boundary because it does not require reimplementing the
openpi action expert or flow-matching sampler in C++.

## Compatibility Gaps

The integration is not automatic. The main gaps are:

### Shape And Model Differences

`gemma.cpp` currently targets PaliGemma2:

- decoder model dim: commonly 2304 for PaliGemma2 3B;
- ViT model dim: 1152;
- image sizes: 224 and 448;
- `.sbs` weight format.

`openpi` uses a PaliGemma-style prefix expert:

- default PaliGemma width: 2048;
- action expert width: 1024 or 2048 depending config;
- openpi checkpoint format: JAX/Orbax params or PyTorch safetensors;
- output is action chunks, not generated text.

The ViT encoder looks close enough to investigate, but the image head and
prefix embedding shape need explicit validation.

### Weight Format

`gemma.cpp` expects `.sbs` weights loaded into `WeightsPtrs`.

`openpi` checkpoints are stored as:

```text
params/_METADATA
params/manifest.ocdbt
params/ocdbt.process_0/manifest.ocdbt
...
```

An adapter or exporter is needed before the HIP backend can consume openpi image
encoder weights.

### Runtime Interface

The gemma.cpp CLI path has HIP backend flags wired in. The C API path exists,
but the current multimodal C API does not expose the experimental HIP backend
options and should be reviewed before being used as an embedded library
boundary.

### Numerical Contract

The HIP path uses BF16 routes and F32 attention preservation decisions. Before
using it inside a robot policy, compare:

```text
JAX image tokens vs HIP image tokens
policy actions with JAX image encoder vs policy actions with HIP image encoder
```

Action drift matters more than text drift for openpi.

## Recommended Implementation Plan

### Phase 1: Offline Equivalence Harness

Do not change `Policy.infer()` yet.

Build a one-off harness that:

1. Loads an openpi checkpoint.
2. Extracts only image encoder weights.
3. Runs JAX `self.PaliGemma.img(...)` for a fixed 224 image.
4. Runs a gemma.cpp/HIP image-token path with exported equivalent weights.
5. Reports max absolute error, RMS error, and timing.

Success criteria:

- image-token shape matches openpi expectations;
- numerical error is stable and explainable;
- HIP runtime beats or meaningfully complements the JAX ROCm path on this
  machine.

### Phase 2: Python-Side Experimental Bridge

Expose the image-token backend through one of:

- a small pybind11 extension;
- a C API wrapper plus `ctypes`/`cffi`;
- a temporary subprocess harness for measurement only.

Keep the interface narrow:

```text
input: N images, HWC or NHWC float32/uint8
output: image embeddings/tokens in openpi prefix layout
```

At this stage, use it only in benchmark scripts, not in the production policy
server.

### Phase 3: Optional openpi Runtime Hook

If Phase 1 and Phase 2 show a real win, add an experimental openpi model path:

```text
Pi0.embed_prefix()
  -> if hip image backend enabled, use external image tokens
  -> else use self.PaliGemma.img(...)
```

This should be behind an explicit config or environment flag.

The existing JAX/PyTorch path must remain the correctness fallback.

### Phase 4: End-To-End Policy Benchmark

Benchmark full policy inference, not only image encoding:

- `pi05_aloha`
- local `pi05_base` checkpoint
- one camera vs three camera views
- fixed prompt/state/noise where possible
- action-chunk output comparison
- warmup and steady-state timing

The key question is whether image encoder acceleration moves full
`policy.infer()` latency enough to justify the extra integration complexity.

## Non-Recommended Paths

### Do Not Replace openpi With gemma.cpp End-To-End

This would require implementing:

- action expert Gemma;
- adaRMS time conditioning;
- state/action projection layers;
- flow-matching sampling;
- normalization and robot-specific transforms;
- checkpoint conversion for all policy weights.

That is a full inference runtime port, not an integration.

### Do Not Start With 448px

Openpi's current model transforms resize images to 224. The gemma.cpp 448 path
is valuable for PaliGemma2, but openpi integration should first match the actual
openpi policy shape.

### Do Not Optimize Before Profiling Full Policy

The local `gfx1150` JAX ROCm path already works through Docker. Before porting
the image encoder, measure where full openpi inference spends time:

- image encoder;
- prefix LLM prefill;
- repeated suffix/action expert flow steps;
- host/device transfer;
- Python transform overhead.

## Open Questions

1. Does openpi's image encoder weight layout exactly match the PaliGemma/SigLIP
   layout assumed by gemma.cpp?
2. What is the exact openpi image-token output shape per camera view?
3. How much of current `pi05_aloha` inference time is image encoding vs action
   expert flow sampling?
4. Can the HIP backend consume openpi-exported weights without going through a
   full `.sbs` model file?
5. How much action output drift is acceptable for a BF16 HIP image-token path?
6. Should the integration target JAX openpi first, or the PyTorch openpi path
   after JAX-to-PyTorch conversion?

## Practical Next Step

The next concrete task should be a measurement harness, not a production
integration:

```text
openpi fixed image -> JAX image tokens
same image -> gemma.cpp HIP image tokens
compare shape/error/time
```

Once this passes, the next target is an end-to-end `policy.infer()` benchmark
with a pluggable image-token provider.

## Initial Execution Results

This section records the first execution pass from 2026-05-10.

### Added Local Tools

Two local scripts were added:

- `scripts/inspect_openpi_img_params.py`
- `scripts/benchmark_policy_breakdown.py`

`inspect_openpi_img_params.py` reads Orbax checkpoint metadata only; it does not
load the full checkpoint tensors. This keeps shape inspection cheap.

`benchmark_policy_breakdown.py` measures:

- total `Policy.infer()`;
- input transforms;
- host-to-device observation construction;
- JAX image embedding;
- JAX text/prefix assembly;
- JAX prefix LLM prefill;
- JAX suffix/action flow loop;
- output transforms.

The benchmark script currently supports the JAX policy path.

### Required Local Fix

The current `main` checkout imported `lerobot` at module import time through
`src/openpi/training/data_loader.py`. This prevented policy inference utilities
from importing `policy_config` in the local ROCm Docker image, because the image
does not include `lerobot`.

The same minimal fix already present on the `amd-gfx1150-bringup` branch was
applied locally:

```text
move lerobot.common.datasets.lerobot_dataset import
from module import time
to create_torch_dataset() after the repo_id == "fake" fast path
```

This keeps debug/fake and inference utilities from requiring `lerobot` unless a
real LeRobot dataset is actually constructed.

### Image Encoder Weight Shapes

Command shape:

```bash
PYTHONPATH=/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src \
python scripts/inspect_openpi_img_params.py \
  --checkpoint-dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_base
```

Important `pi05_base` image encoder metadata:

| Path | Shape | Interpretation |
| --- | ---: | --- |
| `PaliGemma/img/embedding/kernel` | `(14, 14, 3, 1152)` | ViT patch embedding |
| `PaliGemma/img/pos_embedding` | `(1, 256, 1152)` | 224px positional embedding |
| `PaliGemma/img/Transformer/encoderblock/LayerNorm_0/*` | `(27, 1152)` | pre-attention layernorm |
| `PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/{query,key,value}/kernel` | `(27, 1152, 16, 72)` | Q/K/V projections |
| `PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel` | `(27, 16, 72, 1152)` | attention output projection |
| `PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel` | `(27, 1152, 4304)` | MLP up |
| `PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel` | `(27, 4304, 1152)` | MLP down |
| `PaliGemma/img/Transformer/encoder_norm/*` | `(1152)` | final encoder norm |
| `PaliGemma/img/head/kernel` | `(1152, 2048)` | image-token projection |
| `PaliGemma/img/head/bias` | `(2048)` | image-token projection bias |

This confirms that the openpi image encoder is very close to the PaliGemma2 ViT
shape family already targeted by the `gemma.cpp` HIP work:

- 224px image path;
- 14x14 patches;
- 256 image tokens;
- ViT width 1152;
- 27 layers;
- 16 heads;
- head dim 72;
- MLP dim 4304.

The main integration-specific difference is the openpi image-token projection
head output width:

```text
openpi pi05_base img/head/kernel: (1152, 2048)
```

The current `gemma.cpp` PaliGemma2 3B path uses a different decoder width, so
the HIP backend must be made openpi-weight-aware instead of assuming the
existing PaliGemma2 `.sbs` model shape.

### Policy Inference Breakdown

Command shape:

```bash
PYTHONPATH=/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src \
python scripts/benchmark_policy_breakdown.py \
  --config pi05_aloha \
  --checkpoint-dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_base \
  --warmup 1 \
  --runs 3 \
  --num-steps 10
```

Observed result on the local `gfx1150` Docker JAX ROCm path:

| Stage | Avg seconds | Median seconds |
| --- | ---: | ---: |
| total `policy.infer()` | 1.3091 | 1.3069 |
| input transforms | 0.0006 | 0.0006 |
| host-to-device observation | 0.0108 | 0.0110 |
| image embedding, 3 camera views | 0.1684 | 0.1682 |
| text/prefix assembly | 0.0064 | 0.0064 |
| prefix LLM prefill | 0.6801 | 0.6926 |
| composed prefix total | 0.8550 | 0.8675 |
| suffix/action flow loop | 0.3966 | 0.3934 |
| output transforms | 0.0034 | 0.0034 |

The result is directionally clear even with only three measured runs:

- prefix work is the largest block;
- prefix LLM prefill is larger than image embedding;
- three-view image embedding is still a measurable target at about 0.17s;
- suffix/action flow is the second-largest block;
- Python transforms are not a meaningful bottleneck;
- image/prefix acceleration is a plausible next target, but image-only
  acceleration cannot remove the full prefix cost.

The next measurement should compare image embeddings directly:

```text
JAX self.PaliGemma.img(...) image tokens
vs
gemma.cpp HIP ViT image tokens
```

That comparison should use the same 224px transformed images and the same
openpi `pi05_base` image weights.

### OpenPI Image Encoder Golden Artifact

Added:

```text
scripts/export_openpi_image_encoder_golden.py
```

This script creates a deterministic OpenPI-side reference artifact for the
image encoder boundary. It saves:

- the exact NHWC float32 image tensor consumed by `model.PaliGemma.img`;
- the valid image mask;
- decoder-ready image token embeddings;
- selected intermediate tensors for future lower-level debugging.

Command used on the local `gfx1150` ROCm/JAX Docker path:

```bash
PYTHONPATH=/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src \
python scripts/export_openpi_image_encoder_golden.py \
  --config pi05_aloha \
  --checkpoint-dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_base \
  --example ALOHA \
  --camera base_0_rgb \
  --encoder-layer-limit 1 \
  --numpy-seed 0 \
  --output build/openpi_golden/pi05_aloha_base_0_rgb_image_encoder.npz
```

Generated local artifact:

```text
build/openpi_golden/pi05_aloha_base_0_rgb_image_encoder.npz
sha256 363ff1842302a553f1f46a3a65c2725f51a8f14cd71a81c781347f8424bf5d45
numpy_seed 0
encoder_layer_limit 1
```

The artifact is intentionally under `build/` and is not part of the git diff.

Saved arrays:

| Array | Shape | Meaning |
| --- | ---: | --- |
| `image` | `(1, 224, 224, 3)` | exact image encoder input, float32 in `[-1, 1]` |
| `image_mask` | `(1,)` | camera-valid mask |
| `tokens` | `(1, 256, 2048)` | OpenPI decoder-ready image tokens |
| `intermediate_stem` | `(1, 16, 16, 1152)` | patch embedding output |
| `intermediate_with_posemb` | `(1, 256, 1152)` | flattened tokens after positional embedding |
| `intermediate_encoder_block00_sa` | `(1, 256, 1152)` | block00 self-attention branch output |
| `intermediate_encoder_block00_plus_sa` | `(1, 256, 1152)` | block00 post-attention residual |
| `intermediate_encoder_block00_mlp` | `(1, 256, 1152)` | block00 MLP branch output |
| `intermediate_encoder_block00_plus_mlp` | `(1, 256, 1152)` | block00 post-MLP residual |
| `intermediate_encoded` | `(1, 256, 1152)` | ViT encoder output |
| `intermediate_pre_logits_2d` | `(1, 16, 16, 1152)` | 2D encoder output before head |
| `intermediate_logits_2d` | `(1, 16, 16, 2048)` | 2D image-token projection output |

Observed value ranges:

| Array | Min | Max | Mean |
| --- | ---: | ---: | ---: |
| `image` | -1.0 | 1.0 | -0.00156415 |
| `tokens` | -123.0 | 21.0 | -0.0627844 |
| `intermediate_encoded` | -22.875 | 30.625 | -0.0017854 |
| `intermediate_encoder_block00_sa` | -18.25 | 18.125 | 0.0259743 |
| `intermediate_encoder_block00_plus_sa` | -84.0 | 282.0 | 0.0585337 |
| `intermediate_encoder_block00_mlp` | -33.75 | 19.125 | 0.00573317 |
| `intermediate_encoder_block00_plus_mlp` | -81.5 | 284.0 | 0.0642721 |
| `intermediate_stem` | -3.09359 | 3.00532 | 0.0361804 |
| `intermediate_with_posemb` | -85.282 | 284.333 | 0.0325634 |

This is the reference file that a future `gemma.cpp` OpenPI-aware HIP backend
should reproduce before it is allowed to replace `model.PaliGemma.img` at
runtime.

### OpenPI Image Encoder Weight Export

Added:

```text
scripts/export_openpi_image_encoder_weights.py
scripts/validate_openpi_image_encoder_weight_export.py
```

The weight exporter restores only the `params/PaliGemma/img` subtree from the
Orbax checkpoint. It does not restore the full LLM or action expert. Because
the OpenPI policy path loads checkpoint parameters as `bfloat16`, the exporter
defaults to:

```text
restore_dtype = bfloat16
export_dtype = float32 container with bfloat16-converted values
```

This keeps the `.npz` readable from C++/NumPy-style tooling while matching the
runtime values used by OpenPI.

Full 27-layer command:

```bash
PYTHONPATH=/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src \
python scripts/export_openpi_image_encoder_weights.py \
  --checkpoint-dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_base \
  --restore-dtype bfloat16 \
  --layer-limit 0 \
  --output build/openpi_golden/pi05_base_image_encoder_weights_bfloat16_all_layers.npz
```

Generated local artifact:

```text
build/openpi_golden/pi05_base_image_encoder_weights_bfloat16_all_layers.npz
sha256 e3c4f9f9ecbbe81b989d2929459a1694a20aa4414f1778ae66adda35dd244203
size 780,413,874 bytes
uncompressed npy payload 1,659,243,744 bytes
```

A smaller smoke artifact was also generated:

```text
build/openpi_golden/pi05_base_image_encoder_weights_bfloat16_first_1_layers.npz
sha256 e7c1745dd229522935f54b45bccf29bbf83d159bb94f45f77f420e280de33bee
```

The stable exported aliases are:

| Alias | Full export shape | Layout |
| --- | ---: | --- |
| `patch_embed_kernel` | `(14, 14, 3, 1152)` | HWIO |
| `patch_embed_bias` | `(1152,)` | output width |
| `pos_embedding` | `(1, 256, 1152)` | batch, tokens, width |
| `block_ln0_scale`, `block_ln0_bias` | `(27, 1152)` | layer, width |
| `block_q_kernel`, `block_k_kernel`, `block_v_kernel` | `(27, 1152, 16, 72)` | layer, width, heads, head_dim |
| `block_q_bias`, `block_k_bias`, `block_v_bias` | `(27, 16, 72)` | layer, heads, head_dim |
| `block_attn_out_kernel` | `(27, 16, 72, 1152)` | layer, heads, head_dim, width |
| `block_attn_out_bias` | `(27, 1152)` | layer, width |
| `block_ln1_scale`, `block_ln1_bias` | `(27, 1152)` | layer, width |
| `block_mlp_up_kernel` | `(27, 1152, 4304)` | layer, width, mlp_dim |
| `block_mlp_up_bias` | `(27, 4304)` | layer, mlp_dim |
| `block_mlp_down_kernel` | `(27, 4304, 1152)` | layer, mlp_dim, width |
| `block_mlp_down_bias` | `(27, 1152)` | layer, width |
| `encoder_norm_scale`, `encoder_norm_bias` | `(1152,)` | width |
| `head_kernel` | `(1152, 2048)` | width, decoder_width |
| `head_bias` | `(2048,)` | decoder_width |

Validation command:

```bash
PYTHONPATH=/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src \
python scripts/validate_openpi_image_encoder_weight_export.py \
  --golden build/openpi_golden/pi05_aloha_base_0_rgb_image_encoder.npz \
  --weights build/openpi_golden/pi05_base_image_encoder_weights_bfloat16_all_layers.npz \
  --strict
```

Strict validation passed with:

| Check | Max abs | RMS | Mean abs |
| --- | ---: | ---: | ---: |
| patch embedding vs `intermediate_stem` | 2.6226044e-06 | 1.879537e-07 | 1.140688e-07 |
| posemb vs `intermediate_with_posemb` | 2.6226044e-06 | 1.884018e-07 | 1.1405609e-07 |
| bfloat16 head projection vs `tokens` | 0.0625 | 0.0002298438 | 1.8859319e-06 |

The head projection max difference is expected for this validation level: the
OpenPI runtime path is bfloat16, and the exported artifact stores bfloat16
runtime values in a float32 container for portability.

### C++ Raw Bundle And gemma.cpp Smoke Probe

Added in `openpi-Phi`:

```text
scripts/export_openpi_image_encoder_raw_bundle.py
```

This converts the `.npz` golden and weight artifacts into a C++-friendly raw
bundle:

```bash
PYTHONPATH=/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src \
python scripts/export_openpi_image_encoder_raw_bundle.py \
  --golden build/openpi_golden/pi05_aloha_base_0_rgb_image_encoder.npz \
  --weights build/openpi_golden/pi05_base_image_encoder_weights_bfloat16_all_layers.npz \
  --output-dir build/openpi_golden/pi05_base_image_encoder_raw_bundle
```

Generated local bundle:

```text
build/openpi_golden/pi05_base_image_encoder_raw_bundle/
manifest sha256 39f1efc7934b339b71ab51a6307bea73a2e8f12448e451b1bec9672e9e84ffcc
files 34
size 1.6G
raw array payload bytes 1,672,268,736
```

Every `.f32` file in the bundle is a contiguous little-endian float32 array.
The bundle deliberately avoids NPZ/ZIP parsing so C++ can verify the OpenPI
layout before a more permanent loader exists.

Added in `~/gemma.cpp`:

```text
experimental/openpi_image_encoder_bundle_probe.cc
scripts/build_openpi_image_encoder_bundle_probe.sh
```

Build command:

```bash
scripts/build_openpi_image_encoder_bundle_probe.sh
```

Run command:

```bash
build/openpi_image_encoder_bundle_probe \
  --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle \
  --strict 1
```

Observed result:

| Check | Max abs | RMS | Mean abs |
| --- | ---: | ---: | ---: |
| C++ patch embedding vs OpenPI `intermediate_stem` | 2.38418579e-06 | 1.69301979e-07 | 1.00710379e-07 |
| C++ posemb vs OpenPI `intermediate_with_posemb` | 2.38418579e-06 | 1.69877395e-07 | 1.00728254e-07 |
| C++ block00 `with_posemb + sa` residual | 0.629394531 | 0.00362788769 | 0.00152501243 |
| C++ block00 `+sa + mlp` residual | 0.53125 | 0.00567729978 | 0.00269195713 |
| C++ sampled head projection vs OpenPI `tokens` | 0.0261551773 | 0.00561710776 | 0.00267284588 |

This proves that the C++ side can already consume the exported OpenPI image
input, image encoder weights, and selected golden outputs with the intended
layouts. It still does not validate the full 27-layer transformer in C++; that
is the next integration boundary.

The probe was then extended with an optional full CPU reference forward for
OpenPI ViT `block00`:

```bash
build/openpi_image_encoder_bundle_probe \
  --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle \
  --strict 1 \
  --block00_forward 1
```

Observed `block00` CPU reference result:

| Check | Max abs | RMS | Mean abs |
| --- | ---: | ---: | ---: |
| block00 attention branch `sa` vs golden | 0.125 | 0.00462472531 | 0.00145769968 |
| block00 post-attention residual `+sa` vs golden | 0.125 | 0.00495657497 | 0.00145967934 |
| block00 MLP branch `mlp` vs golden | 0.25 | 0.00936971061 | 0.00502001511 |
| block00 post-MLP residual `+mlp` vs golden | 0.5 | 0.012148609 | 0.00635270804 |

This is the first end-to-end C++ reproduction of an OpenPI ViT transformer
block using exported OpenPI weights and OpenPI golden activations. The remaining
CPU reference work is mechanical expansion from `block00` to all 27 blocks,
then final encoder norm and head projection.

The probe was then extended to run the full 27-layer OpenPI ViT image encoder,
including final encoder norm and the OpenPI `1152 -> 2048` image-token head:

```bash
build/openpi_image_encoder_bundle_probe \
  --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle \
  --strict 1 \
  --all_layers_forward 1
```

Observed full CPU reference result:

| Check | Max abs | RMS | Mean abs |
| --- | ---: | ---: | ---: |
| full 27-layer encoded tokens vs `intermediate_encoded` | 1.0 | 0.0244327946 | 0.0162399451 |
| full 27-layer image tokens vs `tokens` | 2.5 | 0.0295090993 | 0.0186109645 |

Runtime for the full CPU reference command was about 13.8 seconds on the local
Ryzen AI 9 HX PRO 370 CPU. This establishes a complete C++ numerical baseline
for the OpenPI image encoder. The next engineering step is to replace pieces of
this CPU reference with the existing `gemma.cpp` HIP kernels while preserving
the same golden comparisons.

### First OpenPI HIP Raw-Bundle Probe

Added in `~/gemma.cpp`:

```text
experimental/openpi_image_encoder_hip_probe.cc
scripts/build_openpi_image_encoder_hip_probe.sh
```

This is intentionally smaller than a full HIP image encoder. It now validates
one complete OpenPI ViT transformer block on HIP:

```text
OpenPI raw bundle
  -> HIP block00 LN0 into BF16
  -> rocBLAS BF16 GEMM for block00 Q/K/V projections
  -> HIP bias/add BF16 rounding
  -> HIP reference attention
  -> rocBLAS BF16 GEMM for block00 attention output projection
  -> HIP post-attention residual
  -> HIP LN1 into BF16
  -> rocBLAS BF16 GEMM for MLP up
  -> HIP GELU
  -> rocBLAS BF16 GEMM for MLP down
  -> HIP post-MLP residual
  -> compare against C++ CPU BF16 references
  -> compare block00 activations against OpenPI golden activations
```

Build command:

```bash
scripts/build_openpi_image_encoder_hip_probe.sh
```

Run command:

```bash
build/openpi_image_encoder_hip_probe \
  --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle \
  --strict 1 \
  --runs 20 \
  --warmup 3
```

Observed result:

| Check | Max abs | RMS | Mean abs |
| --- | ---: | ---: | ---: |
| HIP block00 LN0 vs CPU BF16 reference | 0.0078125 | 1.66051843e-05 | 5.19349566e-08 |
| HIP block00 Q projection vs CPU BF16 reference | 0.03125 | 0.00015605966 | 2.10324914e-06 |
| HIP block00 K projection vs CPU BF16 reference | 0.015625 | 0.000118654033 | 2.13355189e-06 |
| HIP block00 V projection vs CPU BF16 reference | 0.015625 | 6.10053286e-05 | 1.06213561e-06 |
| HIP block00 attention, CPU Q/K/V input, vs CPU reference | 0 | 0 | 0 |
| HIP block00 attention, GPU Q/K/V input, vs CPU reference | 0.0078125 | 8.72262315e-05 | 4.25100345e-06 |
| HIP block00 SA vs CPU BF16 reference | 0.125 | 0.00067444799 | 3.32855865e-05 |
| HIP block00 SA vs OpenPI golden | 0.1875 | 0.00529909496 | 0.001849967 |
| HIP block00 post-attention residual vs CPU BF16 reference | 0.125 | 0.000698557015 | 3.32099605e-05 |
| HIP block00 post-attention residual vs OpenPI golden | 0.5 | 0.00621018738 | 0.00219360618 |
| HIP block00 LN1 vs CPU BF16 reference | 0.001953125 | 4.52029942e-06 | 1.62528499e-08 |
| HIP block00 MLP up vs CPU BF16 reference | 0.03125 | 0.000234082574 | 2.33911108e-06 |
| HIP block00 MLP up + GELU vs CPU BF16 reference | 0.015625 | 3.27898089e-05 | 1.44099179e-07 |
| HIP block00 MLP down vs CPU BF16 reference | 0.0625 | 0.00037147947 | 1.15224895e-05 |
| HIP block00 MLP branch vs OpenPI golden | 0.25 | 0.0103216943 | 0.00568126502 |
| HIP block00 post-MLP residual vs CPU BF16 reference | 0.125 | 0.000430540042 | 1.14635461e-05 |
| HIP block00 post-MLP residual vs OpenPI golden | 0.5 | 0.0132319156 | 0.00714620822 |

Timing:

```text
hip_block00_ln0_ms=0.0365944
hip_block00_q_projection_gemm_ms=0.927322
hip_block00_k_projection_gemm_ms=0.892595
hip_block00_v_projection_gemm_ms=1.04954
hip_block00_qkv_projection_total_gemm_ms=2.86945
hip_block00_attention_cpu_qkv_ms=24.1305
hip_block00_attention_ms=24.1592
hip_block00_sa_projection_gemm_ms=0.703342
hip_block00_ln1_ms=0.0282551
hip_block00_mlp_up_gemm_ms=3.05915
hip_block00_gelu_ms=0.288462
hip_block00_mlp_down_gemm_ms=2.94982
Q/K/V and attention-out shape: rows=256 in=1152 out=1152
MLP-up shape: rows=256 in=1152 out=4304
MLP-down shape: rows=256 in=4304 out=1152
```

This proves that the exported OpenPI raw-bundle layout can feed a native
`gfx1150` HIP/rocBLAS BF16 path directly through a complete OpenPI transformer
block.
It validates:

- raw `.f32` OpenPI weight loading on the C++ side;
- OpenPI block00 LN0/LN1 scale/bias and BF16 output on HIP;
- OpenPI Q/K/V projection layout `(1152, 16, 72)` flattened as `1152 x 1152`;
- host BF16 conversion into `rocblas_bfloat16`;
- row-major GEMM mapping through rocBLAS column-major arguments;
- attention output projection layout `(16, 72, 1152)` flattened as
  `1152 x 1152`;
- OpenPI MLP up/down layouts `(1152, 4304)` and `(4304, 1152)`;
- GELU and both residual boundaries;
- numerical agreement against the CPU BF16 reference.

The attention kernel used here is deliberately a correctness-first HIP
reference kernel: one `(token, head)` block executes the softmax/AV path in the
same serial order as the CPU reference. A parallel reduction version was faster
but produced larger max error because softmax is sensitive to accumulation
order. This makes the current probe suitable for integration validation, not
yet for final performance.

The same probe now also loops the complete block path over all 27 ViT layers,
then runs final encoder norm and the OpenPI `1152 -> 2048` image-token head.

Observed full-image-encoder result:

| Check | Max abs | RMS | Mean abs |
| --- | ---: | ---: | ---: |
| HIP full 27-layer encoded output vs OpenPI `intermediate_encoded` | 1.8125 | 0.023977572 | 0.0157405035 |
| HIP full 27-layer image tokens vs OpenPI `tokens` | 2.0 | 0.0288557254 | 0.0180431094 |

Initial full path timing sample, before resident device weights:

```text
hip_all_layers_timing_ms
ln=4.04516
qkv=68.1325
attention=702.031
sa=29.195
mlp_up=90.0566
gelu=4.13097
mlp_down=86.0122
final_norm=0.036388
head=1.69456
```

The full path was then changed to make the projection kernels, projection
biases, and layernorm scale/bias vectors resident on the GPU for the 27-layer
run. This removes the previous per-layer host slicing, BF16 conversion, and
host-to-device weight copies from the measured loop while preserving the same
golden comparisons.

Resident-weight full path timing sample:

```text
hip_all_layers_timing_ms
ln=2.89072
qkv=46.8257
attention=597.536
sa=16.1216
mlp_up=56.9189
gelu=2.90256
mlp_down=56.387
final_norm=0.039395
head=0.814521
```

The resident-weight run preserved the same final numerical result:

```text
hip_all_layers_encoded_vs_golden max_abs=1.8125 rms=0.023977572 mean_abs=0.0157405035
hip_all_layers_tokens_vs_golden max_abs=2 rms=0.0288557254 mean_abs=0.0180431094
```

The probe code now makes this backend boundary explicit rather than keeping the
full-image-encoder loop inline in `main()`: `OpenPiImageEncoderHipBackend` owns
`OpenPiImageEncoderResidentWeights`, including the final encoder norm and image
head weights, while `RunFullImageEncoderResident(...)` owns the per-run scratch
buffers, execution, timing, and golden comparison. A post-refactor strict serial
run rebuilt successfully and preserved the same final envelope:

```text
build/openpi_image_encoder_hip_probe --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle --strict 1 --runs 5 --warmup 1
hip_all_layers_encoded_vs_golden max_abs=1.8125 rms=0.023977572 mean_abs=0.0157405035
hip_all_layers_tokens_vs_golden max_abs=2 rms=0.0288557254 mean_abs=0.0180431094
hip_all_layers_timing_ms ln=4.77037 qkv=49.3767 attention=716.391 sa=19.6274 mlp_up=57.2915 gelu=4.74886 mlp_down=55.8183 final_norm=0.086944 head=0.890496
```

This boundary has now been split out of the monolithic probe into:

```text
experimental/openpi_image_encoder_hip_backend.h
experimental/openpi_image_encoder_hip_backend.cc
```

The probe still owns raw-bundle loading and block00 per-op validation, but the
resident full-image-encoder execution is now built as a separate object and
linked by `scripts/build_openpi_image_encoder_hip_probe.sh`. A split-backend
strict hybrid run preserved the same envelope:

```text
build/openpi_image_encoder_hip_probe --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle --attention hybrid --strict 1 --runs 5 --warmup 1
hip_all_layers_encoded_vs_golden max_abs=1.8125 rms=0.023977572 mean_abs=0.0157405035
hip_all_layers_tokens_vs_golden max_abs=2 rms=0.0288557254 mean_abs=0.0180431094
hip_all_layers_timing_ms ln=3.55899 qkv=57.4128 attention=97.4592 sa=18.5494 mlp_up=65.5394 gelu=3.03223 mlp_down=62.8253 final_norm=0.042358 head=0.769898
```

The backend now also exposes a non-golden execution API:

```cpp
OpenPiImageEncoderHipOutput Run(
    const std::vector<float>& with_posemb,
    const OpenPiImageEncoderHipOptions& options) const;
```

`RunWithGolden(...)` is now a probe-only validation wrapper around `Run(...)`.
This is the first narrow image-token API boundary needed before wiring the
backend into a real OpenPI/gemma.cpp bridge. A post-API-split strict hybrid run
again preserved the same result:

```text
hip_all_layers_encoded_vs_golden max_abs=1.8125 rms=0.023977572 mean_abs=0.0157405035
hip_all_layers_tokens_vs_golden max_abs=2 rms=0.0288557254 mean_abs=0.0180431094
hip_all_layers_timing_ms ln=4.52261 qkv=52.5017 attention=94.137 sa=19.3205 mlp_up=61.7879 gelu=4.37324 mlp_down=63.3368 final_norm=0.04328 head=1.37137
```

The backend input boundary has since been moved one step earlier. It now owns
resident patch-embedding weights, patch bias, and positional embedding, and
exposes image-entry APIs:

```cpp
OpenPiImageEncoderHipOutput RunFromImage(
    const std::vector<float>& image,
    const OpenPiImageEncoderHipOptions& options) const;

OpenPiImageEncoderHipImageRunResult RunFromImageWithGolden(...);
```

The probe still keeps block00 per-op validation against the exported
`with_posemb`, but the full resident backend path now runs from
`golden_image.f32` through patch embed, posemb, 27 ViT layers, final norm, and
image-token head. The non-golden `RunFromImage(...)` path keeps stem and
posemb on device; only `RunFromImageWithGolden(...)` copies those intermediate
tensors back for validation. A strict hybrid run produced:

```text
hip_patch_embed_stem_vs_golden max_abs=5.24520874e-06 rms=2.6804308e-07 mean_abs=1.70283213e-07
hip_patch_embed_with_posemb_vs_golden max_abs=7.62939453e-06 rms=2.69236309e-07 mean_abs=1.70298267e-07
hip_all_layers_encoded_vs_golden max_abs=1.3125 rms=0.0241603118 mean_abs=0.0157843356
hip_all_layers_tokens_vs_golden max_abs=2 rms=0.029132315 mean_abs=0.0181650731
hip_patch_embed_ms=2.62556
hip_all_layers_timing_ms ln=3.55317 qkv=54.1025 attention=96.0919 sa=17.8706 mlp_up=67.153 gelu=3.79321 mlp_down=68.7326 final_norm=0.078867 head=1.06059
```

A run-only smoke binary now exercises that non-golden API without block00
validation:

```text
experimental/openpi_image_encoder_hip_run.cc
scripts/build_openpi_image_encoder_hip_run.sh
```

The raw-bundle file contract has also been centralized so probe/run entry
points no longer duplicate the same filename and shape table:

```text
experimental/openpi_image_encoder_raw_bundle.h
experimental/openpi_image_encoder_raw_bundle.cc
```

Both HIP binaries now build and link this loader as a separate object.
The raw-bundle-to-backend constructor mapping is also centralized:

```text
experimental/openpi_image_encoder_hip_bundle_bridge.h
experimental/openpi_image_encoder_hip_bundle_bridge.cc
```

The bridge helper exposes:

```cpp
std::unique_ptr<OpenPiImageEncoderHipBackend>
CreateOpenPiImageEncoderHipBackendFromRawBundle(
    const OpenPiImageEncoderRawBundle& bundle);
```

This keeps `openpi_image_encoder_hip_probe` and
`openpi_image_encoder_hip_run` from carrying a long positional constructor
mapping, and leaves a narrow place to replace raw-bundle inputs with real
OpenPI/gemma.cpp weight sources later.

Smoke command:

```bash
build/openpi_image_encoder_hip_run --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle --attention hybrid --runs 5 --warmup 1
```

Observed output summary:

```text
openpi_image_encoder_hip_run attention=hybrid tokens=524288 encoded=294912
encoded_summary sum=-518.354457 mean=-0.00175765807 mean_abs=0.532370224 min=-23.125 max=31
tokens_summary sum=-33024.3571 mean=-0.0629889624 mean_abs=1.2444961 min=-123 max=21
hip_image_encoder_timing_ms patch=3.3658 ln=4.0907 qkv=65.7526 attention=95.7849 sa=18.6739 mlp_up=64.1501 gelu=4.06915 mlp_down=61.8977 final_norm=0.287456 head=0.914184
```

After the shared-loader split, both verification paths still pass. Probe
strict hybrid:

```text
hip_patch_embed_stem_vs_golden max_abs=5.24520874e-06 rms=2.6804308e-07 mean_abs=1.70283213e-07
hip_patch_embed_with_posemb_vs_golden max_abs=7.62939453e-06 rms=2.69236309e-07 mean_abs=1.70298267e-07
hip_all_layers_encoded_vs_golden max_abs=1.3125 rms=0.0241603118 mean_abs=0.0157843356
hip_all_layers_tokens_vs_golden max_abs=2 rms=0.029132315 mean_abs=0.0181650731
```

Run-only smoke:

```text
openpi_image_encoder_hip_run attention=hybrid image=<bundle> tokens=524288 encoded=294912
tokens_summary sum=-33024.3571 mean=-0.0629889624 mean_abs=1.2444961 min=-123 max=21
```

The run-only binary now also accepts an explicit external image tensor:

```bash
build/openpi_image_encoder_hip_run \
  --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle \
  --image /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle/golden_image.f32 \
  --attention hybrid --runs 3 --warmup 1
```

This expects the same OpenPI-normalized `224 x 224 x 3` F32 layout as
`golden_image.f32`. The explicit-image run produced the same token summary:

```text
openpi_image_encoder_hip_run attention=hybrid image=/home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle/golden_image.f32 tokens=524288 encoded=294912
tokens_summary sum=-33024.3571 mean=-0.0629889624 mean_abs=1.2444961 min=-123 max=21
```

The next integration seam from gemma.cpp into this backend is now present:

```text
experimental/openpi_image_encoder_image_adapter.h
experimental/openpi_image_encoder_image_adapter.cc
```

This adapter intentionally does only one thing: copy a `gcpp::Image` into the
OpenPI image encoder input tensor after checking that the image is already
`224 x 224 x 3`. It does not resize or normalize. That is important because
`gcpp::Image::ReadPPM` and `gcpp::Image::Set` already define gemma.cpp's image
normalization behavior, while the OpenPI raw-bundle backend expects the exact
float HWC tensor consumed by patch embedding.

Both HIP build scripts now compile and link this adapter object. A rebuild plus
hybrid smoke/golden validation passed:

```text
scripts/build_openpi_image_encoder_hip_probe.sh
scripts/build_openpi_image_encoder_hip_run.sh

build/openpi_image_encoder_hip_run --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle --attention hybrid --runs 3 --warmup 1
openpi_image_encoder_hip_run attention=hybrid image=<bundle> tokens=524288 encoded=294912
tokens_summary sum=-33024.3571 mean=-0.0629889624 mean_abs=1.2444961 min=-123 max=21

build/openpi_image_encoder_hip_probe --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle --attention hybrid --strict 1 --runs 3 --warmup 1
hip_patch_embed_stem_vs_golden max_abs=5.24520874e-06 rms=2.6804308e-07 mean_abs=1.70283213e-07
hip_patch_embed_with_posemb_vs_golden max_abs=7.62939453e-06 rms=2.69236309e-07 mean_abs=1.70298267e-07
hip_all_layers_encoded_vs_golden max_abs=1.3125 rms=0.0241603118 mean_abs=0.0157843356
hip_all_layers_tokens_vs_golden max_abs=2 rms=0.029132315 mean_abs=0.0181650731
```

The run-only binary has also been wired to a real gemma.cpp image-loading path:

```bash
build/openpi_image_encoder_hip_run \
  --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle \
  --image_ppm IMAGE.ppm \
  --attention hybrid --runs 3 --warmup 1
```

`--image_ppm` uses `gcpp::Image::ReadPPM`, then
`CopyImageToOpenPiImageEncoderTensor(...)`, then `RunFromImage(...)`. The run
build script now links the local Highway static library because
`paligemma/image.cc` pulls in `hwy::Abort`, `hwy::Warn`, and
`hwy::Profiler::Get`.

A synthetic 224x224 black PPM smoke image was generated under
`/home/amd/gemma.cpp/build/openpi_image_encoder_smoke/black_224.ppm` and ran
through this path:

```text
openpi_image_encoder_hip_run attention=hybrid image=build/openpi_image_encoder_smoke/black_224.ppm tokens=524288 encoded=294912
encoded_summary sum=-1052.10687 mean=-0.00356752817 mean_abs=0.489908064 min=-24.375 max=29.875
tokens_summary sum=-35648.9803 mean=-0.0679950338 mean_abs=1.24647694 min=-131 max=24.25
```

The existing raw-bundle path still produced the previous token summary after
this change:

```text
openpi_image_encoder_hip_run attention=hybrid image=<bundle> tokens=524288 encoded=294912
tokens_summary sum=-33024.3571 mean=-0.0629889624 mean_abs=1.2444961 min=-123 max=21
```

The same run-only binary can now export tensors for OpenPI-side consumption:

```bash
build/openpi_image_encoder_hip_run \
  --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle \
  --attention hybrid --runs 3 --warmup 1 \
  --tokens_out /home/amd/gemma.cpp/build/openpi_image_encoder_smoke/hip_tokens.f32 \
  --encoded_out /home/amd/gemma.cpp/build/openpi_image_encoder_smoke/hip_encoded.f32
```

The exported file sizes match the expected tensor sizes:

```text
hip_encoded.f32 1179648 bytes  # 294912 float32 values
hip_tokens.f32 2097152 bytes   # 524288 float32 values
```

A standard-library Python comparison against the raw bundle's golden files
matched the probe metrics:

```text
encoded shape 294912 max_abs 1.3125 rms 0.02416031184947769 mean_abs 0.015784335554965563
tokens shape 524288 max_abs 2.0 rms 0.029132315020924643 mean_abs 0.018165073143364907
```

This gives `openpi-Phi` a practical near-term bridge: call the gfx1150
`gemma.cpp` binary as a local image-token accelerator, read the F32 token file,
and compare it against JAX/OpenPI output before replacing any in-process model
path.

An `openpi-Phi` side bridge verifier now automates that CLI path:

```text
src/openpi/shared/gemma_cpp_image_encoder_bridge.py
scripts/run_gemma_cpp_image_encoder_bridge.py
src/openpi/shared/gemma_cpp_image_encoder_bridge_test.py
```

The reusable module uses only Python standard-library modules, so it does not
require the JAX or ROCm Python environment. The CLI wrapper imports that module
and keeps the same command surface. The default command runs the gemma.cpp
binary, writes F32 tensors under `build/openpi_golden/gemma_cpp_bridge_smoke/`,
and compares them against the raw bundle golden tensors:

```bash
scripts/run_gemma_cpp_image_encoder_bridge.py --strict
```

It can also rebuild the gemma.cpp run binary before executing:

```bash
scripts/run_gemma_cpp_image_encoder_bridge.py --build --strict
```

Validated output:

```text
/home/amd/gemma.cpp/scripts/build_openpi_image_encoder_hip_run.sh
Built /home/amd/gemma.cpp/build/openpi_image_encoder_hip_run
tokens_out build/openpi_golden/gemma_cpp_bridge_smoke/gemma_cpp_tokens.f32 bytes=2097152
encoded_out build/openpi_golden/gemma_cpp_bridge_smoke/gemma_cpp_encoded.f32 bytes=1179648
encoded count=294912 max_abs=1.3125 rms=0.0241603118 mean_abs=0.0157843356
tokens count=524288 max_abs=2 rms=0.029132315 mean_abs=0.0181650731
manifest build/openpi_golden/gemma_cpp_bridge_smoke/bridge_run_manifest.json
```

This is the first end-to-end local bridge owned from the `openpi-Phi` tree: it
drives the `gemma.cpp` HIP image encoder on `gfx1150`, persists the generated
tokens, writes a machine-readable run manifest, and enforces the current strict
compatibility envelope.

The bridge module now also records and validates the tensor shape contract:

```text
encoded_dtype = float32_le
encoded_shape = [1, 256, 1152]
tokens_dtype = float32_le
tokens_shape = [1, 256, 2048]
```

That matches the OpenPI injection point in `Pi0.embed_prefix()`: for each
camera, OpenPI currently calls `self.PaliGemma.img(...)` to get
`[batch, 256, 2048]` image tokens, appends those tokens to the prefix sequence,
and creates an image mask of length 256. The current `gemma.cpp` bridge
therefore represents one camera view. Multi-camera ALOHA/Libero/DROID prefixes
should call the bridge once per valid camera view and concatenate the resulting
`[1, 256, 2048]` token blocks in the same image-key order used by
`observation.images`. The on-disk container is float32 little-endian; when the
tokens are injected into the JAX prefix path, they should be cast to the model's
embedding dtype before concatenating with text embeddings to avoid accidental
float32 promotion.

For the later in-process bridge, the shared module exposes lazy-numpy helpers:

```python
from openpi.shared import gemma_cpp_image_encoder_bridge as bridge

tokens = bridge.load_tokens_numpy(path, dtype=target_embed_dtype)
encoded = bridge.load_encoded_numpy(path)
```

The module itself still imports only standard-library dependencies until these
helpers are called, so the CLI verifier remains runnable from the system Python.

The existing policy breakdown benchmark now has an optional external-token
path:

```bash
scripts/benchmark_policy_breakdown.py \
  --external-image-tokens build/openpi_golden/gemma_cpp_bridge_smoke/gemma_cpp_tokens.f32
```

When this argument is set, the benchmark bypasses `self.PaliGemma.img(...)`,
loads the bridge token block, casts it to `train_config.model.dtype`, repeats it
for each camera view in `observation.images`, and continues through text-prefix,
prefix-LLM, and suffix-loop timing. This does not change production model
behavior; it is a measurement hook for estimating how much latency remains
after replacing OpenPI's JAX image encoder with the `gemma.cpp` image-token
path.

This benchmark path was syntax-checked locally. A direct Docker smoke attempt
with `openpi-jax-rocm-gfx1150:latest` did not reach model execution because the
container's bare `python` does not have `flax` installed; the executable bridge
validation remains `scripts/run_gemma_cpp_image_encoder_bridge.py --strict`.

The Docker issue was resolved: the image relies on pyenv and the mounted repo's
`.python-version` requests Python 3.11, while the container has Python 3.12.8.
The benchmark also needs `OPENPI_DATA_HOME` pointed at the mounted host cache so
the cached `big_vision/paligemma_tokenizer.model` is used instead of attempting
an anonymous GCS download. This is now captured in:

```text
scripts/docker/run_policy_breakdown_gfx1150.sh
```

Validated command:

```bash
scripts/docker/run_policy_breakdown_gfx1150.sh \
  --runs 1 --warmup 0 \
  --external-image-tokens build/openpi_golden/gemma_cpp_bridge_smoke/gemma_cpp_tokens.f32
```

External-token smoke result:

```text
image_embed_source build/openpi_golden/gemma_cpp_bridge_smoke/gemma_cpp_tokens.f32
image_embed_avg_sec 0.005829476023791358
text_prefix_avg_sec 0.005692410020856187
prefix_llm_avg_sec 0.6844058879942168
prefix_total_composed_avg_sec 0.6959277740388643
suffix_loop_avg_sec 0.3913004400092177
total_policy_infer_avg_sec 1.2956330589950085
```

Same-container one-run baseline without external tokens:

```text
image_embed_source openpi_jax
image_embed_avg_sec 0.17604968100204132
text_prefix_avg_sec 0.006716619012877345
prefix_llm_avg_sec 0.7179646269942168
prefix_total_composed_avg_sec 0.9007309270091355
suffix_loop_avg_sec 0.389637924003182
total_policy_infer_avg_sec 1.3326701800106093
```

Interpretation: replacing the JAX image encoder with precomputed external
tokens removes roughly 170 ms from the image-token segment in this one-run
smoke and reduces the measured composed prefix path by roughly 205 ms. The
full `policy.infer` timing does not yet include a real per-camera gemma.cpp
subprocess call, so it should not be read as final end-to-end acceleration.
The main remaining runtime bottleneck after image-token replacement is the
prefix LLM prefill, followed by the action suffix loop.

A pseudo-camera image smoke was also run to test the more realistic data path:

```text
build/openpi_golden/pseudo_camera/pseudo_gradient_224.ppm
```

This is a deterministic 224x224 RGB PPM with horizontal/vertical gradients and
a checker pattern. It was passed through the gemma.cpp PPM path:

```bash
scripts/run_gemma_cpp_image_encoder_bridge.py \
  --output-dir build/openpi_golden/gemma_cpp_pseudo_camera_smoke \
  --image-ppm build/openpi_golden/pseudo_camera/pseudo_gradient_224.ppm \
  --skip-compare --runs 3 --warmup 1
```

The pseudo-image bridge produced valid OpenPI-shaped tensors:

```text
openpi_image_encoder_hip_run attention=hybrid image=build/openpi_golden/pseudo_camera/pseudo_gradient_224.ppm tokens=524288 encoded=294912
tokens_summary sum=-31855.0992 mean=-0.06075878 mean_abs=1.32009263 min=-124 max=22.125
hip_image_encoder_timing_ms patch=2.66164 ln=3.19988 qkv=56.4617 attention=84.4496 sa=15.2988 mlp_up=57.9627 gelu=3.35671 mlp_down=57.6466 final_norm=0.045325 head=0.709904
tokens_out build/openpi_golden/gemma_cpp_pseudo_camera_smoke/gemma_cpp_tokens.f32 bytes=2097152
encoded_out build/openpi_golden/gemma_cpp_pseudo_camera_smoke/gemma_cpp_encoded.f32 bytes=1179648
pseudo_bridge_wall_sec 1.24
```

Those pseudo-image tokens were then consumed by the OpenPI external-token
benchmark:

```bash
scripts/docker/run_policy_breakdown_gfx1150.sh \
  --runs 1 --warmup 0 \
  --external-image-tokens build/openpi_golden/gemma_cpp_pseudo_camera_smoke/gemma_cpp_tokens.f32
```

Result:

```text
image_embed_source build/openpi_golden/gemma_cpp_pseudo_camera_smoke/gemma_cpp_tokens.f32
image_embed_avg_sec 0.005241706006927416
text_prefix_avg_sec 0.005167986004380509
prefix_llm_avg_sec 0.682617627986474
prefix_total_composed_avg_sec 0.6930273199977819
suffix_loop_avg_sec 0.391620766982669
total_policy_infer_avg_sec 1.2898744689882733
```

This confirms the pseudo-camera plumbing works end to end:
`PPM image -> gemma.cpp HIP image encoder -> F32 image tokens -> OpenPI prefix
benchmark`. It also exposes an important deployment constraint: the subprocess
bridge has about 1.24 seconds of wall time in this smoke, even though the
resident HIP kernel timings are much smaller. Therefore a production
integration should not spawn gemma.cpp per camera frame; it should use a
resident process, an in-process C++/Python extension, or another API that keeps
weights and ROCm state loaded across calls.

### Resident gemma.cpp Image Encoder Server

The first resident-process version is now implemented in `~/gemma.cpp`:

```text
experimental/openpi_image_encoder_hip_server.cc
scripts/build_openpi_image_encoder_hip_server.sh
```

The server loads the raw bundle and constructs `OpenPiImageEncoderHipBackend`
once at startup. It then reads line-oriented commands from stdin and writes one
line responses to stdout. Supported commands:

```text
PING
RUN image_ppm=PATH tokens_out=PATH encoded_out=PATH [warmup=N] [runs=N]
RUN image_f32=PATH tokens_out=PATH encoded_out=PATH [warmup=N] [runs=N]
QUIT
```

Startup example:

```bash
/home/amd/gemma.cpp/build/openpi_image_encoder_hip_server \
  --bundle build/openpi_golden/pi05_base_image_encoder_raw_bundle \
  --attention hybrid --warmup 0 --runs 1
```

`openpi-Phi` now has a reusable Python client and smoke wrapper:

```text
src/openpi/shared/gemma_cpp_image_encoder_bridge.py
scripts/run_gemma_cpp_image_encoder_server_smoke.py
```

Validated pseudo-camera smoke:

```bash
scripts/run_gemma_cpp_image_encoder_server_smoke.py \
  --image-ppm build/openpi_golden/pseudo_camera/pseudo_gradient_224.ppm \
  --output-dir build/openpi_golden/gemma_cpp_pseudo_camera_server_smoke \
  --requests 2
```

Result:

```text
request=1 wall_ms=301.057 tokens_out=build/openpi_golden/gemma_cpp_pseudo_camera_server_smoke/tokens_1.f32
request=2 wall_ms=270.245 tokens_out=build/openpi_golden/gemma_cpp_pseudo_camera_server_smoke/tokens_2.f32
tokens_identical=yes
python_resident_smoke_wall_sec 1.48
```

This closes the biggest architectural gap in the previous bridge. The old
subprocess smoke paid about 1.24 seconds for a single image because process
startup, raw-bundle loading, backend construction, and ROCm initialization were
inside every call. The resident server still pays startup once, but subsequent
requests reuse loaded weights and ROCm state; the measured per-request path for
the pseudo camera image is now roughly 270 ms in this smoke. That is still not
the final optimized production path, but it is the right integration shape for
multi-frame policy inference and a much better basis than one subprocess per
camera frame.

### VectorWare GPU Async/Await Assessment

Reference:

```text
https://www.vectorware.com/blog/async-await-on-gpu/
```

The VectorWare post demonstrates Rust `async`/`await` running on the GPU. Its
main idea is not host-side async I/O; it is using Rust futures as
compiler-generated state machines that can express structured concurrency
inside GPU code. The article frames this alongside warp specialization,
JAX/Triton/CUDA Tile style dependency graphs, and possible GPU-native
executors.

For this OpenPI/gemma.cpp integration, the post is useful as a design reference
but not as the immediate fix for the current performance gap:

```text
OpenPI/JAX image_embed baseline:          ~176 ms
gemma.cpp resident pseudo-camera request: ~270 ms
target for net benefit:                   <176 ms
```

Why it is not the direct next move:

- The post targets Rust/NVPTX-style GPU programming, while the current local
  implementation is C++ HIP/rocBLAS on AMD `gfx1150`.
- The current workload is a fixed dense ViT image encoder, not a dynamic
  irregular workload that obviously needs a GPU task executor.
- The measured resident request still includes non-kernel overheads such as PPM
  file read/parse, host-side request handling, device-to-host copies, and
  writing F32 token files.
- Introducing an executor-like abstraction could add register pressure,
  occupancy loss, and polling overhead; the post itself calls out these
  downsides.

Where it is relevant:

- Warp-specialized/tiled attention is a plausible future direction for the
  OpenPI image encoder attention kernel.
- Explicit work/data dependencies are the right mental model for fusing or
  scheduling patch embed, QKV, attention, MLP, and output projection.
- The same structured-concurrency idea also maps to the host integration:
  resident process, persistent buffers, batched camera requests, and a fixed
  execution graph are better than one subprocess per frame.

Immediate engineering priorities before any GPU async/executor experiment:

1. Split resident request timing into image input, GPU compute, D2H copy, and
   output serialization.
2. Add a tokens-only server mode so `encoded` is not copied or written on the
   production path.
3. Replace file-based token output with pipe/socket/shared-memory transfer.
4. Batch multiple camera views in one request so ALOHA's three views do not pay
   per-camera request overhead.
5. Explore HIP Graph capture for the fixed image-encoder execution sequence.
6. Continue optimizing the strict-compatible attention path; this is still the
   largest single kernel-side lever.

Conclusion: the VectorWare post is most useful as medium-term inspiration for
kernel scheduling and structured GPU concurrency. It does not change the next
practical step. The near-term path to push resident server latency below the
OpenPI/JAX image-encoder baseline is to remove file/protocol overhead, avoid
unnecessary outputs, batch cameras, and then revisit attention/tiled-kernel
design.

The bridge code now has a small module-level test covering F32 comparison
metrics and strict tolerance error handling. The local system Python does not
have `pytest` installed, so the pytest file was syntax-checked and the same
assertions were run directly with standard-library Python:

```text
manual bridge module checks passed
```

The probe now exposes an experimental attention selector:

```bash
--attention serial
--attention hybrid
--attention parallel
```

`serial` is the default and remains the most conservative correctness path. A
resident serial rerun with `--runs 20 --warmup 3` produced:

```text
hip_all_layers_timing_ms
ln=2.89072
qkv=46.8257
attention=597.536
sa=16.1216
mlp_up=56.9189
gelu=2.90256
mlp_down=56.387
final_norm=0.039395
head=0.814521
```

The new `hybrid` mode keeps the numerically sensitive softmax max/sum and BF16
probability conversion in the same serial order as the reference, but computes
the logits across tokens in parallel and computes attention-value output
dimensions in parallel. This preserved the strict envelope while cutting full
attention time substantially:

```text
build/openpi_image_encoder_hip_probe --bundle /home/amd/openpi-Phi/build/openpi_golden/pi05_base_image_encoder_raw_bundle --attention hybrid --strict 1 --runs 20 --warmup 3
hip_block00_attention_vs_cpu_reference max_abs=0.0078125 rms=8.72262315e-05 mean_abs=4.25100345e-06
hip_all_layers_encoded_vs_golden max_abs=1.8125 rms=0.023977572 mean_abs=0.0157405035
hip_all_layers_tokens_vs_golden max_abs=2 rms=0.0288557254 mean_abs=0.0180431094
hip_all_layers_timing_ms ln=3.69165 qkv=55.2735 attention=103.942 sa=16.4099 mlp_up=61.7062 gelu=4.8722 mlp_down=60.2815 final_norm=0.084159 head=1.11113
```

The fully `parallel` attention mode is faster again but not yet inside the
numerical envelope. A non-strict A/B run with
`--attention parallel --runs 5 --warmup 1` produced:

```text
hip_all_layers_encoded_vs_golden max_abs=8.8125 rms=0.131365459 mean_abs=0.0809646289
hip_all_layers_tokens_vs_golden max_abs=9.5 rms=0.151700971 mean_abs=0.0903059457
hip_all_layers_timing_ms ln=3.74522 qkv=52.5786 attention=70.4016 sa=16.8277 mlp_up=61.9628 gelu=3.81074 mlp_down=63.0186 final_norm=0.049333 head=0.692624
```

This confirms that attention optimization is the dominant remaining lever. The
hybrid kernel is the first strict-compatible speedup; the fully parallel
reduction still changes softmax accumulation enough to miss the current
OpenPI-compatible tolerance and should remain experimental until it is retuned
against the serial/hybrid reference.

The full path is numerically inside the existing C++ CPU reference envelope:
the CPU full reference saw image-token `max_abs=2.5`, while this HIP raw-bundle
path saw `max_abs=2.0`.

The next HIP step is no longer correctness discovery for this boundary; it is
engineering the validated path into a usable backend. The two immediate pieces
are:

- optimize the strict-compatible hybrid attention kernel further, especially
  the remaining serial softmax section;
- promote the current probe-level backend wrapper into a reusable backend class
  with a narrow image-token API.

### gemma.cpp HIP Microbench Baseline

The local `~/gemma.cpp` build already contains:

```text
build/paligemma2_vit_hip_bench
build/paligemma2_vit_hip_backend_probe
build/gemma_paligemma2_vit_hip
build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs
```

The generic HIP microbench runs on the host ROCm path and detects the native
device as `gfx1150`:

```bash
build/paligemma2_vit_hip_bench --samples 3 --warmup 1 --iters 1
```

Selected 224px results:

| Operation | Best ms | Median ms |
| --- | ---: | ---: |
| patch embed | 0.175 | 0.176 |
| QKV projection | 0.333 | 0.354 |
| attention output projection | 0.175 | 0.176 |
| MLP up | 0.485 | 0.488 |
| MLP down | 0.589 | 0.614 |
| PaliGemma2 3B head projection | 0.295 | 0.325 |
| 16-head QK, f32 | 0.334 | 0.336 |
| 16-head AV, f32 | 0.442 | 0.445 |

Important caveat: this is an operation-level synthetic benchmark using the
current PaliGemma2 `.sbs` shape family. It does not yet prove OpenPI numerical
equivalence and it does not include all full-encoder overheads. Its value here
is to show that the local HIP kernels are available and fast enough to justify
building the OpenPI-aware equivalence path.

An attempted full `paligemma2_vit_hip_backend_probe` run against the local
PaliGemma2 224 `.sbs` exited early without a useful diagnostic in this session.
That should be debugged separately, but it does not block the OpenPI-side golden
artifact work because the next hard requirement is OpenPI weight/layout export
and C++ ingestion, not generation smoke output.

## Summary

The two codebases can be combined, but the correct boundary is narrow.

`openpi-Phi` owns robot policy semantics and action generation. `gemma.cpp`
owns an increasingly optimized `gfx1150` HIP path for PaliGemma2-style ViT
image-token generation. The highest-leverage integration is to reuse the latter
as an optional image-prefix accelerator inside the former, after an offline
shape and numerical equivalence harness proves the boundary is valid.

## Branching and Publish Plan

The work should be published as two independent branches because the two
codebases have different ownership and build surfaces:

| Repository | Base | New branch | Scope |
| --- | --- | --- | --- |
| `phi-media-lab/openpi-Phi` | `main` | `gemma-cpp-image-token-bridge-gfx1150` | Python bridge, benchmark hooks, golden/export utilities, Docker runner, integration documentation |
| `phi-media-lab/gemma.cpp-Phi` | `paligemma2-vit-hip-gfx1150` | `openpi-hip-server` | OpenPI raw-bundle loader, HIP image encoder backend, pseudo-image adapter, resident server, build scripts |

The repositories do not have a source-level dependency on each other in this
stage. The dependency is an executable/protocol boundary:

- `openpi-Phi` can run without `~/gemma.cpp`; the default path remains the JAX
  image encoder.
- The optional bridge expects a separately built `gemma.cpp-Phi` binary or
  resident server.
- `gemma.cpp-Phi` can build and run its probes using exported OpenPI raw-bundle
  artifacts, but it does not import Python or link OpenPI code.

This split keeps each PR reviewable on its own. The OpenPI PR documents the
integration contract and contains the Python-side consumer. The gemma.cpp PR
contains the producer implementation and can iterate on HIP kernels without
forcing OpenPI policy changes. Once both branches are pushed, the OpenPI branch
should reference the gemma.cpp branch as the optional backend required for the
external-token benchmark path.
