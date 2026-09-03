# DFlash2 overlay — GLM-5.3-Flash speculative decoding on DGX Spark (GB10 / SM121)

This is the exact build recipe for the **DFlash2** speculative-decoding layer this repo's
launcher runs. It is a port of upstream vLLM **PR #52816**
("[Spec Decode] DFlash2: local convolution + candidate selector") onto the SM121 GB10 image.

## Just want to run it? Pull the prebuilt image (no build needed)

```bash
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
```

That is the image the launcher in this repo points at. Done.

## Build the overlay yourself

The base image `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8` is prebuilt and public
(upstream vLLM for GB10/SM121 + the fp8-KV sparse-MLA patch — see `patch_v8_fp8.py` for
what that layer does). The overlay here adds DFlash2 on top:

```bash
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8
docker build -t vllm-glm53-flash:dflash2 -f Dockerfile .
```

The `Dockerfile` COPYs the drafter model, the spec-decode module, and 5 patches into the
image's vLLM tree, then runs them and asserts `DFlash2DraftModel` is registered.

## File manifest

| file | destination inside the image (`$VLLM = .../dist-packages/vllm`) |
|---|---|
| `qwen3_dflash2.py` | `$VLLM/model_executor/models/qwen3_dflash2.py` |
| `dflash2/` (`__init__.py`, `speculator.py`) | `$VLLM/v1/worker/gpu/spec_decode/dflash2/` |
| `patch_registry_and_select.py` | registers `DFlash2DraftModel` + top-k candidate selector |
| `patch_glm_aux_capture.py` | GLM aux-hidden-state capture for the drafter |
| `patch_kv_page_lcm2.py` | KV page-size LCM alignment for the drafter |
| `patch_glm5_drafter_group.py` | drafter group-size derivation (block-size math) |
| `patch_prefix_cache_draft_group.py` | prefix-cache repair for the drafter group (#13) |
| `sim_glm5_drafter.py` | standalone drafter simulation / sanity check |

Full port writeup: **NOTES.md** and **GLUE-NOTES.md**.

## Credit

Upstream DFlash2 is vLLM PR #52816 (incoai / Inco AI). The SM121 / GB10 kernels and the
port onto this image are 2Wild's.
