# DFlash2 overlay for radixark/vllm-glm53-flash:sm121-v8

Port of vLLM upstream PR #52816 ("[Spec Decode] DFlash2: local convolution +
candidate selector", merged 2026-08-21, merge commit `b389ac294`, head
`3406ec1da`) onto the image's vLLM `0.1.dev20051+g487ecf187` (~Aug 15 tree,
plus local fork edits).

VLLM root inside the image: `/usr/local/lib/python3.12/dist-packages/vllm`
(call it `$VLLM` below).

## File manifest and COPY destinations

| overlay file                  | destination inside image                                    |
|-------------------------------|-------------------------------------------------------------|
| `qwen3_dflash2.py`            | `$VLLM/model_executor/models/qwen3_dflash2.py`              |
| `dflash2/__init__.py`         | `$VLLM/v1/worker/gpu/spec_decode/dflash2/__init__.py`       |
| `dflash2/speculator.py`       | `$VLLM/v1/worker/gpu/spec_decode/dflash2/speculator.py`     |
| `patch_registry_and_select.py`| run inside image AFTER the copies (edits 6 files in place)  |

Dockerfile sketch:

```dockerfile
FROM radixark/vllm-glm53-flash:sm121-v8
ARG VLLM=/usr/local/lib/python3.12/dist-packages/vllm
COPY qwen3_dflash2.py $VLLM/model_executor/models/qwen3_dflash2.py
COPY dflash2/ $VLLM/v1/worker/gpu/spec_decode/dflash2/
COPY patch_registry_and_select.py /opt/patches/
RUN python3 /opt/patches/patch_registry_and_select.py
# (plus patch_glm_aux_capture.py, produced separately)
```

The patch script is idempotent (safe to re-run), anchors every edit with
`assert`, and `ast.parse`s each file before writing. Optional argv[1]
overrides the vllm root (used for testing).

## What the patch script edits (all marked `# SM121-PORT`)

1. `model_executor/models/registry.py` — adds
   `"DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM")`
   right after the `DFlashDraftModel` entry (line ~630).
2. `v1/worker/gpu/spec_decode/__init__.py` — inside the `method == "dflash"`
   branch of `init_speculator`, routes to `DFlash2Speculator` when the draft
   declares architecture `DFlash2DraftModel` **or** its `dflash_config`
   carries `selector_rank` (extension over the PR, which checks only the
   architecture).
3. `model_executor/models/qwen3_dflash.py` — the PR's subclass hooks:
   `DFlashQwen3Model.decoder_layer_cls`, `DFlashQwen3ForCausalLM.model_cls`,
   and their use sites; plus the `_dflash_layer_causal` fix, **extended** to
   also read `dflash_config["is_causal"]` — the GLM53 drafter checkpoint puts
   `is_causal: false` inside `dflash_config`, which neither the image's
   `causal` key nor the PR's top-level `getattr(config, "is_causal")` would
   see.
4. `v1/worker/gpu/spec_decode/speculator.py` — `draft_logits_spec()` hook and
   the `torch.full` allocation in `DraftModelSpeculator.__init__`, verbatim
   PR. Needed so DFlash2's fp32/-inf sparse draft-logits cache takes effect
   under `draft_sample_method="probabilistic"`. Default behavior for every
   other speculator is unchanged (head_dtype, fill 0.0 — zeros as before).
5. `config/vllm.py` — `_is_dflash2_draft()` + force `use_v2_model_runner`
   for a DFlash2 draft (same signals as the selection in item 2), verbatim
   PR otherwise. Anchored after `_dflash_needs_multi_kv_group()`; the fork's
   local GLM5-Next `hc_mult` edits in the same file are untouched.
6. `model_executor/layers/logits_processor.py` — `_flashinfer_topk`, `_topk`,
   and `LogitsProcessor.get_top_k_tokens`, verbatim from the merged PR (the
   image's file is byte-identical to the PR base, so this is a pure
   addition). The image's flashinfer ships `top_k(input, k, sorted=...,
   deterministic=...)` with exactly the signature the PR calls — verified.

## Overlay-file adaptations vs upstream

- `qwen3_dflash2.py` — **verbatim** merged upstream file, no changes.
- `dflash2/__init__.py` — verbatim (license header only).
- `dflash2/speculator.py` — one adaptation: the import
  `from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax` is
  replaced by a local `@triton.jit gumbel_noised_argmax` copied verbatim from
  the merged `gumbel.py`, importing `tl_rand32`/`tl_rand64` from the image's
  `gumbel.py` and `tldevice` from `vllm.triton_utils`. Reason: PR #52816
  factored that helper out of `gumbel_block_argmax`; the image's `gumbel.py`
  predates the refactor, and it is shared with the production GLM/MTP
  sampling path, so it is left untouched. Same rand primitives means the
  draft walk and target verification draw identical noise.

## Drift checked between the image (g487ecf187 + fork edits) and merge b389ac294

- `dflash/speculator.py`: upstream added DCP/context-parallel plumbing and
  null-block guards after the image was built. `DFlash2Speculator` overrides
  only `_generate_draft` and `draft_logits_spec`; the image's
  `_generate_draft` signature matches the PR's exactly, and every attribute
  DFlash2 touches (`sample_pos`, `sample_idx_mapping`, `sample_indices`,
  `temperature`, `seeds`, `draft_tokens`, `use_fp64_gumbel`, `input_buffers`,
  `num_query_per_req`, `draft_logits`) exists with the same shape/meaning.
  The image fills `sample_idx_mapping` with -1 in `capture()` (upstream later
  moved that to the allocation), so padded rows are inert during CUDA graph
  capture as the DFlash2 kernels require.
- `spec_decode/speculator.py`: the fork carries local GLM5-Next `hc_mult`
  edits above the anchored region; anchors avoid them.
- `gumbel.py`: upstream also changed `logits_cache_stride` to two strides;
  for DFlash2's contiguous `[max_num_reqs, steps, vocab]` cache,
  `stride_1 == vocab_size`, so the image's single-stride indexing is
  equivalent. Not ported.
- `qwen3_dflash.py`: upstream additionally grew `is_neox_style` plumbing
  (`dflash_target_rope_is_neox_style`) and a mapper-based `load_weights`;
  both are orthogonal and not ported — the image's `skip_substrs`-based
  loader with the `"model." + name` prefix will route the new
  `layers.N.{attention_conv,mlp_conv}.*` and `candidate_selector.*` weights
  through `AutoWeightsLoader` into the matching module paths.
- Method detection: the image's `SpeculativeConfig` maps any draft model
  path containing "dflash" to `method="dflash"` (a "dflash2"-named path
  matches), and its `EAGLEConfig` leaves `DFlash2DraftModel` unrenamed
  (starts with "DFlash"). Passing `"method": "dflash"` explicitly in
  `--speculative-config` is still the safe move.

## Drafter checkpoint expectations (already on the nodes)

`architectures=["DFlash2DraftModel"]`, model_type qwen3, 5 layers, hidden
4096, vocab 154880, `dflash_config`: block_size 8, conv_kernel_size 2,
conv_group_size 16, mask_token_id 154856, selector_rank 256,
selector_top_k 16, target_layer_ids [5,14,24,33,42], is_causal false,
sliding_window 2048. No embed_tokens / lm_head (borrowed from target).

- `block_size 8` must equal `1 + num_speculative_tokens` → launch with
  `num_speculative_tokens: 7`.
- `conv_group_size 16` divides hidden 4096 → 256 groups; kernel_projection is
  4096 → 2*2*256.
- The selector codebooks are `[154880, 256]` fp16/bf16 ≈ 158 MB total —
  budget for it in the 24G/rank KV split.

## Remaining risks

1. **Aux hidden-state capture on the GLM target** (`target_layer_ids
   [5,14,24,33,42]`) is NOT handled here — that is
   `patch_glm_aux_capture.py`, produced by the other agent. Without it the
   drafter gets no target features and load/serve will fail or degrade.
2. **lm_head / embed_tokens borrowing**: the checkpoint ships neither; the
   image's existing DFlash target-sharing path must attach the GLM lm_head
   and embeddings to the draft. That path predates this port and worked for
   DFlash1-style heads; it has not been exercised with a 154880-vocab GLM
   target + qwen3-arch draft here. First real launch will prove it.
3. **CUDA graph capture of the selector**: the walk and cache kernels run
   eagerly per step outside the captured model graph, as upstream designed.
   The image's `DFlashCudaGraphManager` predates the PR; upstream did not
   change it for DFlash2, so capture should hold, but watch the first
   capture phase for shape/assert errors.
4. **`support_torch_compile` on `CandidateSelector`**: the image's decorator
   falls back to `get_current_vllm_config()` when the class takes no
   `vllm_config` — verified present — but compile-cache tagging
   (`set_model_tag("dflash2_candidate_selector")`) is exercised at load time;
   if the compile pass trips on SM121, drop the `@support_torch_compile`
   line in `qwen3_dflash2.py` (costs ~0.05 ms/step, eager selector).
5. **fp32 draft-logits cache size**: with `draft_sample_method=
   "probabilistic"`, the cache is `max_num_reqs * 7 * 154880 * 4` bytes
   (≈ 4.3 GB at max_num_reqs=1024, ≈ 0.4 GB at 96). Cap `--max-num-seqs`
   accordingly, or serve greedy drafts (no cache) first.
6. The verify-side rejection reads the cache through the image's
   `gumbel_block_argmax` (single-stride variant) — equivalent for the
   contiguous cache, but it is the one semantic seam between the two trees;
   if acceptance at T>0 looks broken while T=0 is fine, look here first.

## Verification performed

- `ast.parse` clean on all three overlay files and all six patched files.
- Patch script applied against a byte-exact mirror of the image's six files:
  all anchors found, unique, idempotent on re-run.
- In-container test (files copied into site-packages of a throwaway
  `--rm` container, patch script run for real):
  `import vllm.model_executor.models.registry`,
  `qwen3_dflash2`, `dflash2.speculator`, `spec_decode.__init__`,
  `config.vllm`, `LogitsProcessor.get_top_k_tokens` presence, and registry
  resolution of `DFlash2DraftModel` — see the session report.

## Prefix cache with the drafter group (`patch_prefix_cache_draft_group.py`)

The drafter group (a short sliding-window KV cache group) made every prefix
lookup end at zero length: with no group flagged as an EAGLE group the
coordinator flagged all of them, and the draft window then shrank the hit that
the target and Mamba groups had agreed on. The patch flags only the draft
window group and stops it from shrinking the target hit. Two exact-string edits
in `$VLLM/v1/core/kv_cache_coordinator.py`, run after
`patch_glm5_drafter_group.py`. Measured on 2x GB10 at 262K context: hit rate
0.986 on a repeated prompt, 258,048 cached tokens. Details and the verification
recipe: `docs/PREFIX-CACHE-DFLASH2-SM121.md` (issue #13).
