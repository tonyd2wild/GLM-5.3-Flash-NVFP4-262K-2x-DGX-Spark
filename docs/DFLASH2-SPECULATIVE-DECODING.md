# DFlash2 speculative decoding for GLM-5.3-Flash on DGX Spark (GB10 / SM121)

**First working DFlash2 deployment of GLM-5.3-Flash on GB10 hardware** (2026-08-28).
`incoai/GLM-5.3-Flash-DFlash2` is a block-diffusion drafter published for SGLang;
this is the vLLM route, on consumer Blackwell, at tensor-parallel 2.

## Result

| metric | MTP-4 (previous flagship) | **DFlash2 (this)** |
|---|---|---|
| C1 decode, code prompt, warm | 21.8 tok/s | **46.9 tok/s** |
| draft acceptance | ~0.55 (per-position 0.74/0.47/0.27/0.15) | **74.1 %** (420 of 567 drafted) |
| draft tokens per verify step | 4 | **7** (block size 8, minus the target's own token) |
| KV pool cost of the drafter | n/a | **~0** (parasitic slot-sharing, see below) |

**2.15× single-stream decode over the MTP-4 baseline on identical hardware and
context.** Concurrency curve: [BENCH-C1-C6-DFLASH2.md](BENCH-C1-C6-DFLASH2.md).

## What it takes (four patches, `overlay-dflash2/`)

vLLM at the fork point (`0.1.dev20051+g487ecf187`) ships DFlash**1** support but
predates DFlash2 (upstream PR #52816, merged 2026-08-21), and GLM-5.3 could not
feed a drafter at all. The overlay is a bind-mount/COPY layer over the day-0 image:

1. **`qwen3_dflash2.py` + `dflash2/` + `patch_registry_and_select.py`** — the
   upstream DFlash2 drafter model and speculator, ported across one week of API
   drift, plus registry/selection wiring (`DFlash2DraftModel`, `selector_rank`).
   Local adaptations are marked `# SM121-PORT`; notably `_dflash_layer_causal`
   must also read `is_causal` from inside `dflash_config` (GLM's drafter puts it
   there) and the gumbel helper is vendored rather than patched into the shared
   sampling path.
2. **`patch_glm_aux_capture.py`** — the target side. GLM-5.3 had no
   `SupportsEagle3`, so nothing could hand hidden states to a drafter. The patch
   adds the interface plus capture at the drafter's five tap layers, with the
   **mHC contraction** that GLM needs: its layers defer their final `hc_post` to
   the next layer's fused pre, so mid-stack capture must materialize
   `layer.hc_post(...)` first, then contract 4 streams → one 4096-wide tensor
   (`hc_contract` == `mean(dim=1)`, mirroring `deepseek_v4`'s eagle3 path), with
   `sp_all_gather` under sequence parallelism. The runner applies `+1` to
   `target_layer_ids`, so config `[5,14,24,33,42]` logs as `(6,15,25,34,43)`.
3. **`patch_glm5_drafter_group.py`** — the KV layout. **This is the hard one.**
4. **`patch_kv_page_lcm2.py`** — a documented no-op stub, kept so older build
   chains still pass (see "the generic path is a trap" below).

Launch flags (added to the standard TP2 recipe):

```
-v /var/tmp/models/GLM-5.3-Flash-DFlash2:/models/dflash2-draft:ro
--speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7}'
```

`num_speculative_tokens` must be `block_size - 1` = **7**; the drafter is trained
for a block of 8 and one position is the target's own verified token. Asking for
8 drafts a position the model never learned.

## The KV layout problem (nine boots)

GLM-5.3 uses a **custom KV grouping fast path** (`_get_kv_cache_groups_glm5_next`)
in which 34 KDA/mamba layers and 11 kpool tail caches *slot-share* the 11 MLA
layers' tensors. That function bails to the generic uniform-page path if any
non-mamba/non-tail spec is not exactly `MLAAttentionSpec` — and a drafter's five
`SlidingWindowSpec` layers do exactly that.

**The generic path cannot serve this model.** Its page sizes are mutually hostile
(MLA 512 B/token, indexer 33 B/token — the factor 11 divides nothing; KDA state
pages don't scale with block size at all), and every escape fails:

| attempt | outcome |
|---|---|
| stock growth-only unification | impossible for any `--block-size`: needs 33 \| 512 |
| LCM unification | 77.9 MB pages → **27.92 GiB for one 262K request** (~13× inflation) |
| rational block scaling | passes divisibility, drafter block 738 → no backend accepts it (all require multiples of 16) |
| candidate search + `--block-size 5120/2048` | `No common block size for 5280/4752` — sparse-MLA kernels want /32, indexer /64 |
| any uniform page at all | rescales the kpool tail's block away from its pool size → `assert tail_kv_cache.shape[2] == pool_size` in warmup |

**The fix is to never leave the fast path.** `patch_glm5_drafter_group.py`
partitions the drafter's `SlidingWindowSpec` layers out *before* the all-MLA
check, builds the GLM groups unchanged, and appends one drafter group that
slot-shares the MLA tensors exactly like mamba and the tail do:

- the drafter's block is rescaled so its **real page equals the MLA page**
  (`mla_page // drafter_bytes_per_token`, gated on 16-alignment and mutual
  divisibility with the MLA block) — `page_size_padded` stays `None`;
- so the runner takes the **contiguous** reshape, not the padded strided view;
- drafter layer *i* rides MLA tensor *i* at disjoint block ids.

Per-block pool cost is **bit-identical to the drafterless model** — measured and
simulated. The drafter's only KV demand is its sliding-window block ids (~6 per
request).

### The padded-view trap (boot 8, worth knowing)

An earlier version set `page_size_padded` unconditionally. Even when the fit was
exact, that routed the runner into the strided-view path — and FlashInfer's SWA
backend registers *int* kernel block sizes, so a 2304-token manager block was
split ×36 into 64-token kernel blocks, each charged the **full page stride**:

```
setStorage: sizes [5760, 4, 64, 256], strides [2359296, 16384, 256, 1] ...
requiring a storage size of 13587251200 are out of bounds for storage of size 377487360
```

13.59 GB demanded from a 377 MB tensor. A padded stride is inexpressible as one
flat view under kernel-block splitting. Never pad a drafter spec; make it fit.

## Verification

`overlay-dflash2/sim_glm5_drafter.py` builds the real spec geometry (34 mamba + 11 MLA +
11 indexer + 11 tail + 5 drafter SWA), forms groups, emulates tensor binding and
`_reshape_kv_cache` **including the kernel-block split**, and asserts required
storage ≤ bound tensor. It reproduces the boot-8 overrun exactly and passes with
the fix — run it before any boot.

Healthy runtime signatures:

- `Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)`
- `Warming up spec-decode rejection sampler kernels (vocab=154880, num_spec=7, ...)`
- acceptance ratio from `/metrics` (`spec_decode_num_accepted_tokens_total` ÷
  `..._num_draft_tokens_total`) around **0.6–0.8**. A ratio near 0.15 means the
  aux capture or the mHC contraction is wrong — it degrades silently, it does not
  crash.

## Operational notes

- The drafter warns `does not support external multimodal embeddings` — drafting
  is text-only; vision requests still work, they just aren't speculated.
- First inference JIT-compiles `_prepare_dflash_inputs_kernel` and
  `mhc_pre_big_fuse_with_norm_tilelang`; a cold C1 measurement reads ~10 tok/s
  low. Measure warm.
- Concurrency is bounded by free-memory headroom, not the pool: at
  `--kv-cache-memory 4445787956` a 3-way 20K-token prefill drove MemAvailable to
  3.06 GB and Tony's `dgx-anti-oom` watchdog (threshold 3 GB) killed the engine.
  Shipping pin is **6442450944** (678,661-token pool) for headroom under load.
  Changed in [#16](../../pull/16): the previous pin was **3221225472** (310,292-token pool),
  which measured 6 preemptions under load where 6 GiB measured 0. See
  [TP2-SPEC-DEPTH-AND-KV-2026-09-02](TP2-SPEC-DEPTH-AND-KV-2026-09-02.md).


## NVFP4-KV lane (partial)

The same overlay was ported onto the b12x NVFP4-KV stack (drowzeys' recipe: 4-bit
`nvfp4_ds_mla` MLA cache, 368 B/token). **It serves and drafts correctly** —
334,161-token pool, 35.9 tok/s on a 500-token code generation, acceptance 0.563
(per-position 0.77 / 0.64 / 0.59 / 0.52 / 0.46 / 0.39 / 0.36), mean acceptance
length ~4.9 — but it is **not production-ready**: any prompt long enough to require
chunked prefill (~>3K tokens) kills the rank-0 worker with no traceback, no CUDA
error and no OOM entry, on the first chunk (`num_computed_tokens=0`).

Two porting differences worth recording, both caused by `--kv-cache-dtype-skip-layers
sliding_window` (required — `nvfp4_ds_mla` is an MLA-record-only dtype, so the drafter's
KV must stay bf16):

1. **The skip-quant branch stamps the drafter's spec with `page_size_padded`** (from a
   generic 576 B/token estimate, not the real 368 B record) and shrinks its block to the
   backend's largest kernel block. That padded spec then trips the layout classifier's
   "a drafter group must never be padded" guard, the model falls to the generic
   uniform-page path, and `get_uniform_page_size` asserts. Fix: strip the inherited
   padding before any geometry math (no-op on the fp8 lane).
2. **Exact page fit is impossible on this stack** — `2,637,824 / 2048 = 1288 = 8 × 161`,
   not a multiple of any kernel block size — so the drafter falls to the standalone
   (own-tensor) path. That path had never run on real hardware, and it is the prime
   suspect for the chunked-prefill kill.

Ops note for anyone reproducing: on 121 GB nodes at TP2 this configuration leaves very
little headroom, and **swap must be enabled with `vm.swappiness=0`** — not disabled.
With swap off entirely the worker is killed during MoE marlin repack (no valve for the
spike); with swap on and default swappiness, the kernel pages vLLM out mid-load and
triggers a UVM driver livelock that freezes the shard loader at a reproducible point.
