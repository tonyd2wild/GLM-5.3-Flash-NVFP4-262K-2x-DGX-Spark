# Prefix cache with the DFlash2 drafter on SM121 (fix for #13)

**Result.** With one patch to `vllm/v1/core/kv_cache_coordinator.py` the prefix
cache hits on the TP2 DFlash2 recipe. Measured 2026-09-03 on 2x DGX Spark
(GB10), TP=2, DFlash2 width 7, `--block-size 2304`, fp8 KV, 262,144-token
context, one request at a time:

| metric | value |
|---|---:|
| repeated 262,144-token prompt, `prefix_cache` hit rate | 0.986 |
| cached tokens on the repeat | 258,048 (= 112 aligned blocks of 2304) |
| warm time to first token, p95 | 2.98 s |
| cold time to first token (261,632 prompt tokens) | 191.7 s |
| DFlash2 acceptance (2,229 of 2,401 drafts) | 0.928 |

Before the patch the counters on this recipe stayed at zero hits, the
signature reported in #13 (`prefix_cache_queries_total` growing,
`prefix_cache_hits_total` at 0, byte-identical and block-aligned prompts
included).

## Mechanism

The drafter port (`patch_glm5_drafter_group.py`) gives the DFlash2 draft its
own KV cache group with a short sliding window. That group is the one that
needs EAGLE-style handling (the last-block drop). In the image's vLLM tree
(`0.1.dev20051+g487ecf187`) no group is flagged as an EAGLE group for this
model, so `KVCacheCoordinator` falls back to flagging **every** group:
target full attention, the Mamba/linear-attention groups, the kpool tail, and
the draft window.

Two things then go wrong in `find_longest_cache_hit`:

1. The EAGLE last-block drop is applied to the target groups as well as the
   draft group, so every group re-verifies against a shorter hit.
2. The draft group's own hit is shorter than the target hit, because its
   window is short. The loop takes that shorter value as the new
   `curr_hit_length`, clears the EAGLE verifications, and the later groups
   verify against it. The chain ends at a zero-length hit, which is why even
   an exactly block-aligned repeat of the same prompt reports no hit.

`patch_prefix_cache_draft_group.py` changes exactly two places:

- When no group is flagged, flag only the group whose spec is a
  `SlidingWindowSpec` (the draft window). The fallback to all groups is kept
  for models without such a group.
- In the hit loop, a draft-window group must not shrink the hit that the
  target and Mamba groups already agreed on. When its hit is at least as long
  it is recorded; when it is shorter it is left empty so allocation creates
  fresh draft pages. Either way the loop continues with the target hit.

The patched file is byte-identical to the one we qualified in production.

## Relation to upstream

Upstream vLLM main has since replaced the spec-based flagging with an
explicit `is_eagle_group` flag on each KV cache group (the fallback to all
groups still exists). The kpool opt-out gating fix mentioned in #13 (commit
`36bb3795b` on the `glm-release` branch) addresses fine-grained
(partial-block) hits and is a separate change; the zero-hit result on aligned
prompts is the draft-group problem above. Upstream main does not carry the
GLM-5 drafter group, so this patch lives in the overlay.

## How to verify on your fleet

1. Rebuild the DFlash2 overlay image. Both overlay build recipes
   (`overlay-dflash2/Dockerfile` and `docker/dflash2-overlay/Dockerfile`) now
   run `patch_prefix_cache_draft_group.py`; the patch aborts if the anchor text
   is not found exactly once, and it checks that the draft-group predicate
   accepts the drafter's sliding-window group and rejects the kpool tail cache.
2. Send one long prompt twice with `temperature: 0` and read
   `curl :8000/metrics | grep prefix_cache`. The second request must add to
   `prefix_cache_hits_total`; with `--block-size 2304` the cached prefix is the
   prompt length rounded down to a multiple of 2304.
3. Watch `prompt_tokens_details.cached_tokens` in the completion response
   (`--enable-prompt-tokens-details`); it reports the same number.

## Related: long-context concurrency (#14)

With the prefix cache working, agent sessions stop re-prefilling the whole
conversation on every turn. Concurrency is a separate limit: two long-context
requests in decode still collapse on GB10 (#14), so our production recipe runs
`--max-num-seqs 1` at 262K context (4 GiB fp8 KV, 414,615 tokens of
distributed KV capacity); a second request queues instead of both crawling.
