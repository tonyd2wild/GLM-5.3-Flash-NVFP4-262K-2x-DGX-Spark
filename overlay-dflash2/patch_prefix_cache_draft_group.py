from pathlib import Path

# Prefix-cache repair for the DFlash2 draft group on hybrid GLM-5.3-Flash.
#
# Two changes in vllm/v1/core/kv_cache_coordinator.py:
#  1. When no KV cache group is flagged as an EAGLE group, flag only the DFlash
#     draft sliding-window group instead of every group. Flagging the target
#     full-attention, Mamba, and kpool groups makes the EAGLE last-block drop
#     apply to all of them.
#  2. In find_longest_cache_hit, never let the draft sliding-window group shrink
#     the hit length that the target and Mamba groups already agreed on. The
#     draft window is short, so its own hit is shorter than the target hit;
#     before this change that shorter value replaced curr_hit_length and every
#     later group re-verified against it, which ended in a zero-length hit.
#
# Measured on 2x DGX Spark (GB10), TP=2, DFlash2 width 7, block 2304, FP8 KV,
# 262,144-token context: prefix_cache hit rate 0.986 on the repeated 262k prompt
# (258,048 cached tokens = 112 aligned blocks); before the change the counter
# stayed at 0 hits (see issue #13).

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py")
s = p.read_text()

if "SlidingWindowSpec" not in s.split("class ", 1)[0]:
    raise SystemExit("SlidingWindowSpec is not imported in kv_cache_coordinator.py")

helpers_old = "logger = init_logger(__name__)\n\n\n"
helpers_new = (
    "logger = init_logger(__name__)\n\n\n\n"
    "def _glm53_inner_kv_spec(spec):\n"
    "    specs = getattr(spec, \"kv_cache_specs\", None)\n"
    "    if isinstance(specs, dict) and specs:\n"
    "        return next(iter(specs.values()))\n"
    "    return spec\n"
    "\n"
    "\n"
    "def _glm53_is_draft_swa_spec(spec) -> bool:\n"
    "    \"\"\"Return true only for the DFlash sliding-window cache group.\"\"\"\n"
    "    return isinstance(_glm53_inner_kv_spec(spec), SlidingWindowSpec)\n"
    "\n"
    "\n"
)
if s.count(helpers_old) != 1:
    raise SystemExit("helper anchor match count: %d" % s.count(helpers_old))
s = s.replace(helpers_old, helpers_new)

fallback_old = (
    "        if use_eagle and not self.eagle_group_ids:\n"
    "            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))\n"
)
fallback_new = (
    "        if use_eagle and not self.eagle_group_ids:\n"
    "            draft_swa_ids = {\n"
    "                index\n"
    "                for index, group in enumerate(kv_cache_config.kv_cache_groups)\n"
    "                if _glm53_is_draft_swa_spec(group.kv_cache_spec)\n"
    "            }\n"
    "            self.eagle_group_ids = draft_swa_ids or set(\n"
    "                range(len(kv_cache_config.kv_cache_groups))\n"
    "            )\n"
)
if s.count(fallback_old) != 1:
    raise SystemExit("fallback match count: %d" % s.count(fallback_old))
s = s.replace(fallback_old, fallback_new)

hit_old = (
    "                elif _new_hit_length < curr_hit_length:\n"
    "                    # length shrunk; invalidate previous eagle verifications\n"
    "                    eagle_verified.clear()\n"
    "                curr_hit_length = _new_hit_length\n"
)
hit_new = (
    "                elif _new_hit_length < curr_hit_length:\n"
    "                    # length shrunk; invalidate previous eagle verifications\n"
    "                    eagle_verified.clear()\n"
    "                if _glm53_is_draft_swa_spec(spec):\n"
    "                    # A short draft window must not reduce the target and Mamba\n"
    "                    # hit. Leave it empty so allocation creates fresh pages.\n"
    "                    if _new_hit_length >= curr_hit_length:\n"
    "                        for group_id, blocks in zip(group_ids, hit_blocks):\n"
    "                            hit_blocks_by_group[group_id] = blocks\n"
    "                            hit_length_by_group[group_id] = _new_hit_length\n"
    "                    continue\n"
    "                curr_hit_length = _new_hit_length\n"
)
if s.count(hit_old) != 1:
    raise SystemExit("hit-loop match count: %d" % s.count(hit_old))
s = s.replace(hit_old, hit_new)

p.write_text(s)
print("prefix-cache draft-group patch OK")
