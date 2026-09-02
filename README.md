# GLM-5.3-Flash NVFP4 + DFlash2 on 2x NVIDIA DGX Spark

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B total / 18B active MoE) served by vLLM at **tensor-parallel 2 across two DGX Spark** (GB10/SM121), **262,144-token context**, fp8 KV, DFlash2 speculative drafter.

**Current recipe: [CURRENT.md](CURRENT.md).** Read that first — it is the one configuration this repo ships, and it wins over anything below that disagrees.

One launcher: [`launch-glm53-vllm-tp2-dflash2.sh`](launch-glm53-vllm-tp2-dflash2.sh) — **worker Spark4 (rank 1) FIRST, then head Reddie (rank 0)**, which serves :8000.

Weights: [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) (compressed-tensors) at `/var/tmp/models/GLM-5.3-Flash-NVFP4-redhat` — ModelOpt and abliterated NVFP4 quants corrupt tokens on this stack and the launcher refuses them.

Everything else here is reference: the bring-up log, the day-0 bug receipts, the benchmark history, and the open problems.

---

## ⭐ Checkpoint: `RedHatAI/GLM-5.3-Flash-NVFP4` is now the default (corruption fix)

ModelOpt-quantized NVFP4 builds of GLM-5.3-Flash (`LibertAIDAI/GLM-5.3-Flash-NVFP4` and the abliterated variants) emit **intermittent corrupted token IDs** ([vLLM #54150](https://github.com/vllm-project/vllm/issues/54150)). Nearly invisible in English, but when a corrupted token lands inside a tool-call block the parser desyncs and generation can spiral into a repetition lock.

We reproduced and fixed it on this exact cluster (Korean-Hangul probe, `temperature 0`, non-streaming, 3 passes):

| checkpoint | `quant_method` | U+FFFD count (3 runs) |
|---|---|---|
| ModelOpt NVFP4 (LibertAIDAI / keys-ablit) | `modelopt` | 4 / 9 / 8 |
| **[RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)** | **`compressed-tensors`** | **0 / 0 / 0** |

**Default checkpoint: `RedHatAI/GLM-5.3-Flash-NVFP4`.** Ungated, same `Glm5NextForConditionalGeneration` arch, **drop-in** — no flag changes (`--moe-backend marlin`, DFlash2 `k=7`, fp8 KV all identical), just repoint the model path. Loads ~2x faster (11 large shards vs 120 small). Tradeoff: it also quantizes activations to 4-bit (W4A4) where the weight-only builds are W4A16, so expect a few points lower on hard reasoning — but the output is **correct**. Make sure the vision `chat_template_mm.jinja` is present in the weights dir or image requests 500.

Corruption first flagged by [@ajclark](https://github.com/ajclark) (issue #10). Uncensored (abliterated) builds remain available but carry the ModelOpt corruption until a compressed-tensors abliteration exists.

## Weights: censored or uncensored (drop-in)

Pick your weights: **same launcher, same recipe**, just point the model path at either. Both are NVFP4 and load identically.

| | HuggingFace | notes |
|---|---|---|
| **⭐ Default (recommended)** | [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) | **compressed-tensors, corruption-free** (see fix above) |
| Censored (legacy) | [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) | stock NVFP4 weight-only — ⚠️ ModelOpt token corruption |
| **Uncensored (abliterated)** | [drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock) | abliterated (layers 15-45, anchor-stock), no refusals |

Uncensored abliteration credit: [drowzeys/keys](https://github.com/drowzeys).

Two firsts, as far as we can tell: the **first working GLM-5.3-Flash deployment on DGX Spark**
(seven day-0 bugs deep — [docs/DEPLOY-REPORT.md](docs/DEPLOY-REPORT.md)), and the **first
working DFlash2 deployment of this model on GB10** ([docs/DFLASH2-SPECULATIVE-DECODING.md](docs/DFLASH2-SPECULATIVE-DECODING.md)).

> 🔀 **Running all four Sparks?** The same images scale to TP4 with the model-native 1M context —
> see the sibling repo: **[GLM-5.3-Flash at TP4 · 1M KV · 4x DGX Spark →](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)**

---

## Quickstart

**1. Pull the images** (GHCR, public, anonymous — no build required):

```bash
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2   # with DFlash2 (recommended)
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8            # base, fp8 KV, no drafter
```

They contain **only vLLM + our patches** — no model weights (those bind-mount at runtime).
No retag step: the launchers reference these `ghcr.io/tonyd2wild/…` tags directly. (Only the
`docker/` build chain uses local `radixark/…` stage tags, and those never leave the build.)

**2. Fetch the weights** to the same path on both nodes (or NFS-export from the head):
[RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) (default) →
`/var/tmp/models/GLM-5.3-Flash-NVFP4-redhat` — this is the path the launcher checks
(`MODEL_HOST_PATH`); override with `MODEL_HOST_PATH=…` if you keep weights elsewhere. For
DFlash2, also fetch the drafter (2.2 GB) → `/var/tmp/models/GLM-5.3-Flash-DFlash2`.

**3. Install the SM121 top-k fix** on **both** nodes. Both published images still contain a
decode-time top-k kernel that hard-kills the engine on any decode past ~24K context on GB10
(the launcher bind-mounts the fix over it):

```bash
mkdir -p ~/patches
cp docker/sparse_attn_indexer_kpool_sm121.py ~/patches/sparse_attn_indexer_kpool.py
```

Why, and what the crash looks like: [docs/SM121-CRASH-FORENSICS](docs/SM121-CRASH-FORENSICS-2026-08-27.md).

**4. Edit the launcher for your fabric.** Set the IPs/paths at the top of
[`launch-glm53-vllm-tp2-dflash2.sh`](launch-glm53-vllm-tp2-dflash2.sh), **and the NCCL
interface names inside the `docker run` body** (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`,
`NCCL_IB_ADDR_RANGE`) — wrong NIC names fail silently. `ibdev2netdev` and `ip -br a` will
tell you what yours are.

Then launch — **pre-launch ritual on BOTH nodes, every time** (GB10 unified memory):

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches      # both nodes
./launch-glm53-vllm-tp2-dflash2.sh 1    # worker FIRST
sleep 25
./launch-glm53-vllm-tp2-dflash2.sh 0    # then head — serves :8000
```

Without the drafter, use [`launch-glm53-vllm-tp2.sh`](launch-glm53-vllm-tp2.sh) (v8 image,
MTP-4, ~21.8 tok/s) — same ritual.

**5. Smoke test.** Wait for readiness first (~15 min; the shard load dominates). Poll
`/health`, **never `/v1/models`** — that returns 200 even with a dead engine:

```bash
until curl -sf http://<head>:8000/health >/dev/null; do sleep 20; done
```

```bash
curl http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"2+2=?"}],
       "max_tokens":40,"chat_template_kwargs":{"enable_thinking":false}}'
```

Full serve args, NCCL fabric env, and the rationale for every flag:
[docs/DEPLOY-REPORT.md](docs/DEPLOY-REPORT.md).

<details>
<summary><b>Build the images yourself instead</b> (only if you want to modify the patches)</summary>

The base is the public day-0 image:

```bash
docker pull vllm/vllm-openai:glm53-flash-arm64-cu130     # or -x86_64- for x86
cd docker
for i in $(seq 1 9); do
  docker build -f "Dockerfile.glm53-sm121-v$i" -t "ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v$i" . || break
done
docker save ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8 | ssh <worker> docker load
```

The chain is mostly linear (v1→v3→v4→…→v9); **v2 is an optional NaN-debug branch off v1** that nothing else builds on. For DFlash2, build
[`overlay-dflash2/`](overlay-dflash2/) on top of v8 afterwards.

> **Resolved.** The `FROM` lines of v2–v6 once referenced the original ad-hoc tag names
> (`sm121-nope-mla`, `sm121-fi618`, `sm121-fi618-nccl`, `sm121-final`) rather than
> `sm121-v1…v5`, so the loop above did not work. [#5](../../pull/5) (thanks @ozskywalker)
> normalized the chain to v-numbered tags and added `docker/build.sh`; no manual retagging
> between stages is needed.

Published image digests, so you can tell whether a local build differs:
`sm121-v8` → `sha256:d77d375c742fc54f436dec5108b440f58f021bc6600052bf0e8fe5840357e78f` ·
`sm121-v11-dflash2` → `sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`
</details>

---

## Results (TP2, 2026-08-28)

**DFlash2 + fp8 KV — the shipped configuration.** All measured on our own hardware; the
harness is [`probes/bench_c1c6.py`](probes/bench_c1c6.py), run as:
`python3 probes/bench_c1c6.py --url http://<head>:8000 --rounds 2 --max-tokens 400`

| | value |
|---|---|
| Single-stream decode, code prompt, warm | **46.9 tok/s** at 74.1 % draft acceptance |
| Single-stream decode, structured output | **54–61 tok/s** (temp 0, 3 runs) |
| KV pool | **581,040 tokens** @ 262K context, profiler-sized (see the ceiling section) |
| Context | 262,144 |
| KV cost of the drafter | **zero** — it slot-shares the MLA tensors |
| Boot | ~15 min (shard load dominates) |

Concurrency sweep — 2 waves/level, 400-token generations, mixed code+prose, **zero failures**:

| | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| aggregate tok/s | 35.1 | 41.6 | 40.6 | 47.5 | **56.2** | 47.7 |
| per-stream tok/s | 35.1 | 23.2 | 17.3 | 15.3 | 17.5 | 13.3 |
| accepted ÷ drafted | 0.53 | 0.45 | 0.42 | 0.40 | 0.51 | 0.40 |

**The progression**, same fleet, same prompt style:

| config | decode | note |
|---|---|---|
| bf16, no speculation | 14.3 tok/s | |
| fp8 KV + MTP-4 | 21.8 tok/s | previous flagship |
| **fp8 KV + DFlash2** | **46.9 tok/s** | **2.15x** — this repo |
| fp8 KV + DFlash2 at TP4 | 68.5 tok/s | four nodes, 1M context — [sibling repo](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark) |

Throughput tracks how *predictable* the output is, not just the config: structured/list/
tool-argument output drafts at ~0.9 acceptance, freeform prose nearer 0.33. Agentic traffic
lives in the high-acceptance zone. Detail and how to read these:
[docs/BENCH-C1-C6-DFLASH2.md](docs/BENCH-C1-C6-DFLASH2.md).

### KV pool ceiling on TP2 (2026-08-28)

> **Superseded 2026-09-02 — the shipped launcher now pins `--kv-cache-memory 6442450944`
> (6 GiB).** The guidance below ("let the profiler size the pool, never pin") was written
> before we measured what the profiler-sized pool costs under load: at the 3 GiB pin
> @tmooch measured 6 preemptions under load and 0 at 6 GiB, with the pool going
> **310,292 → 678,661 tokens**. Preemption under load costs more than any tok/s figure and
> the memory is available, so 6 GiB was adopted as the default in
> [#16](../../pull/16). Measurement and reasoning:
> [docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md](docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md).
> The sharp edge below is still real — a pin removes the activation reservation — which is
> why the shipped pin is a *measured* one, validated under load, not a guess. Do not raise
> it without repeating that measurement.

The original 2026-08-28 finding, kept because the mechanism still matters:

**Let vLLM's profiler size the pool. Do not pin `--kv-cache-memory`.**

That was the lesson at the time, and it cost us a night of boots to learn. When you pass
`--kv-cache-memory`, vLLM still runs the profile pass but **never subtracts the measured
activation peak** (`gpu_worker.py:475-495`) — it hands you exactly the number you asked
for and `--gpu-memory-utilization` becomes dead. Allocation succeeds, warmup succeeds, a
short generation succeeds, and then the first long prompt has nowhere to put its
activations and the engine dies. We reproduced that failure at four different pins.

Profiler-sized figures, all at `--max-model-len 262144` so they are comparable:

| config | KV pool | verified |
|---|---|---|
| **DFlash2 + fp8 KV** | **581,040 tokens** | serving; survived a 28,818-token prompt with the engine healthy after |
| **No drafter, fp8 KV** | **965,166 tokens** | allocated and booted; long-prompt survival not yet confirmed |

**The DFlash2 drafter costs ~4.8 GiB of KV headroom** — far more than its 2.2 GiB of
weights — which is a real trade nobody had priced: roughly **+91 % decode speed for −40 %
pool**. Choose per workload.

Three traps worth knowing before you tune this yourself:

- **The reported pool inflates with context.** `GPU KV cache size` is
  `int(max_concurrency × max_model_len)` (`kv_cache_utils.py:2264`), so raising
  `--max-model-len` raises the headline number without adding a byte of memory. Only
  `blocks × block_size`, or bytes/token, compares honestly across configs — and it is why
  pool figures published at 900K or 1M context are not comparable to figures at 262K.
- **`Available KV cache memory` is logged by rank 0 only, but the pool is built from the
  minimum across ranks** (`kv_cache_utils.py:2554`). One of our boots logged 6.25 GiB and
  bound 2.29 GiB. Read that line on **every** rank before trusting it.
- **The TP worker rank profiles 4–5 GiB less KV headroom than the head**, reproducibly, on
  both of our node pairs, independent of the drafter, of NFS, and of which physical machine
  is which. That asymmetry caps the pool and we have not explained it; it looks like an
  upstream vLLM question rather than a configuration error.

Operational note: on these 121 GiB unified-memory nodes, `vm.swappiness=0` is mandatory
and **does not survive a reboot**. With swap active the kernel pages vLLM out mid-load and
triggers a UVM driver livelock — one worker thread spinning at 100 %+ CPU, `UVM GPU` kthread
hot, GPU reporting high utilization at idle wattage, shard loading frozen at a reproducible
point. It does not recover on its own.

> A previous revision of this section published a 727,583-token figure at a 7 GiB pin as
> the usable ceiling. That configuration is **unsafe** (pinned, so no activation headroom),
> and no log of that measurement survived — it was quoted from a terminal session rather
> than a captured file. It has been withdrawn rather than restated.

### Tuning notes

- **`temperature: 0` is free throughput** (+13–21 %). vLLM's rejection sampler does an exact
  top-1 match at temp 0 but a probabilistic ratio test above it — and since the draft method is
  greedy, the draft probability is pinned to 1, making the T>0 test strictly harder.
- **`enable_thinking: false` is also the faster setting** (+8 % acceptance) — reasoning traces
  are higher-entropy and draft worse. Caveat: with thinking off GLM emits untagged
  reasoning-prose into `content`, which some agent harnesses mis-parse; see the deploy report.
- **K=7 is the default, but it is a workload choice — it was swept.** Conditional per-position
  acceptance is nearly flat (0.93/0.89/0.84/0.81/0.79/0.59/0.94), so the last position still
  earns; the drafter's `block_size: 8` caps K at 7 anyway, and a lower K gives a prettier
  *ratio* and worse throughput on a single stream. **But the sweep on 2026-09-02
  ([docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md](docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md),
  "Speculative depth: k=7 below C4, k=5 above it") found it inverts under concurrency** —
  k=5 beats k=7 by +29.9% at C4 and +18.3% at C6, while losing on every single-stream prompt.
  That doc's result, quoted: "`num_speculative_tokens` is a **workload choice, not a
  default.** It inverts at C4… **The default stays k=7** (single-user and low-concurrency is
  the common case, and it is what the README's headline figures use). Serving deep
  concurrency should set k=5, or better, use a schedule with the crossover at C4:
  `"num_speculative_tokens_per_batch_size": [[1,3,7],[4,512,5]]`"

---

## Status: work in progress

This repo is an active bring-up log, not a finished product. Everything published here is
measured on our own hardware and dated — **if a number is not in a dated table, treat it as
unverified.**

| | state |
|---|---|
| **DFlash2 + fp8 KV, TP2** | ✅ **Proven.** Benchmarked C1–C6, zero failures. This is the config to copy. |
| **DFlash2 + fp8 KV, TP4 / 1M ctx** | ✅ Serving — 68.5 tok/s, 2,622,494-token pool. See the sibling repo. |
| **DFlash2 + NVFP4 KV, TP2** | ⚠️ **Partial.** Serves and drafts (35.9 tok/s, 0.563 acceptance, 334K pool), but prompts long enough to need chunked prefill (>~3K tokens) kill the rank-0 worker. Root cause open; the standalone drafter KV path is the suspect. [Details](docs/DFLASH2-SPECULATIVE-DECODING.md). |
| **InstantTensor fast load** | ⚠️ Experimental — 15x faster loads, unstable multi-node. See below. |

---

## Why this needs a patched image

The vLLM PR authors' day-0 image (`vllm/vllm-openai:glm53-flash-arm64-cu130`) works on B200.
On GB10/SM121 it fails five separate ways. Our derivative (`docker/Dockerfile.glm53-sm121*`,
applied in order) fixes those five, then adds a sixth patch that unlocks fp8 KV:

1. **NoPE MLA vs the SM12x sparse backend** — the only stock capability-12 sparse-attention
   backend requires the packed `fp8_ds_mla` layout, which hardcodes DeepSeek's `pe_dim=64`.
   GLM-5.3 is NoPE (`qk_rope_head_dim=0`) → assert death in warmup. Fix: extend vLLM's SM90
   NoPE sparse-MLA backend to SM121 with the FA2 path — probed on-GPU with the model's real
   shape before trusting it (`probes/probe_sm121_nope_mla.py`).
2. **FlashInfer 0.6.17 FA2 MLA NaN** — the FA2 scheduler produces NaN for 64–256-row batches
   on SM121 (bisect: `probes/probe_fa2_bisect.py`), and normal prompts land exactly there.
   Fix: FlashInfer **0.6.18 nightly**.
3. **The nightly's dependency sabotage** — it silently downgrades `nvidia-nccl-cu13` to 2.29.7
   (NCCL "internal error" on the Spark IB fabric; re-pin **2.30.7**) and skews
   `nvidia-cutlass-dsl` to a mixed 4.7.0/4.6.2 state (CuTeDSL warmup ICE; re-pin **4.6.2**).
   Audit transitive pins after ANY pip install in these images.
4. **PDL on unvalidated silicon** — vLLM enables Programmatic Dependent Launch for capability
   ≥ 9, including SM121, in the Triton kernels carrying KDA recurrent state. Gated off on SM12x.
5. **Indexer uninitialized top-k** — the kpool top-k destination was `torch.empty` and the
   kernels only guarantee the first `min(k, valid)` entries; short rows carried garbage pool
   ids → bogus token indices → NaN lottery. Fix: init to `-1` + clamp (`docker/patch_v7.py`).
6. **fp8 KV cache unlock** (v8, `docker/patch_v8_fp8.py`) — see below.

Two serve-flag landmines, no code needed:

- **`--block-size 2304`** — vLLM's hybrid block aligner picks a size whose kpool storage tiles
  by 32, but DeepGEMM's arch-12 fp8 paged-MQA accepts only 64-entry pool pages. 2304 is a
  multiple of kpool·64 and of the MLA 128 alignment.
- **`--gpu-memory-utilization 0.85`** — 0.78–0.80 starve the KV cache at 131K+.
  (Credit: barrydeen's independent recipe.)

### fp8 KV cache on GB10: a two-line fix (and, as far as we can tell, a first)

FlashInfer gates fp8 MLA KV to SM90, and naively relaxing the gate fails with CUDA "invalid
argument" (`probes/probe_fa2_fp8.py`). The real cause (`docker/patch_v8_fp8.py`): the fa2 fp8
branch **forces `CTA_TILE_KV=32`, a Hopper 228KB-smem assumption**. On GB10's ~101KB opt-in max
that over-requests shared memory (117,312 B > 101,376 B) at `cudaFuncSetAttribute`, before the
kernel ever launches. **Capping the tile instead of forcing it** (fp8 keeps TKV=16 on
100KB-class devices — 91,680 B, fits) makes fp8 KV work: verified on-GPU, rel-err ~0.005 vs an
fp32 reference, then end-to-end in production.

As far as we can tell this is the first fp8 KV cache for a NoPE-MLA model on any consumer
Blackwell part. Upstream-ready issue drafts with receipts:
`docs/issue-flashinfer-fp8-mla-sm121.md`, `docs/issue-vllm-nope-fp8-ds-mla.md`.

---

## Hard-won operational rules

- **Tear down BOTH ranks before relaunching either.** A rank that rendezvouses with a dying one
  hangs or dies confusingly.
- **`grep '^IMAGE' launch-*.sh` on BOTH nodes before every launch.** Two "mystery" garbage boots
  were a silent image-version mismatch between ranks. Copy whole files between nodes; never
  `sed` over ssh.
- **Capture `docker logs` before `docker rm -f`.**
- **Probe `/health` for liveness, never `/v1/models`** — the latter returns 200 from config alone, with a dead engine behind it.
- **Two consecutive unexplained deaths = stop and diagnose.** Never crash-loop.
- **Swap on with `vm.swappiness=0`** — not off. Fully disabled, the worker dies during MoE
  marlin repack with no valve; at default swappiness the kernel pages vLLM out mid-load and
  triggers a UVM driver livelock that freezes the shard loader.
- `max_tokens` includes reasoning tokens when thinking is on; disable per-request with
  `chat_template_kwargs: {"enable_thinking": false}`.

---

## Deeper reading

| doc | what's in it |
|---|---|
| [DEPLOY-REPORT](docs/DEPLOY-REPORT.md) | the seven day-0 bugs, root causes, receipts, every serve flag |
| [DFLASH2-SPECULATIVE-DECODING](docs/DFLASH2-SPECULATIVE-DECODING.md) | the drafter port: four patches, the KV-layout fix, nine boots of failure modes |
| [BENCH-C1-C6-DFLASH2](docs/BENCH-C1-C6-DFLASH2.md) | full concurrency tables and how to read them |
| [SM121-CRASH-FORENSICS](docs/SM121-CRASH-FORENSICS-2026-08-27.md) | why the fleet "randomly" died: a topk kernel bug and phantom KV backing |
| [GB10-KV-MEMORY-LADDER](docs/GB10-KV-MEMORY-LADDER.md) | why KV budgets above vLLM's suggestion die, and the driver-level mechanism |
| [KV-HUNT-672K-TP2-RECORD](docs/KV-HUNT-672K-TP2-RECORD.md) | the 8-attempt hunt past the 507K wall |
| **[OPEN-PROBLEMS](docs/OPEN-PROBLEMS.md)** | **everything we broke and could not fix — reproducible, with next probes. Start here if you want to contribute.** |

**Debugging kit** (reusable for any day-0 model on new silicon): `probes/probe_sm121_nope_mla.py`
(probe a kernel with your real geometry before patching arch gates) · `probes/probe_fa2_bisect.py`
(NaN bisect over batch shapes) · `probes/probe_mhc.py` (A/B a Triton kernel vs its torch
reference) · `probes/gb10_alloc_probe.py` (map the allocation wall) · `probes/bench_c1c6.py`.
The deploy report also describes the env-gated forward-hook NaN localizer (`GLM53_NAN_DEBUG=1`)
that names the first module emitting non-finite values.

### Fast loading: InstantTensor (experimental)

The v9 image adds the InstantTensor direct-I/O loader (`--load-format instanttensor`): loads
drop from ~10 minutes to 40–100 seconds and the page cache stays empty. **But in all four of
our v9 TP2 boots a rank died silently ~1 minute after loading** (exit code None, nothing in
dmesg) at every KV budget — so the shipped launchers do not enable it and the stable image
remains v8. This matches the known multi-node instability class for direct-IO loaders on Spark
(eugr/spark-vllm-docker#29). Because direct I/O never fills the page cache it also defeats the
first layer of the GB10 KV-allocation wall — full story in
[docs/GB10-KV-MEMORY-LADDER.md](docs/GB10-KV-MEMORY-LADDER.md). Credit: jack6464 (NVIDIA forum).

### vLLM v0.28.0 status (checked 2026-08-27)

**Not viable for GLM-5.3 yet**: the `glm5_next` architecture is not in the v0.28.0 release
(vllm-project/vllm#53906 still unmerged at check time) and no rebased day-0 image exists. The
day-0 image used here is itself a main-branch dev snapshot (`0.1.dev20051`) cut near the 0.28
branch point — this stack already runs 0.28-era engine code *plus* the GLM support 0.28 lacks.
Porting is mechanical when it opens: the patches are guarded string-replacements that apply or
refuse loudly.

---

## Credits

- **Model**: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) ·
  **Quant**: [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) (default, compressed-tensors)
  (their sm_121 notes were used directly) ·
  **Drafter**: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
- **barrydeen** — the gmu 0.85 reference config and quantization-coverage table from their
  independently published DGX Spark recipe
- **@ozskywalker** — [#5](../../pull/5), Dockerfile tag chain + build script
- vLLM [PR #53906](https://github.com/vllm-project/vllm/pull/53906) authors for the day-0 image;
  FlashInfer for the 0.6.18 SM90-NoPE MLA path; upstream
  [PR #52816](https://github.com/vllm-project/vllm/pull/52816) for DFlash2
- Deployed and debugged by Knox (Claude) for [@tonyd2wild](https://github.com/tonyd2wild)
