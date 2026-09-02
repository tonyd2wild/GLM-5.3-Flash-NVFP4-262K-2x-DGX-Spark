# CURRENT recipe

The one configuration this repo ships today. If anything else in the repo disagrees with
this file, this file wins.

Last verified: 2026-09-02.

---

## Topology

GLM-5.3-Flash NVFP4 + DFlash2, **tensor-parallel 2 across two DGX Spark (GB10/SM121) nodes**,
**262,144-token context**.

| rank | host | IP | role |
|---|---|---|---|
| **1** | Spark4 | `192.168.192.4` | worker (`--headless`) — **launch FIRST** |
| **0** | Reddie | `192.168.192.2` | head — serves the OpenAI API on **:8000** |

Rendezvous: master addr `192.168.192.2`, master port `29521`, `--nnodes 2`,
`--distributed-executor-backend mp`.

**1M-context TP2 variants starved the KV pool and crashed the fleet three times on
2026-09-01. 262K is the current context.** Do not raise `--max-model-len` on TP2.

## The one launcher

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches      # BOTH nodes, every launch
./launch-glm53-vllm-tp2-dflash2.sh 1    # Spark4, rank 1, worker — FIRST
sleep 25
./launch-glm53-vllm-tp2-dflash2.sh 0    # Reddie, rank 0, head — serves :8000
```

[`launch-glm53-vllm-tp2-dflash2.sh`](launch-glm53-vllm-tp2-dflash2.sh) is the only launcher
for this recipe. The drop-caches step is not optional on these unified-memory nodes.

Readiness takes ~15 minutes (shard load dominates). Poll `/health`, never `/v1/models` —
the latter returns 200 from config alone with a dead engine behind it.

## Image

```
ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
```

Public, anonymous pull, on **both** nodes. Also required on both nodes before launch:
`cp docker/sparse_attn_indexer_kpool_sm121.py ~/patches/sparse_attn_indexer_kpool.py`
(the SM121 top-k fix the launcher bind-mounts over the image's copy).

## Weights

```
/var/tmp/models/GLM-5.3-Flash-NVFP4-redhat
```

This is the path the launcher checks (`MODEL_HOST_PATH`), on both nodes.
[RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4),
compressed-tensors.

**ModelOpt and abliterated NVFP4 quants corrupt tokens on this stack; the launcher refuses
them.** The guard reads `quantization_config.quant_method` from `config.json` and exits 5 on
`modelopt` unless `ALLOW_MODELOPT=1` (vLLM
[#54150](https://github.com/vllm-project/vllm/issues/54150)).

The drafter, also on both nodes: `/var/tmp/models/GLM-5.3-Flash-DFlash2` (2.2 GB,
[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)).

## Flags that define the recipe

| flag | value | why |
|---|---|---|
| `--tensor-parallel-size` | `2` | two Sparks |
| `--max-model-len` | `262144` | 1M starved the pool and crashed 3× on 2026-09-01 |
| `--speculative-config` | `{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7}` | DFlash2, **k=7** |
| `--kv-cache-dtype` | `fp8_e4m3` | |
| `--kv-cache-memory` | `6442450944` (**6 GiB, pinned**) | see below |
| `--enforce-eager` | on | see below |
| `--gpu-memory-utilization` | `0.85` | 0.78–0.80 starve KV; 0.87 is not reachable on every node |
| `--block-size` | `2304` | DeepGEMM arch-12 fp8 paged-MQA needs 64-entry pool pages |
| `--moe-backend` | `marlin` | |
| `--max-num-seqs` | `6` | |
| `--max-num-batched-tokens` | `8192` | |
| `--chat-template` | `chat_template_mm.jinja` (from the weights dir) | vision; image requests 500 without it |
| `--served-model-name` | `glm-5.3-flash` | |
| tool/reasoning | `--tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 --default-chat-template-kwargs '{"enable_thinking":false}'` | |

### KV **is** pinned at 6 GiB

This reverses earlier README guidance ("let the profiler size the pool, do not pin
`--kv-cache-memory`"). The shipped launcher pins **6442450944 bytes**. Rationale and
measurement: [`docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md`](docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md),
section "KV 3 -> 6 GiB: unconditional" — 6 preemptions under load at 3 GiB, **0 at 6 GiB**,
with the pool going **310,292 -> 678,661 tokens**. Preemption under load costs more than any
tok/s figure, and the memory is available. Adopted as the default in
[#16](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/pull/16).

Measured KV pool at the shipped pin: **678,661 tokens**.

### `--enforce-eager` stays on for TP2

CUDA graphs are the **TP4 sibling repo's lane**, not this one. Same doc, section "CUDA graphs
do NOT transfer from TP4": `cudagraph_mode: FULL_AND_PIECEWISE` is a clear win at TP4
(503 -> 530 tok/s aggregate) and **flat** at TP2 — C1 +4.1%, C2 −4.6%, C4 +1.1%, C6 −1.7%,
mean ≈ −0.3%. TP2 keeps `--enforce-eager`.

## Expected numbers

All from [`docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md`](docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md),
section "Speculative depth: k=7 below C4, k=5 above it" — measured 2026-09-02, RedHatAI
compressed-tensors checkpoint, DFlash2, temperature 0, 8-prompt set, median of 3, each arm
under its own `VLLM_CACHE_ROOT`.

Shipped config (k=7), aggregate by concurrency:

| C | tok/s |
|---|---|
| 1 | 44.75 |
| 2 | 68.18 |
| 4 | 64.83 |
| 6 | 84.82 |

Shipped config (k=7), single stream by prompt:

| prompt | tok/s |
|---|---|
| count-to-100 | 67.12 |
| code | 47.33 |
| tooluse | 48.72 |
| sql | 41.19 |
| json | 46.73 |
| math | 41.26 |

`num_speculative_tokens` is a workload choice, not a default: the same section measures k=5
beating k=7 by +29.9% at C4 and +18.3% at C6, while losing on every single-stream prompt.
Deep-concurrency serving should set k=5, or use the schedule with the crossover at C4:
`"num_speculative_tokens_per_batch_size": [[1,3,7],[4,512,5]]`. **The default stays k=7.**

## How we quote numbers

- **Decode** is quoted from **real prompts** — prose and code — because acceptance, and
  therefore throughput, is workload-bound. A single decode number without the prompt mix is
  not comparable to anything.
- **The counting prompt** (count-to-100 and friends) is quoted only as a **labeled
  draft-acceptance ceiling**. It is the most predictable text the drafter will ever see; it is
  not a decode figure and must never be presented as the headline.
- **Prefill** is quoted **cold only**. First inference JIT-compiles kernels, so a warm prefill
  number measures the cache, not the stack.

## Four Sparks?

Use the sibling repo:
**[GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)**
— TP4, the model-native 1M context, and the lane where CUDA graphs pay.

## Do not use

Superseded. Kept for history and for links from open issues and PRs; **moving to `archive/`
in the next cleanup commit**. None of these describes the current recipe:

| path | superseded by |
|---|---|
| `launch-glm53-vllm-tp2.sh` | `launch-glm53-vllm-tp2-dflash2.sh` (no-drafter v8 lane) |
| `launch-glm53-vllm-tp4.sh` | the TP4 sibling repo |
| `cache_flusher.sh` | the `drop_caches` line above |
| `overlay-dflash2/` | `docker/dflash2-overlay/` |
| `docs/DEPLOY-REPORT.md` | this file + `docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md` |
| `docs/BENCH-C1-C6-DFLASH2.md` | `docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md` |
| `docs/GB10-KV-MEMORY-LADDER.md` | the 6 GiB pin above |
| `docs/KV-HUNT-672K-TP2-RECORD.md` | the 6 GiB pin above |
| `docs/NVFP4-KV-BUILD-SPEC.md` | fp8 KV is the shipped lane; NVFP4 KV is unresolved |
| `docs/kv_hunt_log.md` | the 6 GiB pin above |
| `probes/bench_glm53.py` | `probes/bench_robust.py` |

Still current: `docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md`,
`docs/SM121-CRASH-FORENSICS-2026-08-27.md`, `docs/DFLASH2-SPECULATIVE-DECODING.md`,
`docs/OPEN-PROBLEMS.md`, `docs/issue-*.md`, `docker/`, `probes/bench_c1c6.py`,
`probes/bench_robust.py`, and the probe kit.
