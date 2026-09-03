# GB10 unified-memory field report: loading a 178 GiB checkpoint without losing the node

A field report against [OPEN-PROBLEMS §4](OPEN-PROBLEMS.md) (UVM driver
livelock under memory pressure) and §5 (pinned KV budgets), from a second
2x DGX Spark fleet running the TP2 DFlash2 recipe with the RedHatAI NVFP4
weights at 262K context. Everything below was measured between 2026-09-01
and 2026-09-03 on two GB10 nodes (121.69 GiB unified memory each). Numbers
are from receipts we keep per attempt; where a number is an estimate it says
so.

## 1. What actually breaks at load time, and the guard that stops it

Streaming the checkpoint through the page cache leaves almost no free
contiguous blocks of 2 MiB or larger on either node. The NVIDIA driver then
fails an allocation during vLLM's profile run (`NV_ERR_NO_MEMORY` from
`_memdescAllocInternal`), and the process spins exactly as §4 describes. We
saw this twice in a row before adding a guard.

What worked was not swappiness but keeping free memory up during the load:

- A per-node **memory guard** runs beside the rank from launch to readiness.
  Every second it reads `/proc/meminfo`; when `MemFree` drops under a floor
  (8 GiB) it writes `3` to `drop_caches` and `1` to `compact_memory`. It
  stops at the `Application startup complete` marker (rank 0 log), a stop
  file, or a 45-minute deadline, and restores every sysctl it touched.
- With the guard the head passed the point where it used to die, with
  `MemFree` never below 2.6 GiB and no driver record in the kernel log.
  Both loads and the full qualification later completed with **0 major page
  faults in the rank cgroups** on both nodes (see §3).

The guard writes a JSON receipt of every drop so the effect is auditable.
The cadence matters: one drop three minutes before the profile run was not
enough on its own; the floor loop was.

## 2. Two sharp edges we hit on the way

**Do not raise `vm.min_free_kbytes` before vLLM's startup check.** Our first
guard set a 4 GiB kernel floor at launch. vLLM's startup free-memory check
on integrated GPUs uses `psutil` `MemAvailable`, and the kernel watermark
reserve lowers that figure by about 6.2 GiB. Both ranks then failed the
check with 98.89 and 100.6 GiB "free" against the 102.22 GiB that
`--gpu-memory-utilization 0.84` demands (0.84 x 121.69). Two more facts from
that dead end: `torch.cuda.mem_get_info()` on GB10 reports `MemFree`, not
`MemAvailable`, so the two numbers disagree by design; and
`watermark_boost_factor` is 15000 on these kernels, so a 4 GiB floor can
grow to about 15 GiB of kept-free memory.

**After the pinned allocation there is almost nothing contiguous left.** With
the roughly 90 GiB of model memory pinned, only about 500 to 650 MiB of
order-9 (2 MiB) free blocks remain on the head even after compaction. Any
later allocation that needs large pages competes for that. This is the
reason the guard has to keep working until readiness, not only until the
weights are loaded.

## 3. Major page faults during serving are not what they look like

We gate the qualification on the `pgmajfault` counter of each rank's cgroup.
Two different things moved it:

- **Deterministic first-touch faults.** About two minutes into a 262K
  accuracy run the `VLLM::Worker` thread first-touched untouched pages of
  `libcrypto.so.3` (three faults) and `libssl.so.3` (one). Attributed with
  `bpftrace`; identical timing in two runs. Not memory pressure.
- **Page-cache eviction of model pages.** The guard's own drops evict
  checkpoint pages that the ranks still map. On the worker that produced 237
  major faults in the first minutes of the suite. The fix was ordering: stop
  the worker guard the moment the head reports ready, then run one **warm
  pass** that reads the checkpoint files back into the page cache before the
  first request. In the final run the warm pass finished 88 s before the
  suite started and both counters stayed at 0 for the whole suite.

If you gate on faults, gate after the warm pass, and never `docker exec`
into a rank during the measured window: the exec joins the rank cgroup and
its own faults are counted.

One warning about the warm pass. It floods the page cache with the whole
payload, which is the same memory state we were in during both hard
power-offs in section 5. We accept that risk only because the pass runs
while the ranks already hold their memory, it is short, and a node we can
power-cycle is nearby. On a remote node, weigh it against section 5 first:
read back only the files the ranks actually map, or skip the pass and gate
on faults with a tolerance instead.

## 4. KV budget and capacity at 262K, one stream

§5 warns that pinning `--kv-cache-memory` drops the activation reservation.
Our working point, for the record (fp8 KV, `--block-size 2304`, DFlash2
width 7, `--enforce-eager`, `--max-num-seqs 1`):

| setting or result | value |
|---|---:|
| `--gpu-memory-utilization` | 0.84 |
| `--kv-cache-memory-bytes` | 4294967296 (4 GiB) |
| distributed KV capacity reported by vLLM | 414,615 tokens |
| KV use during one 262,144-token request | 63 % |
| cold TTFT, 261,632 prompt tokens | 191.7 s |
| decode p50, 8,192-token prompts | 55.6 tok/s |
| prefix-cache hit rate on the repeated 262K prompt | 0.986 |
| DFlash2 acceptance | 0.928 |
| policy minimum idle `MemAvailable` after load | 4.5 GiB (met) |

The option is `--kv-cache-memory-bytes`; the launcher in this repository
writes the same option as the prefix `--kv-cache-memory`.

Higher budgets failed the startup check or the profile run on our nodes; the
4 GiB pin is what ended the driver failures. One stream is deliberate: the
long-context concurrency collapse in #14 reproduces here.

Also worth a receipt: a one-shot `gc.collect()` plus `torch.cuda.empty_cache()`
right after warm-up (before vLLM freezes the GC heap) returns the allocator's
warm-up slack on these unified-memory nodes. We log the allocator and
`MemAvailable` snapshots before and after as JSON so the effect is visible per
boot.

## 5. Hard power-offs with the GPU idle

Two log-less hard power-offs of the head in nine long preflight runs, both
while a single-threaded `sha256sum` pass over the 110 GB payload flooded the
page cache (about 115 GB cached, 3.8 GB free), with the GPU idle at 4.9 W,
43 C, 208 MHz and the thermal zones at 44 to 46 C. Zero power-offs during
GPU-heavy runs. So at least one power-off class on these units is not the
GPU clock, and a clock cap does not cover it. The board does not power back
on by itself after that event even with the firmware set to Power On after
AC loss; a physical AC cycle was needed. Wake-on-LAN is not supported by the
NIC or firmware. A smart plug on the head's cord is the only remote recovery
we found.

The clock cap itself (`nvidia-smi -lgc 300,2200` in a oneshot unit after
`nvidia-persistenced`) cost nothing measurable on this recipe: decode stayed
at 55.6 tok/s while the GPUs ran at 47 to 50 W and 69 to 71 C instead of
78 W and 86 C.

## 6. Small things for Sparkrun users

- `sparkrun stop` can leave the worker rank running with the model still in
  memory while reporting both hosts idle. After every stop, check `docker ps`
  on the worker; stop leftovers with `docker stop -t 120` and `docker rm`.
- Sparkrun containers are `--rm`: a reboot deletes them and their in-container
  logs. Copy `/tmp/sparkrun_serve.log` out before you stop anything you may
  need as evidence.
