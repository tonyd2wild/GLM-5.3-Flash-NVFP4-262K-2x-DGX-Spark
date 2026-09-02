# TP2 tuning: KV pool, speculative depth, and CUDA graphs

Measured 2026-09-02 on a 2x DGX Spark (GB10) pair, RedHatAI compressed-tensors checkpoint,
DFlash2, temperature 0, 8-prompt set, median of 3, **each arm under its own
`VLLM_CACHE_ROOT`** (vLLM [#53366](https://github.com/vllm-project/vllm/issues/53366) omits
`num_speculative_tokens` from the config hash, so a shared cache can serve one arm an
artifact compiled for a different k).

Findings originate in [#12](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/pull/12)
by @tmooch; this doc records the reproduction on the reference fleet.

## KV 3 -> 6 GiB: unconditional

@tmooch measured 6 preemptions under load at 3 GiB and 0 at 6 GiB, with the pool going
310,292 -> 678,661 tokens. Preemption under load costs more than any tok/s figure, and the
memory is available. **Adopted as the default.**

## Speculative depth: k=7 below C4, k=5 above it

`num_speculative_tokens` is a **workload choice, not a default.** It inverts at C4:

| C | k=7 | k=5 | delta |
|---|---|---|---|
| 1 | 44.75 | 42.18 | −5.7% |
| 2 | 68.18 | 64.12 | −6.0% |
| **4** | 64.83 | **84.20** | **+29.9%** |
| **6** | 84.82 | **100.33** | **+18.3%** |

Single stream goes the other way — k=5 is slower on every prompt:

| prompt | k=7 | k=5 | delta |
|---|---|---|---|
| count-to-100 | 67.12 | 55.50 | −17.3% |
| code | 47.33 | 43.15 | −8.8% |
| tooluse | 48.72 | 44.94 | −7.8% |
| sql | 41.19 | 38.12 | −7.5% |
| json | 46.73 | 43.83 | −6.2% |
| math | 41.26 | 39.80 | −3.5% |

**Mechanism.** At low concurrency the engine is weight-read-bound, so the extra draft
positions are nearly free and more k means more accepted tokens per step. Under load,
verify compute binds — and on a 288-expert MoE it grows super-linearly in k, because each
extra verified token activates more experts — so fewer, higher-quality draft positions win.

**The default stays k=7** (single-user and low-concurrency is the common case, and it is
what the README's headline figures use). Serving deep concurrency should set k=5, or better,
use a schedule with the crossover at C4:

```json
"num_speculative_tokens_per_batch_size": [[1,3,7],[4,512,5]]
```

### Do not tune toward acceptance ratio

Under k=5 the acceptance **ratio rose on every prompt** (count-to-100 0.940 -> 0.977,
code 0.664 -> 0.748, tooluse 0.736 -> 0.825) while throughput **fell on every prompt**. The
ratio rises because you stopped drafting the low-probability tail, not because the drafter
improved. The metric that tracks throughput is **mean accepted length**, which fell
**4.84 -> 4.15**.

## CUDA graphs do NOT transfer from TP4

`cudagraph_mode: FULL_AND_PIECEWISE` is a clear win on the 4-node fleet (best aggregate
503 -> 530 tok/s, prose +17%). On TP2, with only that flag changed, it is **flat**:

| C | eager | FULL_AND_PIECEWISE | delta |
|---|---|---|---|
| 1 | 44.75 | 46.58 | +4.1% |
| 2 | 68.18 | 65.05 | −4.6% |
| 4 | 64.83 | 65.53 | +1.1% |
| 6 | 84.82 | 83.34 | −1.7% |

Mean ≈ −0.3%. Single stream is mixed (tooluse +13.5%, prose +10.2%, sql −15.9%).

**Plausible mechanism:** at TP4 each rank holds ~1/4 of the model, so per-step kernel-launch
overhead is a larger share of a shorter step and removing it pays. At TP2 each rank holds
~1/2, the step is longer and more weight-read-bound, and the same overhead is proportionally
smaller. **TP2 keeps `--enforce-eager`.**
