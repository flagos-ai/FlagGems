---
title: Pre-Tuning
weight: 85
---

# Achieving Optimal Performance with Gems

While *FlagGems* kernels are designed for high performance, achieving optimal end-to-end speed
in full model deployments requires careful integration and consideration of runtime behavior.
In particular, two common performance bottlenecks are:

- **Runtime autotuning overhead** in production environments.
- **Suboptimal dispatching** due to framework-level kernel registration or interaction with the Triton runtime.

These issues can occasionally offset the benefits of highly optimized kernels.
To address them, we provide two complementary optimization paths designed to ensure that
*FlagGems* operates at peak efficiency in real inference scenarios.

## Pre-tuning Model Shapes for Inference Scenarios

*FlagGems* provides [`LibTuner`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/utils/libentry.py#L139),
a lightweight enhancement to Triton’s autotuning system.
`LibTuner` introduces a **persistent, per-device tuning cache** that
helps mitigate runtime overhead in the Triton’s default autotuning process.

### Why Pre-tuning?

Triton typically performs autotuning during the first few executions of a new input shape,
which may cause latency spikes—especially in latency-sensitive inference systems.
`LibTuner` addresses this with:

- _Persistent caching_: Best autotune configs are saved across runs.
- _Cross-process sharing_: Cache is shared across processes on the same device.
- _Reduced runtime overhead_: Once tuned, operators skip tuning in future runs.

This is particularly useful for operators like `mm` and `addmm`,
which often trigger the autotune logic in Triton.

### How to Use Pre-tuning

To proactively warm up your system and to populate the cache:

1. Identify key input shapes used in your production workload.
1. Run the pre-tuning script to benchmark and cache best configs.
   You can run `python examples/pretune.py` as an example.
1. Deploy normally, and *FlagGems* will automatically pick the optimal config
   from cache during inference.

> [!TIP]
> **Tips**
> - The `pretune.py` script accepts example shapes and workloads
>   which can be used to simulate your model's actual use cases.
>   You can customize it for batch sizes, sequence lengths, etc.
> - In frameworks like **vLLM** (`v0.8.5+`), enabling `--compile-mode`
>   automatically performs a warmup step.
>   If *FlagGems*  is enabled, this also triggers `LibTuner`-based
>   pre-tuning implicitly.

For more details (e.g. customizing your tuning cache path and settings),
refer to the [`examples/pretune.py`](https://github.com/flagos-ai/FlagGems/blob/master/examples/pretune.py)
as an example.
