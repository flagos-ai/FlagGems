---
title: Pre-Tuning
weight: 85
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


# Achieving Optimal Performance with FlagGems

While *FlagGems* kernels are designed for high performance usage scenarios,
achieving optimal end-to-end speed in full model deployments still demands
for careful integration and consideration of the system's runtime behavior.
In particular, two common performance bottlenecks are:

- *Runtime autotuning overhead* in production environments.
- *Sub-optimal dispatching* due to the framework-level kernel registration and/or
  the interaction with the Triton runtime.

These issues can occasionally offset the benefits of highly optimized kernels.
To address them, we provide two complementary optimization paths designed to ensure that
*FlagGems* operates at peak efficiency in real inference scenarios.

## Pre-tuning model shapes for inference scenarios

*FlagGems* provides [`LibTuner`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/utils/libentry.py#L206),
a lightweight enhancement to the auto-tuning system in Triton.
`LibTuner` introduces a **persistent, per-device tuning cache** that
helps mitigate some runtime overhead in the default auto-tuning process in Triton.

### How LibTuner works?

Triton typically performs auto-tuning during the first few executions of a new input shape,
which may cause latency spikes — especially in latency-sensitive inference systems.
`LibTuner` addresses this with:

- **Persistent caching**: Best auto-tuning configurations are persisted across runs.
- **Cross-process sharing**: The tuning cache is shared across processes on the same device.
- **Reduced runtime overhead**: Once tuned, operators will skip the tuning step in future runs.

This is particularly useful for operators like `mm` and `addmm`,
which often trigger the auto-tuning logic in Triton.

### How to Use Pre-tuning

To proactively warm up your system in order to populate the tuning cache:

{{% steps %}}
1. Identify key input shapes used in your production workload.
1. Run the pre-tuning script to benchmark and cache best configs.
   You can run `python examples/pretune.py` as an example.
1. Deploy normally, and *FlagGems* will automatically pick the optimal config
   from cache during inference.
{{% /steps %}}

> [!TIP]
> **Tips**
>
> - The `pretune.py` script accepts example shapes and workloads which can be used
>   to simulate your model's actual use cases. You can customize it for batch sizes,
>   sequence lengths, etc.
> - In frameworks like **vLLM** (`v0.8.5+`), adding `--compile-mode` to the command line
>   will automatically initiate a warmup step.
>   If *FlagGems*  is enabled, this flag also triggers `LibTuner`-based pre-tuning implicitly.

For more details (e.g. customizing your tuning cache path and settings),
refer to the [`examples/pretune.py`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/examples/pretune.py)
as an example.

## Caching ATen routing and launch plans

Small, frequently repeated pointwise operators can spend more host time in
shape routing, dtype promotion, grid calculation, and Triton argument expansion
than in the GPU kernel itself. The experimental ATen plan cache stores this
structural work after warmup. It uses a monomorphic last-plan entry followed by
a bounded LRU for dynamic shapes.

The feature is disabled by default. Enable it explicitly through Python or an
environment variable:

```python
import flag_gems

flag_gems.enable_aten_plan_cache(max_size=128)
```

```bash
export FLAGGEMS_ATEN_PLAN_CACHE=1
export FLAGGEMS_ATEN_PLAN_CACHE_SIZE=128
```

`FLAGGEMS_ATEN_PLAN_CACHE_INCLUDE` and `FLAGGEMS_ATEN_PLAN_CACHE_EXCLUDE`
accept comma-separated glob patterns over the fully qualified kernel name.
`arange_func` is excluded by default because its measured setup cost was lower
than a plan-cache lookup. Runtime statistics are available from
`flag_gems.aten_plan_cache_stats()`.

Cache keys include the vendor, device, device capability, dtype, shape, stride,
broadcast pattern, scalar values, output semantics, and code-generation
configuration. Cached plans do not retain tensors, data pointers, streams,
events, or graphs. Plans are cleared after `fork`, and FlagTune changes
invalidate affected launch plans. Use `disable_aten_plan_cache()` for a quick
A/B test or `uninstall_aten_plan_cache()` to restore the original wrappers.
