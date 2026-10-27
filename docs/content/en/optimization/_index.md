---
title: Backend Optimization
weight: 45
bookCollapseSection: true
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

# Backend Hardware Optimization Guides

On heterogeneous AI accelerators, the optimal launch configurations and tiling sizes for Triton kernels heavily depend on the physical hardware architecture and compiler backend. When optimizing for a specific hardware target, we recommend thoroughly understanding its memory hierarchy and instruction constraints before exploring tiling parameters via the [FlagGems tuning workflow](../usage/tuning.md).

- **[Huawei Ascend NPU Optimization](ascend/)**: AI Core memory and compute paths (Cube vs. Vector), on-chip buffer budgeting (L0/L1/UB), Triton-Ascend compiler parameters, and common bottleneck troubleshooting.
- **[Hygon DCU Optimization](hygon/)**: DCU/HIP memory hierarchy, Wavefront occupancy limits, DUMMA hardware matrix instruction shapes, and grouped scheduling for L2 reuse.

> [!TIP]
> The configurations and tuning suggestions in these guides represent empirical baselines. Always verify both numerical precision and end-to-end latency on the physical target device with your installed compiler toolchain before adopting them in production.
