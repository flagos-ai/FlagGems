---
title: 性能预调优
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


# 性能预调优（Pre-tuning）

虽然 *FlagGems* 算子内核本身具备卓越的计算性能，但在复杂的全模型生产部署场景下，端到端延迟往往还会受到运行时特性的影响。其中最典型的性能瓶颈包括：

- **运行期自动调优开销（Runtime Autotuning Overhead）**：Triton 默认会在首次遇到新 Tensor Shape 时在线遍历候选配置，导致首次推理出现严重的延迟尖刺（Latency Spike）。
- **调度分发损耗（Dispatch Overhead）**：高频小算子在 Python 解释器与框架分发层产生的额外调度开销。

为了解决在线调优引起的延迟波动，*FlagGems* 提供了轻量级的离线/启动期预调优机制，确保模型在真实推理场景下全程保持极致性能。

## 基于 `LibTuner` 的 Shape 预调优

*FlagGems* 实现了 [`LibTuner`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/utils/libentry.py#L206) 工具类，对 Triton 原生自动调优机制进行了深度优化。`LibTuner` 引入了**按硬件设备持久化组织的调优配置缓存**，有效规避了运行时重复探索配置的性能损耗。

### `LibTuner` 的核心优势

对于 `mm`（矩阵乘法）、`addmm` 等重度依赖切块（Tiling）参数优化的算子，Triton 默认调优行为容易造成推理服务抖动。`LibTuner` 通过以下特性彻底解决了该问题：

- **配置持久化（Persistent Caching）**：将基准测试选出的最优配置持久化落盘，服务重启后可立即复用；
- **多进程安全共享（Cross-process Sharing）**：同一硬件设备上的多个推理 Worker 进程可共享同一份调优缓存；
- **零运行时调优损耗（Zero Runtime Overhead）**：一旦缓存建立，后续执行自动匹配并加载最优配置，彻底消除在线探索耗时。

### 预调优操作流程

{{% steps %}}
1. **收集关键 Shape**：根据实际生产业务流量，梳理出最具代表性的 Batch Size 与输入/输出序列长度组合；
2. **执行预调优脚本**：运行预调优脚本对目标 Shape 进行基准评测并持久化最佳参数配置。例如运行：
   ```shell
   python examples/pretune.py
   ```
3. **正式上线部署**：生产服务启动并启用 *FlagGems* 后，算子在推理过程中将毫秒级命中落盘的最优配置，无需预热调优。
{{% /steps %}}

> [!TIP]
> **最佳实践**
>
> - `pretune.py` 支持传入自定义 Shape 列表与测试负载，可精准契合目标大模型的真实推理配置。
> - 在 **vLLM**（`v0.8.5+`）等主流框架中，命令行传入 `--compile-mode` 会自动触发预热流程；当环境中启用 *FlagGems* 时，该参数也会隐式激活基于 `LibTuner` 的预调优流程。

更多调优缓存路径配置与高级选项，请参阅 [`examples/pretune.py`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/examples/pretune.py)。
