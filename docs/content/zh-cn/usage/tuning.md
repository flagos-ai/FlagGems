---
title: 性能预调优
weight: 85
---

<!--
# Achieving Optimal Performance with FlagGems
-->
<!--
While *FlagGems* kernels are designed for high performance usage scenarios,
achieving optimal end-to-end speed in full model deployments still demands
for careful integration and consideration of the system's runtime behavior.
In particular, two common performance bottlenecks are:
-->
# 基于预优化实现 FlagGems 的更好性能

尽管 *FlagGems* 中的内核是设计用于高性能使用场景的，在完整的模型部署环境中，
要想实现最佳的端到端性能，仍然需要精心地集成并仔细考查系统的运行时行为。
尤其是要注意以下两种常见的性能瓶颈：

<!--
- *Runtime autotuning overhead* in production environments.
- *Sub-optimal dispatching* due to the framework-level kernel registration and/or
  the interaction with the Triton runtime.
-->
- 在生产环境中运行时的自动调优过程的性能开销；
- 因为框架层算子注册以及与 Triton 运行时交互所带来的算子派发操作的性能损失。

<!--
These issues can occasionally offset the benefits of highly optimized kernels.
To address them, we provide two complementary optimization paths designed to ensure that
*FlagGems* operates at peak efficiency in real inference scenarios.
-->
这类问题在有些极端情况下会使得高度优化的内核所带来的改进不那么可观。
为了解决这类问题，我们提供两种辅助的优化方式，目的是为了确保 *FlagGems*
能够在真实的推理场景中提供最佳效率。

<!--
## Pre-tuning model shapes for inference scenarios

*FlagGems* provides [`LibTuner`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/utils/libentry.py#L206),
a lightweight enhancement to the auto-tuning system in Triton.
`LibTuner` introduces a **persistent, per-device tuning cache** that
helps mitigate some runtime overhead in the default auto-tuning process in Triton.
-->
## 针对推理场景的模型形状预优化

*FlagGems* 所提供的 [`LibTuner`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/utils/libentry.py#L206)
类可以看作是对 Triton 中自动调优系统的一种轻量级改进措施。
`LibTuner` 引入了一个**持久的、按设备组织的调优缓存**，
与 Triton 系统中默认的自动调优进程相比，这一缓存有助于降低一些运行时的性能开销。

<!--
### How LibTuner works?

Triton typically performs auto-tuning during the first few executions of a new input shape,
which may cause latency spikes — especially in latency-sensitive inference systems.
`LibTuner` addresses this with:
-->
### `LibTuner` 的工作原理

在处理新的输入形状时，Triton 通常会在前几轮执行过程中开展自动调优工作。
这些调优动作可能会导致系统的延迟显著上升，尤其是在对延迟敏感的推理系统中。
`LibTuner` 解决这一问题的方式如下：

<!--
- **Persistent caching**: Best auto-tuning configurations are persisted across runs.
- **Cross-process sharing**: The tuning cache is shared across processes on the same device.
- **Reduced runtime overhead**: Once tuned, operators will skip the tuning step in future runs.
-->
- **持久缓存**：将自动调优的配置持久化保存，跨多次运行复用；
- **跨进程共享**：对于在同一设备上运行的多个进程而言，这个调优缓存可以被共享；
- **降低运行时开销**：一旦执行过调优动作，将来执行算子时会自动跳过调优步骤。

<!--
This is particularly useful for operators like `mm` and `addmm`,
which often trigger the auto-tuning logic in Triton.
-->
这一优化过程对于类似 `mm` 和 `addmm` 这类算子而言非常有用，
因为这类算子在 Triton 当中常常会触发自动调优逻辑。

<!--
### How to Use Pre-tuning

To proactively warm up your system in order to populate the tuning cache:
-->
### 预优化机制的用法

{{% steps %}}
1. <!--Identify key input shapes used in your production workload.-->
   识别生产环境负载中使用的关键输入形状；

1. <!--Run the pre-tuning script to benchmark and cache best configs.
   You can run `python examples/pretune.py` as an example.-->
   运行预调优脚本进行性能基准测试并将最佳配置缓存下来。
   例如，你可以运行 `python examples/pretune.py` 来执行预调优。

1. <!--Deploy normally, and *FlagGems* will automatically pick the optimal config
   from cache during inference.-->
   按正常步骤部署应用，*FlagGems* 会在推理过程中自动从缓存中读取最佳配置。

{{% /steps %}}

<!--
> [!TIP]
> **Tips**
>
> - The `pretune.py` script accepts example shapes and workloads which can be used
>   to simulate your model's actual use cases. You can customize it for batch sizes,
>   sequence lengths, etc.
> - In frameworks like **vLLM** (`v0.8.5+`), adding `--compile-mode` to the command line
>   will automatically initiate a warmup step.
>   If *FlagGems*  is enabled, this flag also triggers `LibTuner`-based pre-tuning implicitly.
-->
> [!TIP]
> **提示**
>
> - `pretune.py` 脚本可以接受形状示例和负载示例，这些示例数据可用来模拟模型的实际使用场景。
>   你可以在批次大小、序列长度等多个角度进行定制。
> - 在 *vLLM*（`v0.8.5+`）这类框架中，在命令行中添加 `--compile-mode`
>   可以自动触发预热步骤。
>   如果 *FlagGems* 已被启用，这一参数选项也会隐式触发基于 `LibTuner`
>   的预调优操作。

<!--
For more details (e.g. customizing your tuning cache path and settings),
refer to the [`examples/pretune.py`](https://github.com/flagos-ai/FlagGems/blob/master/examples/pretune.py)
as an example.
-->
如果希望进一步了解细节（例如如何定制调优缓存的路径和配置等），
可以参阅源码仓库中的 [`examples/pretune.py`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/examples/pretune.py)
文件。
