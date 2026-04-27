---
title: 分布式环境
weight: 70
---

<!--
# Multi-GPU Deployment

In real-world LLM deployment scenarios, multi-GPU or multi-node setups are often required
to support large model sizes and high-throughput inference.
*FlagGems* supports these scenarios by accelerating operator execution across multiple GPUs.
-->
# 多 GPU 与分布式环境

在现实世界的 LLM 部署场景中，人们通常需要多 GPU 或者多节点的环境来支持较大的模型，
与/或完成高吞吐量的推理任务。
*FlagGems* 通过允许跨多个 GPU 完成算子执行加速来支持这类使用场景。

<!--
## 1. Single-node vs multi-node usage

For **single-node deployments**, the integration is straightforward. You can import `flag_gems`
and invoke `flag_gems.enable()` at the beginning of your script.
This enables acceleration without requiring any additional changes.
-->
## 1. 单节点与多节点用法

对于**单节点部署**而言，集成工作是相对简单直接的。
你可以在你的代码开始部分 `import flag_gems` 之后调用 `flag_gems.enable()`。
无需其他的变更，你就可以获得算子加速的效果。

<!--
In **multi-node deployments**, however, this approach is insufficient.
Distributed inference frameworks (like vLLM) spawn multiple worker processes across nodes,
where every process must initialize `flag_gems` individually.
If the activation occurs only in the launch script on one node, worker processes
on other nodes will fall back to the default implementation which is not accelerated.
-->
然而，在**多节点部署**环境中，这种方法是不够的。
分布式的推理框架（例如 vLLM）需要跨多个节点来启动多个工作进程，
每个进程都需要独立初始化 `flag_gems`。
如果对 `flag_gems` 的启用或激活操作仅发生在第一个节点的启动代码上，
其他节点上的工作进程会回退为默认的算子实现，也就是没有被加速过的版本。

<!--
## 2. Example: integration with vLLM and DeepSeek

To enable *FlagGems* in a distributed vLLM + DeepSeek deployment:
-->
## 2. 示例：与 vLLM 和 DeepSeek 集成

要在一个分布式的 vLLM + DeepSeek 部署环境中启用 *FlagGems*，需执行以下步骤：

{{% steps %}}

1. <!--**Baseline verification**-->
   **基线检验**

   <!--
   Before conducting this experiment, please verify that the model can load
   and serve correctly without integrating *FlagGems*.
   For example, loading a model like `Deepseek-R1` typically requires **at least two H100 GPUs**
   and it can take **up to 20 minutes** to initialize, depending on the checkpoint size and
   the system I/O bandwidth and latency.
   -->
   在开始此实验之前，请先检查模型在没有与 *FlagGems* 集成的前提下可以正常启动并提供服务。
   例如，加载类似于 `Deepseek-R1` 这类模型通常需要**至少 2 块 H100 GPU 卡**，
   并且其初始化过程可能需要**将近 20 分钟**的时间，取决于检查点的大小、系统 I/O
   的带宽与延迟。

1. <!--**Inject `flag_gems` into vLLM worker code**-->
   **将 `flag_gems` 注入到 vLLM 工作进程代码中**

   <!--
   Locate the appropriate model runner script depending on your vLLM version:

   - If you are using the **vLLM v1 architecture** (available in vLLM ≥ 0.8),
     modify `vllm/v1/worker/gpu_model_runner.py`
   - If you are using the **legacy v0 architecture**, modify `vllm/worker/model_runner.py`

   In either file, insert the following logic after the last `import` statement:
   -->
   基于你所使用的 vLLM 版本，找到模型运行脚本的位置：

   - 如果所使用的是**vLLM v1 架构**（在 vLLM ≥ 0.8 环境中可用），
     要修改 `vllm/v1/worker/gpu_model_runner.py` 文件；
   - 如果所使用的是**老的 v0 架构**, 则需要修改 `vllm/worker/model_runner.py` 文件。

   打开所找到的文件，在最后一行 `import` 语句之后插入如下逻辑：

   ```python
   import os
   if os.getenv("USE_FLAGGEMS", "false").lower() in ("1", "true", "yes"):
       try:
           import flag_gems
           flag_gems.enable()
           flag_gems.apply_gems_patches_to_vllm(verbose=True)
           logger.info("Successfully enabled flag_gems as default ops implementation.")
       except ImportError:
           logger.warning("Failed to import 'flag_gems', falling back to default implementation.")
       except Exception as e:
           logger.warning(f"Failed to enable 'flag_gems': {e}, falling back to default implementation.")
   ```

1. <!--**Set environment variables on all nodes**-->
   **在所有节点上设置环境变量**

   <!--
   Before launching the service, ensure the following environment variable is set
   on all nodes:
   -->
   在启动服务之前，要确保所有节点上都设置了下面的环境变量：

   ```shell
   export USE_FLAGGEMS=1
   ```

1. <!--**Start distributed inference and verify**-->
   **启动分布式推理服务并检验运行状态**

   <!--
   Launch the service and check the startup logs on each node for messages
   indicating that operators have been overridden.
   -->
   启动分布式推理服务，在所有节点上检查启动日志，搜索表明算子已经被覆盖的消息。

   ```none
   Overriding a previously registered kernel for the same operator and the same dispatch key
   operator: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
     registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
   dispatch key: CUDA
   previous kernel: registered at /pytorch/aten/src/ATen/....
        new kernel: registered at /dev/null:488 (Triggered internally at ....)
   self.m.impl(
   ```

   <!--
   This confirms that `flag_gems` has been successfully enabled across all nodes.
   -->
   出现这类消息则意味着 `flag_gems` 已经被成功地跨多个节点启用了。

{{% /steps %}}
