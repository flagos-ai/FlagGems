---
title: 分布式环境
weight: 70
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

# 多 GPU 与分布式环境部署

在超大规模语言模型（LLM）的真实生产部署中，往往需要多卡或跨节点分布式集群来承载百亿/千亿级参数模型，以实现高吞吐、低延迟的推理服务。*FlagGems* 全面支持在多 GPU 及多机分布式部署环境中稳定运行。

## 1. 单机多卡与多机集群部署差异

- **单机单卡 / 单机多卡（单进程拉起）**：集成非常简单。只需在主进程脚本入口处引入 `flag_gems` 并调用 `flag_gems.enable()`，所有子线程或由主进程分发的 CUDA 计算均能透明享受加速。
- **多机多节点 / 多进程集群（如 vLLM 分布式 Worker）**：主流分布式推理引擎通常会通过 Ray、PyTorch RPC 或多进程独立拉起各节点 Worker。若仅在 Master 节点的启动脚本中调用 `enable()`，其他节点或子进程由于处于独立的 Python 运行环境中，将无法自动继承加速配置，从而回退为未经优化的默认实现。

## 2. 实践案例：vLLM + DeepSeek 跨节点分布式部署

以基于 vLLM 部署大型模型（例如 DeepSeek-R1 系列）为例，在分布式环境中启用 *FlagGems* 的标准实操步骤如下：

{{% steps %}}

1. **基线连通性验证**
   在集成 *FlagGems* 之前，建议首先确保原版模型在分布式环境下能够正常加载并完成基础推理服务验证。

2. **在 vLLM Worker 中注入初始化逻辑**
   根据所使用的 vLLM 版本，定位对应的模型执行器（Model Runner）源码：
   - **vLLM v1 架构**（vLLM ≥ 0.8 推荐架构）：编辑 `vllm/v1/worker/gpu_model_runner.py`
   - **vLLM v0 遗留架构**：编辑 `vllm/worker/model_runner.py`

   在文件末尾的 `import` 之后添加环境变量控制的注入逻辑：

   ```python
   import os

   if os.getenv("USE_FLAGGEMS", "false").lower() in ("1", "true", "yes"):
       try:
           import flag_gems

           flag_gems.enable()
           flag_gems.apply_gems_patches_to_vllm(verbose=True)
           logger.info(
               "Successfully enabled FlagGems as default ATen and vLLM custom ops backend."
           )
       except ImportError:
           logger.warning(
               "Failed to import 'flag_gems', falling back to default implementations."
           )
       except Exception as e:
           logger.warning(
               f"Failed to enable 'flag_gems': {e}, falling back to default implementations."
           )
   ```

3. **集群各节点配置环境变量**
   在启动集群服务前，确保所有计算节点的环境变量中均声明了启用标识：

   ```shell
   export USE_FLAGGEMS=1
   ```

4. **启动服务并确认算子接管**
   正常拉起分布式服务。检查各节点 Worker 的启动日志，确认输出包含 ATen 算子覆盖与 vLLM 补丁注入信息：

   ```none
   Overriding a previously registered kernel for the same operator and the same dispatch key
   operator: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
   dispatch key: CUDA
   ...
   Patched RMSNorm.forward_cuda with FLAGGEMS custom_gems_rms_forward_cuda
   ```

   日志出现上述记录即表明集群内所有节点的工作进程均已成功挂载 *FlagGems* 算子。

{{% /steps %}}
