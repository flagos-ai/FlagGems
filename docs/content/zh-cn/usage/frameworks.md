---
title: 框架集成
weight: 60
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


<!--
# Integration with Popular Frameworks
-->
# 主流深度学习框架集成

为了便于开发者在真实生产环境中无缝使用 *FlagGems*，本章提供了与业界主流深度学习训练和推理框架的集成示例。这些集成仅需极少量的代码适配，即可完整保留原始工作流结构。

完整端到端示例代码可参阅代码仓库中的 [`examples/`](https://github.com/flagos-ai/FlagGems/tree/master/examples) 目录。

## 1. Hugging Face Transformers

将 *FlagGems* 与 Hugging Face 的 [`transformers`](https://github.com/huggingface/transformers) 库集成非常轻量且直观。在模型推理时，只需通过上下文管理器即可激活加速，无需修改模型定义或分词器（Tokenizer）逻辑：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import flag_gems

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("sharpbai/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("sharpbai/Llama-2-7b-hf")

# 转移到目标设备并设置为评估模式
device = flag_gems.device
model.to(device).eval()

# 准备输入并在 FlagGems 加速上下文中执行文本生成
inputs = tokenizer("Prompt text", return_tensors="pt").to(device=device)
with flag_gems.use_gems():
    output = model.generate(**inputs, max_length=100, num_beams=5)
```

在此模式下，文本生成链路中涉及的所有兼容算子均会被透明替换并加速。更多完整测试代码请参考：
- `examples/model_llama_test.py`
- `examples/model_llava_test.py`

## 2. vLLM 高性能推理引擎

[vLLM](https://github.com/vllm-project/vllm) 是业界广泛采用的高吞吐 LLM 服务推理引擎，核心特性包括 PagedAttention（分页注意力机制）、Continuous Batching（连续批处理）等。

> [!IMPORTANT]
> **重要迁移通知与开发者贡献指引**
>
> 原 `src/flag_gems/fused` 下的 vLLM 专有算子已正式迁移至独立的 [**FlagGems-vllm**](https://github.com/flagos-ai/FlagGems-vllm) 仓库进行专门维护与演进。
> 如果您需要对 vLLM 相关的算子进行特性增强、性能调优或提交 Bug 修复，**请直接向 [FlagGems-vllm](https://github.com/flagos-ai/FlagGems-vllm) 仓库提交 Issue 与 Pull Request**。本仓库（FlagGems）将聚焦于通用 ATen 算子库及基础神经网络组件。

*FlagGems* 可以深度集成至 vLLM 中，既支持替换标准的 PyTorch ATen 算子，也支持接管 vLLM 内部的定制 Kernel：

### 2.1 替换 vLLM 中的标准 PyTorch 算子

对于 vLLM 中调用的通用 PyTorch 算子（如 `add`、`masked_fill` 等），只需在模型初始化前执行 `flag_gems.enable()`，即可全局生效：

```python
import flag_gems

flag_gems.enable()
```

### 2.2 替换 vLLM 内部专有定制算子

为了进一步发挥硬件潜能，*FlagGems* 针对 vLLM 的专有算子提供了热补丁（Monkey Patching）注入接口：

```python
flag_gems.apply_gems_patches_to_vllm(verbose=True)
```

该接口会自动将 vLLM 内部的若干 C++/Triton 算子（如 RMSNorm、RotaryEmbedding、SiluAndMul 等）替换为经过 *FlagGems* 深度调优的实现。启用 `verbose=True` 会打印补丁替换日志：

```none
Patched RMSNorm.forward_cuda with FLAGGEMS custom_gems_rms_forward_cuda
Patched RotaryEmbedding.forward_cuda with FLAGGEMS custom_gems_rope_forward_cuda
Patched SiluAndMul.forward_cuda with FLAGGEMS custom_gems_silu_and_mul
```

### 2.3 vLLM 端到端集成完整示例

```python
from vllm import LLM, SamplingParams
import flag_gems

# 步骤 1：全局启用 ATen 算子加速
flag_gems.enable()

# 步骤 2（可选）：注入 vLLM 内部算子热补丁以获得极致性能
flag_gems.apply_gems_patches_to_vllm(verbose=True)

# 步骤 3：正常启动 vLLM 离线推理或在线 Serving
llm = LLM(model="sharpbai/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.8, max_tokens=128)

outputs = llm.generate(["Tell me a joke."], sampling_params)
print(outputs[0].outputs[0].text)
```

## 3. Megatron-LM 大规模并行训练框架

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 是用于超大规模语言模型分布式预训练的高性能框架。由于其内部执行循环与分布式数据流水线高度耦合，集成 *FlagGems* 时推荐采取更为聚焦的局部注入策略。

### 3.1 推荐集成方式：作用于前向与反向计算

在 Megatron-LM 中，最稳健的做法是在 [`megatron/training.py`](https://github.com/NVIDIA/Megatron-LM/blob/v2.6/megatron/training.py#L347) 的 `train_step` 函数中，使用 `flag_gems.use_gems()` 仅包裹计算核心 `forward_backward_func`：

```python
def train_step(forward_step_func, data_iterator, model, optimizer, lr_scheduler):
    """单步分布式训练循环."""
    args = get_args()
    timers = get_timers()

    # 梯度置零
    if args.DDP_impl == "local" and args.use_contiguous_buffers_in_local_ddp:
        for partition in model:
            partition.zero_grad_buffer()
    optimizer.zero_grad()

    # 使用 FlagGems 加速前向与反向计算步骤
    import flag_gems

    with flag_gems.use_gems():
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            timers,
            forward_only=False,
        )

    # 释放未使用的显存碎片
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()
    ...
```

此方案确保加速仅集中在最耗时的矩阵计算与模型层运算中，同时规避了对分布式数据迭代器与通讯初始化的潜在干扰。

### 3.2 适用范围与注意事项

> [!WARNING]
> Megatron-LM 社区更新迭代活跃，各衍生分支（如 Megatron-Core、各类定制流水线）在架构上可能存在差异，集成时请确认所处分支的执行时序。

若需要将加速扩展至优化器（Optimizer Step）或张量通信预处理等更广泛的环节，可在训练启动脚本中尝试使用全局 `flag_gems.enable()`，建议在正式大规模训练前完成单步数值一致性校验。
非常欢迎大家向 [FlagGems 贡献更多框架集成方案](../contribution/overview.md)！
