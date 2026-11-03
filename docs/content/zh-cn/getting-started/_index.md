---
title: 开始使用
weight: 30
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
# Get Started With FlagGems

## 1. Install FlagGems
-->
# 快速上手 FlagGems

## 1. 安装 FlagGems {#install-flaggems}

*FlagGems* 支持作为纯 Python 包开箱使用，也支持编译构建 C++ 原生扩展以获取最佳的运行时调度性能。详细的安装步骤与编译选项请参阅[安装指南](install.md)。

## 2. 验证安装环境

在安装 `flag_gems` 及其依赖后，建议按照以下步骤验证基础环境是否就绪：

### 2.1 验证 PyTorch 环境

首先确认当前 Python 环境中能够正确导入 `torch` 并识别底层加速硬件：

```shell
python -c "import torch; print(torch.__version__)"
```

如果您使用的是[非 NVIDIA 硬件平台](../usage/non-nvidia.md)，通常需要安装硬件厂商配套的 PyTorch 硬件加速插件。例如在摩尔线程（Moore Threads）GPU 平台上，可运行：

```shell
python -c "import torch_musa; print(torch_musa.__version__)"
```

### 2.2 验证 Triton 编译器环境

接下来验证 `triton` 编译器的安装状态：

```shell
python -c "import torch, triton; print(triton.__version__)"
```

> [!WARNING]
> **注意事项**
>
> 在很多国产及异构硬件平台上，厂商定制的 `triton` 包名称可能与社区原生包完全相同（例如均命名为 `triton`）。请务必**核对版本来源**，确保实际导入的是对应硬件厂商所适配的 Triton 或 FlagTree 编译器分支。

### 2.3 验证 FlagGems 安装

最后验证 `flag_gems` 库的导入与版本信息：

```shell
python -c "import flag_gems; print(flag_gems.__version__)"
```

## 3. 开始使用 FlagGems

*FlagGems* 提供了灵活便捷的接入方式：

### 3.1 全局无侵入式启用

在程序入口处执行 `flag_gems.enable()`，后续所有支持的 PyTorch 算子都会被自动替换为优化实现：

```python
import flag_gems
import torch

# 全局启用 FlagGems 算子加速
flag_gems.enable()

# 用户模型与业务逻辑代码无需改动 ...
```

### 3.2 局部上下文生效

利用 `with flag_gems.use_gems():` 上下文管理器，可以仅在关键热点代码段或性能基准测试中启用加速：

```python
import flag_gems
import torch

M, N, K = 1024, 1024, 1024
A = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
B = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)

# 仅在此代码块内使用 FlagGems 算子
with flag_gems.use_gems():
    C = torch.mm(A, B)
```

### 3.3 直接显式调用

您也可以绕过 PyTorch 的 ATen 分发链路，直接调用 `flag_gems.ops` 或 `flag_gems.fused` 下的高性能算子实现：

```python
import flag_gems
import torch

a = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
b = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
c = flag_gems.ops.mm(a, b)

# 融合算子调用示例（MoE 对齐切块）
num_tokens = 4096
block_size = 64
topk = 2
num_experts = 128

topk_ids = torch.randint(
    low=0,
    high=num_experts,
    size=(num_tokens, topk),
    device=flag_gems.device,
    dtype=torch.int32,
)
sorted_ids, expert_ids, num_tokens_post_pad = (
    flag_gems.fused.moe_align_block_size.moe_align_block_size(
        topk_ids, block_size, num_experts
    )
)
```

更多精细控制选项（如白名单筛选、日志追踪、多卡分布式适配等），请继续阅读[使用指南](../usage/_index.md)。
