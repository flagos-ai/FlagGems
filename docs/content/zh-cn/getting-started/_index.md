---
title: 开始使用
weight: 30
---

<!--
# Get Started With FlagGems

## 1. Install FlagGems
-->
# 开始使用 FlagGems

## 1. 安装 FlagGems {#install-flaggems}

<!--
*FlagGems* can be installed either as a pure python package
or a package with C-extensions for better runtime performance.
See [installation](/FlagGems/getting-started/installation/) for some detailed guidance
on different installation options.
-->
*FlagGems* 可以以纯 Python 包的形式安装，也可以附带 C++ 扩展特性来安装，
以实现更好的运行时性能。你可以参阅[安装指南](/FlagGems/getting-started/installation)
的说明，了解不同安装方式、选项的详细指令。

<!--
## 2. Verify the installation

After having installed the `flag_gems` package and its dependencies,
you may want to verify if they work as expected.
-->
## 2. 检查安装状态

在安装了 `flag_gems` 包及其所依赖的其他软件包之后，
你可能希望检查这些软件包是否能够正常工作。

<!--
### 2.1 Verify PyTorch environment

The first thing you want to check is that you can import
`torch` in your working environment:
-->
### 2.1 检查 PyTorch 环境

你要执行的第一项检查是确认能够在你的工作环境中导入 `torch` 软件包：

```shell
python -c "import torch; print(torch.__version__)"
```

<!--
If you are using a [non-NVIDIA platform](/FlagGems/usage/non-nvidia/),
you may have a PyTorch plugin that is provided by the backend vendor.
You can perform a similar verification against this plugin.
For example, on a MooreThreads GPU platform, you can verify if the plugin
works using the following command:
-->
如果你所使用的是一个[非 NVIDIA 的平台](/FlagGems/usage/non-nvidia/)，
你可能需要安装一个由后端硬件厂商所提供的 PyTorch 插件。
你也可以针对这一插件执行类似的验证操作。
例如，在一个使用摩尔线程 GPU 的平台上，你可以使用下面的命令来检查
是否插件能够正常工作：

```shell
python -c "import torch_musa; print(torch_musa.__version__)"
```

<!--
### 2.2 Verify the Triton setup

The next verification is against the `triton` package.
You can check if `triton` is working as expected using the following
command:
-->
### 2.2 检查 Triton 安装

接下来的一项检查是针对 `triton` 软件包的。
你可以使用下面的命令来检查 `triton` 包是否可以正常使用。

```shell
python -c "import torch, triton; print(triton.__version__)"
```

<!--
Note that if you are using a [non-NVIDIA platform](/FlagGems/usage/non-nvidia/),
you have to consult your platform vendor to see if they have
a customized version.
-->
如果你所使用的是一个[非 NVIDIA 的平台](/FlagGems/usage/non-nvidia/)，
你可能需要咨询的平台供应商，了解他们是否提供了一个定制版本。

<!--
> [!WARNING]
> **Warning**
>
> Usually, the vendor-customized version of `triton` has the same name
> with the upstream package. Please **double confirm** that you are using
> the correct package before proceeding.
-->
> [!WARNING]
> **警告**
>
> 通常，厂商定制的 `triton` 软件包与上游社区获得的软件包同名。
> 在进入下一步之前，你需要**反复确认**自己所使用的包是正确的版本。

<!--
### 2.3 Verify the FlagGems installation

You can do a similar verification for the `flag_gems` package,
using the following command:
-->
### 2.3 检查 FlagGems 的安装

针对 `flag_gems` 包，你也可以使用下面的命令执行类似的检查：

```shell
python -c "import flag_gems; print(flag_gems.__version__)"
```

<!--
## 3. Start using FlagGems

You can enable the accelerated operators from `flag_gems` in many ways.
The following code snippet enables `flag_gems` globally:
-->
## 3. 开始使用 FlagGems

你可以用多种不同方式来启用 `flag_gems` 所提供的加速算子。
下面的代码段在全局范围内启用 `flag_gems`：

```python
import flag_gems

flag_gems.enable()

# 你自己的代码 ...
```

<!--
You can also enable `flag_gems` in a specific context for a certain
section of your code, as shown below:
-->
你也可以针对自己代码中的某一部分，在特定的上下文中启用 `flag_gems`，
如下例所示：

```python
# 你的代码 ...

# 在特定上下文中启用 flag_gems
with flag_gems.use_gems():
    # 在上下文中使用加速的算子
    # ...
```

<!--
For example:
-->
例如：

```python
import torch
import flag_gems

M, N, K = 1024, 1024, 1024
A = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
B = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
with flag_gems.use_gems():
    C = torch.mm(A, B)
```

也可以绕过 PyTorch，直接调用 `flag_gems.ops` 库中的算子。

```python
import torch
from flag_gems import ops
import flag_gems

a = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
b = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
c = ops.mm(a, b)
```

对于融合算子，也可以直接从 `flag_gems.fused` 导入并使用：

```python
import torch
import flag_gems
from flag_gems.fused.moe_align_block_size import moe_align_block_size

# moe_align_block_size 的使用示例
num_tokens = 4096
topk = 2
num_experts = 128

# topk_ids 表示每个 token 对应的 expert 索引
topk_ids = torch.randint(
    low=0,
    high=num_experts,
    size=(num_tokens, topk),
    device=flag_gems.device,
    dtype=torch.int32,
)

sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
    topk_ids=topk_ids,
    block_size=128,
    num_experts=num_experts,
)
```

<!--
Check the [](/FlagGems/usage/) section for more detailed
documentation on the various usage patterns about *FlagGems*.
-->
你可以查阅[使用指南](/FlagGems/zh-cn/usage/)一节中的详细文档，
了解 *FlagGems* 的多种使用模式。
