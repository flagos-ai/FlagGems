---
title: 基本用法
weight: 20
---

<!--
# Basic Usage
-->
# 基本用法

<!--
To use the operators from the *FlagGems* operator library, import `flag_gems` and enable acceleration
before running your program. You can enable it globally or for a specific code block.
Besides these, you can invoke operators from the `flag_gems.ops` package directly.
-->
要使用 *FlagGems* 算子库中的算子，可以在运行你的程序之前导入 `flag_gems` 并启用加速。
你可以在全局启用 `flag_gems`，也可以针对特定的代码段启用 `flag_gems`。
除此之外，你还可以直接调用 `flag_gems.ops` 包中的指定算子。

<!--
## 1. Global Enablement

To apply *FlagGems* optimizations across your entire program or your interaction session:
-->
## 1. 全局启用

如果希望在你的整个程序中或者整个交互会话期间启用 *FlagGems* 算子，可以执行下面的语句：

```python
import flag_gems

# 全局性地启用 FlagGems 算子
flag_gems.enable()
```

<!--
Once enabled, all supported operators in your code will be replaced automatically
by the optimized *FlagGems* implementations — no further changes needed.
This means the supported `torch.*` / `torch.nn.functional.*` calls will be dispatched
to FlagGems implementations automatically. For example:
-->
一旦启用，你的代码中的所有被支持的算子都会自动替换为 *FlagGems* 中优化过的实现，
除此之外无需其他修改。这意味着 *FlagGems* 所支持的 `torch.*` / `torch.nn.functional.*`
调用都会被自动派发到加速版本的实现。例如：

```python
import torch
import flag_gems

flag_gems.enable()

x = torch.randn(4096, 4096, device=flag_gems.device, dtype=torch.float16)
y = torch.mm(x, x)
```

<!--
## 2. Scoped Enablement

When needed, you can enable *FlagGems* only within a specific code block
using a `with...` statement:
-->
## 2. 指定作用域的启用

在必要的时候，你可以使用 `with... ` 语句仅针对指定的代码块启用 *FlagGems*：

```python
import flag_gems
import torch

# 针对特定操作启用 flag_gems
with flag_gems.use_gems():

    # 这段代码会使用 FlagGems 中被加速的算子
    x = torch.randn(4096, 4096, device=flag_gems.device, dtype=torch.float16)
    y = torch.mm(x, x)
```

<!--
This scoped usage is useful when you want to:

- perform performance benchmarks, or
- compare correctness between implementations, or
- apply acceleration selectively in complex workflows.
-->
限定作用域的用法在以下场景比较有用：

- 对算子作性能基准测试，或者
- 比较不同实现之间的精度，或者
- 在复杂的工作流中有选择地应用加速算子

<!--
## 3. Direct invocation

You can bypass the PyTorch dispatch process and directly invoke operators from
the `flag_gems.ops` package.
-->
## 3. 直接调用

你也可以略过 PyTorch 中的派发过程，直接调用 `flag_gems.ops` 包中的算子：

```python
import torch
from flag_gems import ops
import flag_gems

a = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
b = torch.randn(1024, 1024, device=flag_gems.device, dtype=torch.float16)
c = ops.mm(a, b)
```

对于融合算子，你可以直接从 `flag_gems.fused` 包中导入并使用：

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
## 4. Query Registered Operators

After having enabled *FlagGems*, you can check the operators registered:
-->
## 4. 查询已注册的算子

在启用了 *FlagGems* 之后，你可以检查系统中已经注册的算子列表：

```python
import flag_gems

flag_gems.enable()

# 获取已注册的算子函数名
registered_funcs = flag_gems.all_registered_ops()
print("Registered functions:", registered_funcs)

# 获取已注册算子的主键
registered_keys = flag_gems.all_registered_keys()
print("Registered keys:", registered_keys)
```

<!--
This is useful for debugging or verifying which operators are active.
-->
这一 API 有助于调试或者检查哪些算子在起作用。

<!--
## 5. Advanced Usage

For advanced usage scenarios, check the following related documentation:

- [Selective enablement](/FlagGems/usage/selective/)
- [Using experimental operators](/FlagGems/usage/experimental/)
- [Enabling logging](/FlagGems/usage/debugging/) for debugging
- [Using FlagGems on non-NVIDIA hardware](/FlagGems/usage/non-nvidia/)
- [Using FlagGems in a multi-GPU or distributed environment](/FlagGems/usage/distributed/)
- [Integrating FlagGems with a popular framework](/FlagGems/usage/frameworks/)
- [Building your own models using FlagGems modules](/FlagGems/usage/modules/)
- [Enable pre-tuning for better performance](/FlagGems/usage/tuning/)
- [Using C++ wrapped operators for optimal performance](/FlagGems/usage/cpp/)
-->
## 5. 进阶用法

你可以阅读以下相关文档，了解一些高级的使用场景：

- [选择性地启用算子](/FlagGems/zh-cn/usage/selective/)
- [使用实验性质的算子](/FlagGems/zh-cn/usage/experimental/)
- [启用日志输出](/FlagGems/zh-cn/usage/debugging/)以方便调试
- [在非 NVIDIA 平台上使用 FlagGems](/FlagGems/zh-cn/usage/non-nvidia/)
- [在多 GPU 或分布式环境中使用 FlagGems](/FlagGems/zh-cn/usage/distributed/)
- [将 FlagGems 与某个常用框架集成](/FlagGems/zh-cn/usage/frameworks/)
- [使用 FlagGems 模块来构造自己的模型](/FlagGems/zh-cn/usage/modules/)
- [启用预调优获得更佳性能](/FlagGems/zh-cn/usage/tuning/)
- [使用 C++ 封装的算子获得更好性能](/FlagGems/zh-cn/usage/cpp/)
