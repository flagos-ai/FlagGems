---
title: 使用 C++ 封装的算子
weight: 90
---

<!--
# Using C++-Based Operators for Optimal Performance

Another advanced optimization path with *FlagGems* is the use of *C++ wrapped operators*
for selected operations. While Triton kernels offer reasonably good compute performance,
Triton itself is a DSL implemented in Python. This means that both the operator definitions and
the runtime dispatchers are written in Python, which can introduce **non-trivial overhead**
in latency-sensitive or high-throughput scenarios.
-->
# 使用 C++ 封装的算子获得更好的性能

使用 *FlagGems* 时的另一条高级的优化路径是针对所选的操作使用其中的**C++ 封装的算子**。
尽管 Triton 内核通常能够给出相当不错的计算性能，Triton 本身是使用 Python 实现的 DSL。
这意味着算子的定义以及算子的运行时派发机制都是用 Python 编写的，
因此在延迟非常敏感或者对吞吐要求极为苛刻的场景下会存在**不可忽视的性能开销**。

<!--
To address this, *FlagGems* provides a C++ runtime solution that encapsulates
the operator's wrapper logic, registration mechanism, and runtime management in C++,
while still reusing the underlying Triton kernels for the actual computation.
This approach preseves the kernel-level efficiency from Triton
while significantly reducing Python-related overhead, enabling tighter integration
with low-level CUDA workflows and improving overall inference performance.
-->
为了解决这一问题，*FlagGems* 提供了一套 C++ 运行时解决方案，用 C++ 语言来实现
算子的封装逻辑、注册机制和运行时管理，与此同时仍然复用下层的 Triton 内核来完成实际计算。
这种方法能够保留 Triton 中内核级别的效率，同时大幅降低 Python 语言相关的性能开销，
使得用户能够与底层的 CUDA 工作流进行更为紧密的集成，提升整体的推理性能。

<!--
## 1. Installation

To use the C++ operator wrappers:
-->
## 1. 安装

要使用 C++ 算子封装：

{{% steps %}}

1. <!--Follow the [installation guide](/FlagGems/installation/) to compile
   and install the C++ version of `flag_gems`.-->
   遵从[安装指南](/FlagGems/zh-cn/installation/)中的指令编译、安装带有 C++
   扩展特性的 `flag_gems` 包。

1. <!--Verify that the installation is successful using the following snippet:-->
   使用下面的代码段来验证安装是否成功：

   ```python
   try:
       from flag_gems import c_operators
       has_c_extension = True
   except Exception as e:
       c_operators = None  # 避免在 c_operators 不可用时出现 import 错误
       has_c_extension = False
   ```

   <!--
   If `has_c_extension` is `True`, then the C++ runtime execution path is available.
   -->
   如果 `has_c_extension` 为 `True`，则 C++ 运行时所支持的执行路径是可用的。

1. <!--
   When installed successfully, C++ wrappers will be preferred **in patch mode**.
   When explicitly [building models](/FlagGems/usage/modules/) using modules
   provided by *FlagGems*, they have a higher precedence over their Python
   equivalents as well.
   -->
   安装成功之后，C++ 封装的算子**在补丁模式下**具有更高的优先级。
   当显式使用 *FlagGems* 所提供的模块来[构建模型](/FlagGems/zh-cn/usage/modules/)时，
   C++ 封装的算子也比其对应的 Python 等价实现的优先级更高。

   <!--
   For example, the operator `gems_rms_forward` will by default use the C++ wrapper
   version of `rms_norm`. You can refer to the actual usage in the
   [`normalization.py`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/modules/normalization.py#L33)
   to better understand how C++ wrapped operators can be integrated and invoked.
   -->
   例如，算子 `gems_rms_forward` 默认会使用 C++ 封装版本的 `rms_norm`。
   你可以参考源码库中的
   [`normalization.py`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/modules/normalization.py#L33)
   文件，更好地了解如何集成 C++ 封装的算子以及如何调用它们。

{{% /steps %}}

<!--
## 2. Invoke C++ operators explicitly

If you want to **directly invoke** the C++-wrapped operators, thus bypassing
any patching logics or fall back paths, you can use the `torch.ops.flag_gems`
namespace as shown below:
-->
## 2. 显式调用 C++ 算子

如果你希望**直接调用** C++ 封装的算子，略过打补丁逻辑或者其他回退执行路径，
可以按下面的代码所给的那样，使用 `torch.ops.flag_gems` 名字空间：

```python
output = torch.ops.flag_gems.fused_add_rms_norm(...)
```

<!--
This gives you *precise control* over operator dispatching, which can be beneficial
in some performance-critical contexts.
-->
这种方式能够让你对算子的派发进行更为**精确的控制**，在某些性能很关键的语境中可能很有用。

<!--
## Reference: Currently supported C++-wrapped operators
-->
## 参考：目前支持的 C++ 封装的算子

<!--
| Operator Name        | Description                              |
| -------------------- | ---------------------------------------- |
| `add`                | Element-wise addition                    |
| `bmm`                | Batch Matrix Multiplication              |
| `cat`                | Concatenate                              |
| `fused_add_rms_norm` | Fused addition + RMSNorm                 |
| `mm`                 | Matrix multiplication                    |
| `nonzero`            | Returns the indices of non-zero elements |
| `rms_norm`           | Root Mean Square normalization           |
| `rotary_embedding`   | Rotary position embedding                |
| `sum`                | Reduction across dimensions              |
-->
| 算子名称             | 描述                                     |
| -------------------- | ------------------------- |
| `add`                | 逐元素的加法              |
| `bmm`                | 批量的矩阵乘法            |
| `cat`                | 串接                      |
| `fused_add_rms_norm` | 加法与 RMSNorm 的融合     |
| `mm`                 | 矩阵乘法                  |
| `nonzero`            | 返回非零元素的索引        |
| `rms_norm`           | 均方根归一化              |
| `rotary_embedding`   | 旋转位置编码              |
| `sum`                | 跨维度的降维（规约）      |

<!--
We are actively expanding this list as part of our ongoing performance optimization work.
-->
作为持续性能优化工作的一部分，我们一直在努力扩大这一列表。
