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
为了解决这一问题，*FlagGems* 提供了一套 C++ 运行时解决方案，用 C++
语言来实现算子的封装逻辑、注册机制和运行时管理，与此同时仍然复用下层的 Triton 内核来完成实际计算。
这种方法能够保留 Triton 中内核级别的效率，同时大幅降低 Python 语言相关的性能开销，
使得用户能够与底层的 CUDA 工作流进行更为紧密的集成，提升整体的推理性能。

## 1. 架构

*FlagGems* 中 C++ 封装的算子构建于
[`libtriton_jit`](https://github.com/flagos-ai/libtriton_jit) 之上。
`libtriton_jit` 是一个多后端的 Triton JIT C++ 运行时，它在 C++ 中重新实现了
Triton JIT 运行时逻辑（参数特化、内核缓存和发射），
而实际的 kernel 编译仍然委托给上游的 Triton 编译器完成。

在这一整套技术栈中：

- Triton 内核（`*.py`）仍然是设备端计算的唯一源头；
- `libtriton_jit` 负责 JIT 特化、内核缓存以及特定后端的 kernel 发射，
  目前已支持 **NVIDIA（CUDA）**、**摩尔线程（MUSA）**、**华为昇腾（NPU）**
  与**天数智芯（IX）** 四种后端；
- *FlagGems* 的 C++ 封装算子（位于 `lib/` 目录下，例如 `rms_norm.cpp`、`mm.cpp`）
  以 C++ 实现张量元数据处理、形状/类型提升以及参数准备，最后通过
  `libtriton_jit::TritonJITFunction` 调用 Triton 内核；
- 在封装算子之上，*FlagGems* 还提供两个面向 Python 的扩展模块
  （`src/flag_gems/csrc/cstub.cpp` 与 `src/flag_gems/csrc/aten_patch.cpp`）
  以及一个可安装的 C++ 库目标（`FlagGems::operators`），
  它们共同把同一份 C++ 封装通过**四种不同的调用方式**暴露给上层用户
  （详见[§3 C++ 算子的四种调用方式](#3-c-算子的四种调用方式)）。

无论最终走的是哪一种调用方式，**包装器（wrapper）本身的逻辑都在 C++ 中执行**
（张量元数据处理、参数类型与特化判定、kernel 缓存查找与启动参数准备等），
这部分 Python 开销被无条件消除，同时底层计算仍然复用相同的 Triton 内核。

而 **PyTorch dispatcher 的开销**是否也能被省掉，则取决于所选的调用路径：

- 经过 dispatcher 的路径（`torch.ops.flag_gems.*`、以及 ATen 直接替换）
  仍然要付出常规的 dispatcher 代价；不过由于 dispatcher 背后挂的是
  C++ 实现而不是 Python 包装器，整体的 boxed-call 开销仍然比纯 Python
  自定义算子小得多。
- 绕过 dispatcher 的路径（`c_operators` pybind 模块、原生 C++ API）
  则完全没有 dispatcher 开销；其中原生 C++ API
  还会在调用路径上进一步消除 Python 解释器本身的参与。

各种方式的具体权衡参见[§3 C++ 算子的四种调用方式](#3-c-算子的四种调用方式)。

## 2. 安装与启用

想让 C++ wrapper **全量生效**，需要同时满足以下两步：

{{% steps %}}

1. **编译安装时：打开 C++ 扩展并以 Release 构建**

   从源码安装，至少传入 `-DFLAGGEMS_BUILD_C_EXTENSIONS=ON` 与
   `-DCMAKE_BUILD_TYPE=Release`（后者保证 FlagGems 与随同构建的
   `libtriton_jit` 都按目标平台开启优化，否则 wrapper 会明显变慢）：

   ```shell
   CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DCMAKE_BUILD_TYPE=Release" \
   pip install -v -e .
   ```

   > [!NOTE]
   > 如果上述命令构建失败，可以尝试加上 `--no-build-isolation`，
   > 让 pip 复用当前环境中已安装的 PyTorch 以及
   > `requirements_<backend>.txt` 预装的构建依赖。

   其他可选参数：

   - `-DFLAGGEMS_BACKEND=<CUDA|IX|MUSA|NPU>`：选择目标后端（默认 `CUDA`）；
   - `-DFLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP=ON`：编译 `add`/`div`/`fill`
     这几个 pointwise dynamic 算子；
   - `-DFLAGGEMS_BUILD_CTESTS=ON`：编译 `ctests/` 下的 GTest 用例
     （验证 §3.4 原生 C++ API 的唯一手段）；
   - `-DFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON -DTritonJIT_ROOT=<path>`：
     使用外部预装的 `libtriton_jit`。

   完整的各后端示例与 `libtriton_jit` 细节，参见
   [安装指南](../../getting-started/install/#install-from-source)。

1. **运行时：`export USE_C_EXTENSION=1`**

   只装上 C++ 扩展**还不够**。`src/flag_gems/config.py`
   把一部分上层行为挂在这个环境变量上——不设它，下面这些路径会悄悄走 Python 回退：

   | 路径 / 行为                                | 编译好就可用 | 还需 `USE_C_EXTENSION=1` |
   | ------------------------------------------ | :----------: | :----------------------: |
   | §3.1 `torch.ops.flag_gems.*`               | ✅           | —                        |
   | §3.3 `c_operators` pybind                  | ✅           | —                        |
   | §3.2 ATen 直接替换（`aten_patch`）         | ❌           | ✅                       |
   | `flag_gems.enable()` 的 C++ 分支           | ❌           | ✅                       |
   | `GemsRMSNorm` 等 `nn.Module` 的 C++ 分支   | ❌           | ✅                       |

   所以正常使用请：

   ```shell
   export USE_C_EXTENSION=1
   ```

1. **快速验证**

   下面这个片段可以一次性检查 Python 侧能看到的三条路径：

   ```python
   import torch
   import flag_gems
   from flag_gems import c_operators, aten_patch
   from flag_gems.config import has_c_extension, use_c_extension

   assert has_c_extension, "C++ extension was not built"
   assert use_c_extension, "please `export USE_C_EXTENSION=1`"
   assert hasattr(torch.ops.flag_gems, "mm"), "§3.1 torch.ops.flag_gems.* not registered"
   assert aten_patch.get_registered_ops(), "§3.2 no ATen op has been replaced"
   _ = c_operators.mm                                                 # §3.3
   ```

   §3.4 原生 C++ API 无法从 Python 观察；要验证它，需要编译时加
   `-DFLAGGEMS_BUILD_CTESTS=ON`，然后运行 `ctest`：

   ```shell
   BUILD_DIR=$(ls -d build/*/ | head -n 1)
   ctest --test-dir "${BUILD_DIR}" --output-on-failure
   ```

   > [!NOTE]
   > 手动单独跑某个 test 二进制时（例如
   > `"${BUILD_DIR}/ctests/test_triton_mm"`），需要
   > `export FLAGGEMS_SOURCE_DIR=$(pwd)/src/flag_gems`，
   > C++ 运行时才能找到 Triton 内核的 `.py` 源文件；
   > 通过 `ctest` 跑会自动设置。

1. **典型使用场景**

   在前两步都完成的前提下，下面两种写法都会**自动优先使用 C++ 封装算子**，
   无需修改调用点：

   - **补丁模式 `flag_gems.enable()`**：monkey-patch `torch.*` 接口，
     当 `use_c_extension == True` 时 patch 后的函数走
     `torch.ops.flag_gems.*`（§3.1），否则回退到 Python 实现。
   - **用 FlagGems 的 `nn.Module` 类搭模型**：例如
     [`flag_gems.modules.GemsRMSNorm`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/modules/normalization.py)，
     内部已经内置「C++ 可用就走 `torch.ops.flag_gems.*`，否则走 Python」
     的分支判断。参考
     [`gems_rms_forward` 的实现](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/modules/normalization.py#L33)。

{{% /steps %}}

## 3. C++ 算子的四种调用方式

C++ 扩展构建完成之后，同一份底层 C++ 封装算子实际上可以通过**四种**
不同的方式被调用。这几种方式面向不同的使用场景，并且分别有着不同的 dispatcher 开销。

### 3.1 通过 `torch.ops.flag_gems.*`（自定义算子名字空间）

所有 C++ 封装算子都会在 `src/flag_gems/csrc/cstub.cpp` 中通过
`TORCH_LIBRARY(flag_gems, m)` 注册为 PyTorch 的**自定义算子（custom op）**，
归入 `flag_gems` 名字空间。你可以在 Python 中显式地调用它们，
从而绕过所有 patch 逻辑与 Python 侧的回退路径：

```python
output = torch.ops.flag_gems.fused_add_rms_norm(...)
out    = torch.ops.flag_gems.mm(a, b)
```

### 3.2 通过 ATen 直接替换（在 dispatcher 层面透明接管 `torch.*`）

对于其中的一部分算子，*FlagGems* 还会在 `src/flag_gems/csrc/aten_patch.cpp`
中使用 `TORCH_LIBRARY_IMPL(aten, <dispatch_key>, m)` 把 C++ 实现
**直接注册到 `aten::` 名字空间**。具体的 dispatch key 按后端选择：

- 对于 NVIDIA CUDA 与天数智芯（IX）使用 `CUDA`；
- 对于华为昇腾（NPU）与摩尔线程（MUSA）使用 `PrivateUse1`。

由于注册是直接发生在 PyTorch 的 dispatcher 中的，
在受支持的设备上调用标准的 PyTorch API（例如 `torch.nonzero(x)`、
`x.copy_(y)`）时，会**透明地派发到 FlagGems 的 C++ 实现**，
完全不需要在 Python 层做 monkey patch。
这是对一个已有模型做加速时门槛最低的一条路径。

> [!NOTE]
> 由于 `TORCH_LIBRARY_IMPL` 在模块导入时就会执行，
> 这种方式所替换的算子集合在构建期就已经确定，
> 目前还不支持在运行时对单个算子作启用/禁用控制。

### 3.3 通过 `c_operators` pybind 模块（直连、不经 dispatcher）

同样的一组 C++ 封装算子在 `src/flag_gems/csrc/cstub.cpp` 中还会通过
`PYBIND11_MODULE(c_operators, …)` 导出为一个 Python 扩展模块：

```python
from flag_gems import c_operators

out = c_operators.mm(a, b)
c_operators.fused_add_rms_norm(input, residual, weight, eps)
```

这一路径**完全绕开了 PyTorch 的 dispatcher**，因此是从 Python 调用
FlagGems C++ 算子开销最低的方式。它最适合用在延迟非常敏感的
microbenchmark，或者内循环中 dispatcher 开销已经能被测量出来的场景。

### 3.4 通过原生 C++ API（`flag_gems::` 函数与 GTest）

每一个封装算子本身都是一个普通的 C++ 函数，位于 `flag_gems::` 名字空间，
在 `include/flag_gems/operators.h` 中声明，并随安装出的 CMake 目标
`FlagGems::operators` 一起发布。下游的 C++ 代码可以直接链接该目标并调用：

```cpp
#include "flag_gems/operators.h"

at::Tensor c = flag_gems::mm_tensor(a, b);
at::Tensor y = flag_gems::rms_norm(x, weight, eps);
```

仓库中 `ctests/` 下的 GTest 用例（例如 `ctests/test_triton_mm.cpp`）正是以这种方式调用 FlagGems 的算子的。
当你希望把 FlagGems 嵌入到一个非 Python 的 C++ 应用中，或者需要写 C++ 单元测试时，这就是合适的路径。

### 小结

| 调用方式                         | 入口                              | 经 dispatcher |
| -------------------------------- | --------------------------------- | ------------- |
| `torch.ops.flag_gems.*`          | `TORCH_LIBRARY(flag_gems, …)`     | 是            |
| ATen 直接替换                    | `TORCH_LIBRARY_IMPL(aten, …)`     | 是            |
| `flag_gems.c_operators` pybind   | `PYBIND11_MODULE(c_operators, …)` | 否            |
| 原生 C++ API                     | `operators.h` 中的 `flag_gems::*` | 否            |

<!--
## Reference: Currently supported C++-wrapped operators
-->
## 参考：目前支持的 C++ 封装的算子

以下算子目前在 *FlagGems* 中提供了 C++ 封装实现。

- `add`（pointwise dynamic C++）
- `div`（pointwise dynamic C++）
- `fill`（pointwise dynamic C++）
- `addmm`
- `mm`
- `bmm`
- `cat`
- `contiguous`
- `copy`
- `embedding`
- `exponential_`
- `zeros`
- `argmax`
- `max`
- `sum`
- `softmax`
- `sort`
- `topk`
- `nonzero`
- `rms_norm`
- `fused_add_rms_norm`
- `rotary_embedding`
- `flash_attn_varlen_func`
- `reshape_and_cache_flash`
- `rwkv_mm_sparsity`
- `rwkv_ka_fusion`

> [!NOTE]
> 标注为 *pointwise dynamic C++* 的算子仅在启用 CMake 选项
> `-DFLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP=ON` 时才会被编译；
> 详情参阅[安装指南](../../getting-started/install/#install-from-source)。

<!--
We are actively expanding this list as part of our ongoing performance optimization work.
-->
作为持续性能优化工作的一部分，我们一直在努力扩大这一列表。
