---
title: 逐点动态（Pointwise Dynamic）算子
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
# Pointwise Dynamic Operators

## 1. Pointwise Operations
-->
# 逐点动态算子（Pointwise Dynamic）

## 1. 逐点操作 {#pointwise-operations}

逐点算子（Pointwise Operators）本身天然具备高度的并行性。大多数并行编程教程通常都以两个连续向量的点加（Pointwise Addition）作为入门示例。对于 [Triton 官方教程中的 `vector_add`](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py)，其任务划分方案非常直接：每个线程块（CTA，Cooperative Thread Array）分别从输入向量中读取一段连续切片，并在计算后写入输出向量的对应连续区间。

然而，在深度学习框架的实际生产场景中，逐点算子的工况要复杂得多：

- **内存非连续性**：输入张量可能并非连续内存布局（Non-contiguous）；即使在物理内存上连续，其排布也可能非行优先（Non row-major），或者张量是非稠密的（Non-dense）、甚至是存在内部跨步重叠（Overlapping）。
- **动态任意维度**：各输入张量可能具有任意的维度与形状，无法简单视作形状相同的一维连续向量。
- **广播语义（Broadcasting）**：输入张量之间可能存在维度不一致但满足广播机制的情况。
- **混合参数类型**：输入可能同时包含张量（Tensor）与标量/非张量参数（如浮点系数、布尔标志等）。
- **模板与样板代码冗余**：不同逐点算子在索引计算与内存寻址逻辑上高度共通，如果为每个算子重复编写低级调度与寻址逻辑，维护成本极高。

为此，*FlagGems* 提出了一套基于代码自动生成的通用架构，其核心设计原则包括：

- **规避冗余内存搬运**：逐点算子通常是典型的访存密集型（Memory-bound）算子，严禁通过显式拷贝使张量连续化，而应直接在物理内存上进行跨步寻址。
- **通用灵活的参数支持**：全面支持任意秩（Rank）、大小（Size）、步长（Stride）的张量，原生支持广播机制与张量/标量混合参数。
- **公共逻辑高度抽象与模板化**：将复杂的坐标映射与参数打包提炼为共享的代码生成模板，最大限度减少样板代码。
- **硬件后端可配置**：共享的基础设施具备良好的可配置性，便于在不同硬件后端之间平滑迁移与调优。

基于以上原则，*FlagGems* 提供了 `@pointwise_dynamic` 装饰器：它为逐点算子提供了统一的高级封装，并能根据输入的形状、步长和运算配置，自动生成高度优化的底层 Triton Kernel 及其外层分发代码。

## 2. 算子代码生成机制 {#code-generation}

`@pointwise_dynamic` 的基本用法是修饰一个带有返回值的 `triton.jit` 函数，该函数定义了从输入到输出的标量级计算映射关系。
这个 JIT 函数类似于 CUDA 中的 `__device__` 函数（即设备端计算逻辑 Payload）。`@pointwise_dynamic` 会自动为其生成一个负责全局内存加载、索引映射与存储回写的 Triton Kernel，类似于 CUDA 中的 `__global__` 入口函数。

为了支持任意秩、形状与步长的输入张量，代码生成器会将输出张量的形状（即逐点计算的任务空间 Task Space）以及各输入张量在每个维度上的步长（Strides）解包为整数并传递给底层 Kernel。
在 Triton Kernel 内部，系统将一维任务 ID 映射为任务空间的多维索引（Multi-index），随后根据各张量在各维度的步长将其转化为物理内存偏移量（Memory Offset）。

例如，对于形状均为 `(2, 3)` 的两个张量相加，任务空间即为 `(2, 3)`。线性 Task ID 为 `4` 时，其映射的任务空间坐标为 `(1, 1)`：
- 若左操作数 `lhs` 的步长为 `(3, 1)`，对应的内存物理偏移为 `1 * 3 + 1 * 1 = 4`；
- 若右操作数 `rhs` 的步长为 `(1, 2)`，对应的内存物理偏移为 `1 * 1 + 1 * 2 = 3`。

针对满足广播语义的张量，系统会首先通过广播推导出统一的任务空间形状，并将各张量基于该形状生成对应的跨步视图（Strided View），共享底层物理存储的同时赋予新的逻辑步长。

> [!TIP]
> 开发者编写被修饰的 JIT 函数时，通常可将其当作标量计算表达式来编写。但需要注意：生成的外层 Kernel 在调用该函数时传入的实际参数是向量化的 `tl.tensor`。因此切勿在控制流（如 `if` 或 `while`）中直接对输入张量做布尔条件判断，因为 Triton 严禁非标量张量充当分支条件。

## 3. 元数据推导（Metadata Computation） {#metadata-computation}

各类逐点算子在执行前都需要完成相似的元数据校验与推导。*FlagGems* 将该逻辑统一封装在 `PointwiseDynamicFunction` 的公共入口中，主要涵盖：

- **形状推断（Shape Inference）**：根据输入张量的形状进行广播规则运算，确定输出形状；
- **输出内存布局推导（Layout Inference）**：推导最契合硬件访存效率的输出张量步长顺序；
- **类型提升（Type Promotion）**：依据预设的类型转换策略自动推导输出张量的数据类型（Dtype）；
- **设备推断（Device Inference）**：确定输出张量所在设备及待启动 Kernel 的硬件上下文；
- **输出内存分配（Output Allocation）**：根据推导出的形状与类型在显存中申请输出 Tensor；
- **任务空间降维（Rank Reduction）**：当输入输出张量在物理内存上均为稠密、无重叠且跨步规则一致时，系统会自动将多维任务空间压缩为 1 维，大幅降低索引计算开销。

`PointwiseDynamicFunction` 同样支持用户传入预先分配好的输出张量。若提供了预分配张量，系统将直接复用其内存空间，并在执行前严格校验其形状、数据类型与设备一致性。

## 4. 动态缓存与分发（Caching and Dispatching） {#caching-and-dispatching}

使用 `@pointwise_dynamic` 装饰后，返回的是一个 `PointwiseDynamicFunction` 代理对象。该对象会根据运行时解析到的任务空间秩（Rank）动态编译并缓存对应的底层 Python 调度模块。
分发命中仅与**任务空间的秩**相关，而与具体的张量形状大小解耦，因此几乎不会因 Shape 变化引起额外的重复编译。

## 5. `@pointwise_dynamic` 装饰器使用指南

### 5.1 基础用法

使用 `@pointwise_dynamic` 修饰逐点算子后，开发者无需再手动编写张量寻址、切块平铺、广播对齐、类型提升等繁琐的样板代码。只需声明核心数学运算逻辑即可：

```python
@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)
```

### 5.2 区分张量与标量参数

默认情况下，`@pointwise_dynamic` 将所有参数均视作 Tensor 并生成加载/存储指令。当参数中包含标量（如系数因子、阈值）时，可通过 `is_tensor` 列表指定各参数属性：

```python
@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0, "DEFAULT")],
)
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


a = torch.randn(128, 256, device="cuda")
b = torch.randn(256, device="cuda")
add_func(a, b, 0.2)
```

对于非张量参数，还可在 `dtypes` 中显式指定其标量类型；对于张量参数，`dtypes` 中的占位项将被忽略，由 Triton 在运行时动态推断。

### 5.3 输出数据类型与类型提升策略 {#output-dtypes}

通过 `promotion_methods` 参数可以精细声明输出张量类型的推导规则。与硬编码输出类型相比，声明规则能够自适应任意输入类型的组合。

`promotion_methods` 接受一个元组列表（每个元组对应一个输出参数），每个元组由相关输入参数的索引与类型提升规则组成。常见策略包括：

- `DEFAULT`：标准数值计算类型提升规则，适用于大部分算术运算；
- `NO_OPMATH`：直接保持输入类型不变，适用于纯数据搬运或非数值变换（如 `cat`、`where`）；
- `INT_TO_FLOAT`：强制将整型提升为对应浮点类型（如 `sin`、`cos`）；
- `ALWAYS_BOOL`：输出始终提升为布尔类型（如逻辑比较 `eq`、`lt`）；
- `COMPLEX_TO_FLOAT`：将复数类型转换为浮点类型（如 `abs`）；
- `BOOL_TO_LONG`：将布尔类型转换为长整型（如 `pow`）。

| 提升规则枚举       | 典型算子示例                 |
| :----------------- | :--------------------------- |
| `DEFAULT`          | `add`、`mul`                 |
| `NO_OPMATH`        | `where`、`nextafter`、`cat`  |
| `INT_TO_FLOAT`     | `sin`、`cos`                 |
| `ALWAYS_BOOL`      | `eq`、`ge`                   |
| `COMPLEX_TO_FLOAT` | `abs`                        |
| `BOOL_TO_LONG`     | `pow`                        |

### 5.4 多输出算子支持 {#number-of-outputs}

当算子需要返回多个张量时，需显式配置 `num_outputs`，以便代码生成器构建多个写回通道：

```python
@pointwise_dynamic(
    promotion_methods=[
        ((0, 1), "DEFAULT"),
        ((0, 1), "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def polar_kernel(abs, angle):
    real = abs * tl.cos(angle)
    imag = abs * tl.sin(angle)
    return real, imag
```

## 6. 原地操作与高级调用

### 6.1 原地操作（In-place）与输出重定向

`PointwiseDynamicFunction` 支持通过关键字参数传递预分配的输出张量。为了明确区分输入与输出，约定**所有输入参数按位置传递，所有输出参数通过关键字参数 `out{index}` 传递**。

利用此特性可以极简地实现 In-place 算子：

```python
@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


# 原地加法实现
def add_(A, B, *, alpha=1):
    return add_func(A, B, alpha, out0=A)


# 显式指定输出目标张量
def add_out(A, B, *, alpha=1, out=None):
    return add_func(A, B, alpha, out0=out)
```

### 6.2 手动实例化（Manual Instantiation） {#manual-instantiation}

在某些特殊算子中，我们希望跳过通用的元数据推导逻辑，由外部手动构造好带有负跨步（Negative Strides）或偏移指针的张量视图，直接将其作为逐点操作执行。
此时可通过调用 `instantiate(ndim)` 方法针对指定的张量秩获取编译后的底层函数。

例如，`flip`（张量翻转）算子在数学上并非标准逐点算子，但通过对输入张量构造负步长视图并调整数据起始指针后，即可等价转化为逐点数据复制：

```python
@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    strides = list(A.stride())
    flip_dims_b = [False for _ in A.stride()]
    for dim in dims:
        assert -A.dim() <= dim < A.dim(), (
            f"Dimension out of range (expected [{-A.dim()}, {A.dim()-1}], got {dim})"
        )
        assert not flip_dims_b[dim], f"dim {dim} appears multiple times"
        flip_dims_b[dim] = True

    offset = 0
    for i in range(len(flip_dims_b)):
        if flip_dims_b[i] and A.size(i) > 1 and A.stride(i) != 0:
            offset += strides[i] * (A.shape[i] - 1)
            strides[i] = -strides[i]

    if A.numel() <= 1:
        return A.clone()

    out = torch.empty_like(A)
    # 构造翻转后的跨步视图
    flipped_A = StridedBuffer(A, strides=strides, offset=offset)

    # 针对当前维度直接获取专用 Kernel 并调用
    kernel = copy_func.instantiate(A.ndim)
    kernel(flipped_A, out0=out)
    return out
```
