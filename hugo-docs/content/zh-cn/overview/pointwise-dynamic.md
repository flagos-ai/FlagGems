---
title: 逐点动态（Pointwise Dynamic）算子
weight: 30
---
<!--
# Pointwise Dynamic Operators

## Pointwise Operations
-->
# 关于逐点动态算子

## 逐点操作 {#pointwise-operations}

<!--
Pointwise operators are trivial to parallelize.
Most parallel programming guides begin with pointwise addition between 2 contiguous vectors.
For [`vector_add` in Triton](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py),
it is simple to implement a task partitioning schema that each CTA reads a contiguous range
from each input vector and writes to a contiguous range of the output vector.
-->
逐点算子（Pointwise operators）比较容易并行化执行。
大多数并行计算编程指南的开篇都会使用计算两个连续向量的单点加和操作作为示例。
对于 [Triton 语言中的 `vector_add`](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py)
而言，很容易实现一种任务切分模式，让每个**块集群（Cooperative Thread Array，CTA）**
从每个输入向量中读入一个连续的范围，向输出向量中的一个连续范围写出结果。

<!--
However, actual use caes for pointwise operators may be more complicated.

- The input tensors might be contiguous: they may be contiguous in memory but not in a row-major order;
  or they may not be dense; or they may have internal overlapping.
- The input tensors may have arbitrary and/or different number of dimensions.
  It is not always possible to view them as contiguous vectors of the same shape.
- The input tensors may have different but broadcastable shapes;
- The inputs may be a mixture of tensors and non-tensor inputs;
- Different pointwise operators share common logic to compute indices,
  while it is tedious to rewrite them over-and-over for each operator.
-->
但是，逐点算子的实际使用场景可能比这复杂得多。例如：

- 输入的张量可能是连续的；它们可能在内存中是连续的，但逐行遍历时并不连续；
  或者这些张量可能不是稠密张量（dense tensor）；
  又或者这些张量可能具有内部重叠。
- 输入的张量可能具有任意的维数（dimension）且/或不同张量之间的维数互不相同。
  我们并不总能将它们视为形状（shape）相同的连续向量。

<!--
We propose an approach that is based on code generation to solve these problems.
The principles of our design are:

- Pointwise operations are generally memory-bound, so avoid copying tensors to make them contiguous vectors;
- Pointwise operators should support inputs of arbitrary and/or different ranks, sizes, strides,
  broadcasting inputs, mixing tensor and non-tensors;
- Different pointwise operators should share common internal facilities, either as a library,
  or as a template-based code generation mechanism, to reduce boilerplate code.
- The common internal facilities should be configurable for adaption of different backends.
-->
FlagGems 给出的解决方案是基于代码生成来解决这类问题。方案设计的原理包括：

- 逐点运算通常是访存密集类型的计算，因此应该避免通过复制张量来使之成为连续向量。
- 逐点算子要支持任意秩（rank）、大小（size）、步长（stride）的张量，
  支持广播输入（broadcasting input）、混合张量甚至非张量的输入。
- 不同的逐点算子要使用相同的内部设施来减少样板文件代码，无论所指的设施是一个库，
  还是基于模板的代码生成机制，
- 上述共享的内部设施应该是可配置的，方便适配不同类型的后端（设备）。

<!--
The result is a decorator `@pointwise_dynamic`.
It provides a common wrapper for pointwise operator and a mechanism to generate triton kernels
and corresponding wrappers based on the operation and the input configurations.
-->
团队最终给出的解决方案是 `@pointwise_dynamic` 修饰符。
这一修饰符可以为逐点算子提供一个公共的封装层，并提供生成 Triton 内核（kernel）的机制，
基于操作和输入的配置来构造不同的封装层。

<!--
## Code generation

The basic usage of `@pointwise_dynamic` is using it to decorate a `triton.jit` function
that has return value, which is used to map inputs to outputs.
The JIT function is similar to a function with `__device__` declaration specifier,
a function that can be called from device.
We generate a Triton JIT function to call it, which acts like a CUDA kernel
(a function with `__global__` declaration specifier) that loads and stores data at global memory.
-->
## 代码生成  {#code-generation}

`@pointwise_dynamic` 的基本用法是用来修饰一个带有返回值的 `triton.jit` 函数，
修饰符的作用是在输入与输出之间建立映射关系。
JIT 函数与带有 `__device__` 声明说明符的函数类似，这类函数可以从设备上调用。
我们会生成一个 Triton JIT 函数来调用它，将它视为 CUDA 核（带有 `__global__` 声明说明符的函数）
一样的实体，从全局内存中读取数据或者将数据写回全局内存。

<!--
In order to support input tensors of different ranks, shapes and/or  strides,
we pass the shape of the output tensor (which is also the task-space for pointwise operation) ,
and the strides of each tensor at every dimension.
The shape and strides are unpacked and passed to kernel as integers.
Due to the lack of support for tuple as arguments to Triton kernels,
we have to generate different kernels for different number of integers in the shape and strides.
Although Triton supports tuple as arguments since version 3.3, it does not support all operations on tuples
such as indexing, iteration and so on.
-->
为了支持不同秩（rank）、形状（shape）与/或步长（stride）的输入张量，我们提供输出张量的形状，
也就是逐点运算的任务空间（task-space），以及在不同维（dimension）上每个张量的步长信息。
形状与步长信息会被解析后以整数的形式传递给 Triton 内核。
由于 Triton 内核不支持元组（tuple）类型的参数，我们必须为形状与步长中的不同整数个数生成不同的内核。
尽管 Triton 从 3.3 版本开始支持元组类型的参数，它仍不支持对元组进行索引或者遍历这类操作。

<!--
In the Triton kernel, we map indices in task-space to the tensor multi-index based on the shape of the task space.
We then map them from tensor multi-index to memory offsets on each tensor according to its strides at each dimension.
For example, for a binary add operation of tensor of shape `(2, 3)` and `(2, 3)`, the task space is `(2, 3)`,
then task-id 4 is mapped to `(1, 1)` in the task space.
Say that the strides for the `lhs` are `(3, 1)`, the memory offset at the tensor is `4`,
and the strides for the rhs are `(1, 2)`, thus the memory offset for it is `3`.
-->
在 Triton 内核中，我们基于任务矿建的形状将任务空间中的索引映射到张量的多索引（multi-index）之上。
接下来，我们基于张量在不同维上的步长，将张量的多索引映射成每个张量上的内存偏移。
例如，对于一个二元加（add）操作，假定张量的形状分别为 `(2, 3)` 和 `(2, 3)`，
任务空间维 `(2, 3)`，则 task-id 为 4 的任务会映射到任务空间中的 `(1, 1)`。
如果 `lhs` 的步长为 `(3, 1)`，则张量上的内存偏移为 `4`；
如果 `rhs` 的步长为 `(1, 2)`，则张量上的内存偏移为 `3`。

<!--
For tensors with broadcastable but different shapes, we first broadcast those shapes
to get the shape of task space and view each tensors as the task shape,
which returns new tensors that share the same storage, but with new strides w.r.t the new shape.

In most cases, you can treat the decorated Triton JIT function as a scalar function that represents the operation.
But keep in mind that the generated kernels call the decorated function with `tl.tensor`s as inputs.
So avoid using `tl.tensor`s as conditions in control flow (`if` or `while`),
since Triton does not support non-scalar tensors as condition.
-->
对于可广播但形状不同的多个张量，我们会首先广播形状信息以获得任务空间的形状，
将每个张量视为任务的形状，进而得到共享相同存储的新张量，但就新的形状而言，步长也会不同。

在大多数情况下，你可以将修饰过的 Triton JIT 函数视为一个代表该操作的标量函数。
不过，需要注意的是，生成的内核会使用 `tl.tensor` 作为输入来调用修饰后的函数。
因此，需要避免在函数的控制流语句（如 `if` 或 `while`）中使用 `tl.tensor` 进行条件判断，
因为 Triton 不支持在条件中使用非标量的张量数据。

<!--
In the description above, we map task indices (integer) to memory offsets of each tensors,
since we view tasks in pointwise operation as a 1d-tensor and partitions it for each CTA.
We also have other task-space and partitioning schema, but for briefness, it is omitted here.

In addition to kernels, we also generate wrappers for the corresponding kernel.
The wrapper expect the outputs has the right shape, stride, dtype and device metadata,
and is ready for the computation.
-->
在上面的描述中，我们将任务索引（整数）映射为每个张量的的内存偏移，
因为我们将逐点操作中的任务视为一维张量，并在各个 CTA 之间对其进行切分。
我们也提供一些其他任务空间视图和划分模式，不过出于文字简洁考虑，在此不一一赘述。

除内核本身之外，我们还为其生成封装层。
封装层会期望算子输出具有正确的形状、步长、数据类型（dtype）和设备元数据，
这样的输出可以继续参与计算。

<!--
## Metadata Computation

Since pointwise operators shares similar logic at metadata computation,
which has been implemented as a common function used by all `PointwiseDynamicFunction`s.
It involes:
-->
## 元数据计算  {#metadata-computation}

由于逐点算子在元数据计算方面执行类似的逻辑，因此我们将它实现为一个可被所有
`PointwiseDynamicFunction` 调用的公共函数。这一函数的主要任务是：

<!--
- *shape inference*: infer the output shape by broadcasting input tensor shape;

- *ouput layout inference*: infer an appropriate layout (stride order) for output tensors if necessary;

- *type promotion*: infer output dtypes according to prescribed rules;

- *device inference*: infer the output device and the device to launch the kernel.

- *output allocation*.

- *infer the rank of the task-space*.
  This is a factor related to the code generation which depends on the arguments.
  It also involes trying to reduce the dimension of task-space to `1`
  when all pre-allocated tensors are dense and non-overlapping and have the same size
  and stride for each dimension.
-->
- **形状推测**：通过广播输入张量的形状推测输出的形状；
- **输出布局推理**：在必要时为输出张量推算一种合适的布局（步长顺序）；
- **类型提升**：根据预先设定的规则推测输出的数据类型（dtype）；
- **设备推测**：推测输出设备和要启动该内核的设备；
- **输出分配**：为输出分配内存；
- **推到任务空间的秩**：这是影响代码生成的一个方面，要依据输入参数来确定。
  在所有预先分配的张量都是稠密张量、彼此不重叠，并且在各维上尺寸与步长均相同时，
  此过程还会尝试将任务空间的维度缩减为 `1`。

<!--
Pre-allocated output tensors can also be passed into `PointwiseDynamicFunctions`.
In the cases where there are pre-allocated tensors in output tensors, the shape, layout,
dtype and device of theses pre-allocated tensors are respected and checked.

The metadata computation can also be skipped, but when doing so you should ensure that
the outputs have correct metadata and are pre-allocated, and you have to provide the rank of the task-space.
-->
`PointwiseDynamicFunction` 也可以接受预先分配的输出张量作为参数。
如果输出张量中包含预分配张量，系统会注意到其形状、布局、数据类型和设备信息，
并执行相应的检查。

元数据计算这一步骤可被略过，不过如果忽略元数据计算，你需要确保输出参数包含正确的元数据，
并且已经预分配，并且你必须提供任务空间的秩。

<!--
## Caching and dispatching

The decorator `@pointwise_dynamic` returns a `PointwiseDynamicFunction` object,
which servers as the proxy to all the decorated function.
It caches all the generated python modules and dispatches to them.

The dispatch result depends only on the rank of the task-space, rather than the shape of the task-space.
-->
## 缓存和派发 {#caching-and-dispatching}

修饰符 `@pointwise_dynamic` 会返回一个 `PointwiseDynamicFunction` 对象，
这一对象会为所有被修饰的函数扮演代理中介的角色。
该对象会缓存所生成的 Python 模块，并完成对这些模块的派发。

派发的结果仅仅取决于任务空间的秩，而不是任务空间的形状。

<!--
## Use the `pointwise_dynamic` decorator

### Basic
-->
## `pointwise_dynamic` 修饰符的使用

### 基础用法

<!--
Decorating the pointwise operator function with `pointwise_dynamic` can save the manual handling of
tensor addressing, tensor read/write, parallel tiling, tensor broadcasting, dynamic dimensions,
non-contiguous storage, type promotion, etc.
-->
使用 `@pointwise_dynamic` 来修饰逐点算子函数可以避免手动执行张量寻址、张量读写、
并行平铺、算子广播、动态维度、非连续存储、类型提升等动作。

<!--
For example, in the following code, you only need to provide a Triton JIT function
describing the computational logic (the Payload), the decorated function can then
take torch tesors as inputs and outputs, and support broadcasting, type-promotion, etc.
-->
例如，在下面的代码中，你只需要提供描述计算逻辑（负载）的 Triton JIT 函数，
被修饰的函数会将 Torch 张量作为输入和输出，并且满足广播、类型提升需求。

```python
@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)
```

<!--
Since the decorated function does not provide enough information for the code generation,
we supply other necessary information by passing arguemnts to `pointwise_dynamic`.
-->
由于被修饰的函数无法为代码生成提供足够的信息，我们会通过为 `pointwise_dynamic`
传递参数来提供必要的信息。

<!--
### Tensor/Non-Tensor

By default, `@pointwise_dynamic` treats each argument as a tensor, and generates code to load/store them.
But it can be configured by passing a list of boolean values to the parameter `is_tensor`
to indicate whether the corresponding argument is tensor or nor.
-->
### 张量与非张量

默认情况下，`@pointwise_dynamic` 会将每个参数都视为一个张量，并生成读写操作的代码。
不过你也可以通过为参数 `is_tensor` 传递一个布尔值列表来进行配置，
标示是否对应的参数是一个张量。

<!--
For non-tensor arguments, its type can be specfied by passing `dtypes` to the decorator, although not required.
For tensor arguments, the corresponding value in `dtypes` is ignored, since its dtype is dynamic
and Triton can dispatch according to it.
-->
对于非张量参数，其类型可以通过向修饰符传递 `dtypes` 来指定，尽管这一动作不是必需的。
对于张量参数，`dtypes` 参数中对应的值会被忽略，因为其具体类型是动态确定的，
Triton 会根据具体类型来完成派发。

<!--
For example, in the following code, the `alpha` parameter is defined as a non-tensor floating point number,
while the `x` and `y` parameters are defined as tensors.
-->
例如，在下面的代码中，参数 `alpha` 被指定为一个非张量的浮点数，而参数 `x` 和 `y`
被指定为张量参数。

```python
@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0,"DEFAULT")]
)
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha

a = torch.randn(128, 256, device="cuda")
b = torch.randn(256, device="cuda")
add_func(a, b, 0.2)
```

<!--
### Output dtypes

For pointwise operators to allocate outputs with correct dtype, `promotion_methods` is required.
Since the output dtype may be depedent on the input dtypes with some rules,
specifying the rule is more expressive than providing output dtypes directly.
-->
### 输出数据类型  {#output-dtypes}

为了让逐点算子能够正确地根据数据类型来为输出参数分配空间，需要指定 `promotion_methods` 参数。
由于算子可能会依据某种规则来基于输入数据类型来决定输出的数据类型，
与直接指定输出的数据类型相比，指定判定规则的表达能力会更强一些。

<!--
`promotion_methods` is a list of tuples (one per output), each of which consists of
several argument indices and a promotion method.
An argument index (an integer) is used to indicate the position of the argument,
which is dependent by the promotion method.
-->
`promotion_methods` 是一个列表，其中每个元素是一个元组，对应一个输出。
元组中进一步包含若干参数索引和一个类型提升方法枚举值。
参数索引是一个整数，用来给出参数的位置，其含义取决于类型提升方法。

<!--
The promotion method (an enum or string) denotes the method of type promotion.

- `DEFAULT` is the default rule for type promotion, which is suitable for most numeric operations;
- `NO_OPMATH` means copy data type as-is, which is suitable for non-numeric operation, like data-copy.
- `INT_TO_FLOAT` promotes integer to float.
- `ALWAYS_BOOL` always promote to boolean value.
- `COMPLEX_TO_FLOAT` promotes a complex number to float.
- `BOOL_TO_LONG` promotes a boolean value to long integer.
-->
类型提升方法（字符串或枚举值）用来标记类型提升方法。

- `DEFAULT` 是类型提升的默认规则，对于大多数数值操作而言都是适用的；
- `NO_OPMATH` 意味着直接复制数据类型，适用于非数值操作（如数据拷贝）；
- `INT_TO_FLOAT` 显式要求将整数提升为浮点数；
- `ALWAYS_BOOL` 显式要求将类型提升为布尔值；
- `COMPLEX_TO_FLOAT` 显式要求将复数值转换为浮点值；
- `BOOL_TO_LONG` 显式要求将布尔值提升为长整数值。

```python
class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    NO_OPMATH = (1,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)
    BOOL_TO_LONG = (5,)
```

类型提升示例

| 提升方法           | 算子示例                      |
| ------------------ | ----------------------------- |
| `DEFAULT`          | `add`                         |
| `NO_OPMATH`        | ` where`、`nextafter`、`cat`  |
| `INT_TO_FLOAT`     | `sin`                         |
| `ALWAYS_BOOL`      | `eq`                          |
| `COMPLEX_TO_FLOAT` | `abs`                         |
| `BOOL_TO_LONG`     | `pow`                         |

<!--
### Number of outputs

For pointwise operations with multiple output tensors, we need to inform `pointwise_dynamic`
about the number of outputs so it could generate code to store the output tensors.
For number of inputs, it can be inferred from the length of `is_tensor` of `dtypes`.
-->
### 输出参数个数  {#number-of-outputs}

对于需要输出多个张量的逐点运算而言，我们需要通知 `pointwise_dynamic` 输出参数个数，
这样修饰符逻辑才能生成用来保存输出张量的代码。
注意，对于输入参数的个数，修饰符逻辑能够根据 `dtypes` 的 `is_tensor` 数组长度进行推算。

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

<!--
## Use PointwiseDynamicFunction

### Basic
-->
## 使用 `PointwiseDynamicFunction`

### 基本用法

<!--
`PointwiseDynamicFunction` can be called with the same function signature as the decorated function,
as shown in previous examples.
-->
用户可以使用与被修饰函数相同的函数签名格式来调用 `PointwiseDynamicFunction`，
正如前面的示例所展示的那样。

<!--
### In-place Operation & Output arguments

Since `@pointwise_dynamic` generates wrappers that take outputs as arguments,
we can use it to implement inplace-operations.
For all `PointwiseDynamicFunction`s, you can pass output parameters to it using keyword arguments.
To discriminate between input arguments and output arguments, we follow a simple rule that
all input arguments must be passed by position and all output arguments must be passed
using keyword arguments.
-->
### 原地操作与输出参数

由于 `@pointwise_dynamic` 修饰符会生成封装逻辑，将算子的输出作为参数，
我们可以用它来实现原地（in-place）计算操作。
对于所有的 `PointwiseDynamicFunction` 对象，你都可以使用关键字参数（keyword aruments）
将输出参数传递给它。为了区分输入参数和输出参数，我们遵循一个基本的原则：
所有输入参数都要使用位置参数（positional arguments）来传递，
而所有输出参数都要使用关键字参数来传递。

<!--
The output parameters are named as `out{output_index}`.
Since the decorated function does not have name for return values,
we simply use the naming rule by suffixing `out` with output index.

We can implement inplace operations with it. For example:
-->
输出参数的命名约定为 `out{output_index}`。
由于被修饰的函数没有为返回值（输出参数）命名，这里的规则只是用 `out`
加上输出参数的索引。

我们可以使用这一机制来实现原地操作。例如：

```python
@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


def add_(A, B, *, alpha=1):
    return add_func(A, B, alpha, out0=A)
```

<!--
We can also pass pre-allocated outputs tensor, which is not in the input tensors.
For example:
-->
我们也可以为算子传递预分配的输出张量，这些张量不在输入张量之内。
例如：

```python
@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


def add_(A, B, *, alpha=1, out=None):
    return add_func(A, B, alpha, out=out)
```

<!--
Note that in these cases, you have to ensure that the output has the right metadata.
-->
注意，在这里，你必须确保输出参数具有正确的元数据。

<!--
### Manual Instantiation

For some operations you may want to skip the metadata computation, especially the process
to reduce the rank of task space, and prepare all inputs and outputs manually.
You can call the `instantiate()` method of `PointwiseDynamicFunction` with a specific task rank
to get a specific cached function and call it directly.
-->
### 手动实例化 {#manual-instantiation}

对于某些操作，你可能想要跳过元数据计算这一步，尤其是处理过程会缩减任务空间的秩时。
这时你希望手动准备所有输入和输出。
你可以使用特定的任务秩来调用 `PointwiseDynamicFunction` 的 `instantiate()` 方法，
得到一个特定的、被缓存的函数，并直接调用这一函数。

<!--
For example, the `flip` operator is not a pointwise operator in the sense that
each element in the output only depends on the element in the inputs at the corresponding position.
But if we can create a view of the input tensor with negative strides and shifted data pointer,
it can be framed as a pointwise copy. That is how we implement it with `pointwise_dynamic`.
-->
例如，算子 `flip` 不能算是一个逐点算子，原因是其输出中的每个元素仅仅依赖于输入中对应位置的元素。
不过，如果我们可以使用负的步长加上移位后的数据指针为输入张量创建一个视图，
这一操作也可以利用逐点复制逻辑来实现。
这就是我们使用 `pointwise_dynamic` 来实现它的方式。

```python
@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x

def flip(A: torch.Tensor, dims) -> torch.Tensor:
    strides = list(A.stride())
    flip_dims_b = [False for _ in A.stride()]
    for dim in dims:
        assert (
            dim >= -A.dim() and dim < A.dim()
        ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
            -A.dim(), A.dim() - 1, dim
        )
        assert not flip_dims_b[
            dim
        ], "dim {} appears multiple times in the list of dims".format(dim)
        flip_dims_b[dim] = True
    n = 0
    offset = 0
    for i in range(len(flip_dims_b)):
        if flip_dims_b[i] and A.size(i) > 1 and A.stride(i) != 0:
            offset += strides[i] * (A.shape[i] - 1)
            strides[i] = -strides[i]
            n += 1
    if n == 0 or A.numel() <= 1:
        return A.clone()
    out = torch.empty_like(A)
    # a flipped view of A
    flipped_A = StridedBuffer(A, strides=strides, offset=offset)

    overload = copy_func.instantiate(A.ndim)
    overload(flipped_A, out0=out)
    return out
```
