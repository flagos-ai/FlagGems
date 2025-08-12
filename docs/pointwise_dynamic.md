# Pointwise Dynamic

## Pointwise operations

Pointwise operators are trivial to parallelize. Most parallel programming guides begin with pointwise addition between 2 contiguous vectors. For [vector_add in Triton](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py), it is simple to implement a task partitioning schema that each CTA reads a contiguous range from each input vector and writes to a contiguous range of the output vector.

However, actual use caes for pointwise operators may be more complicated.

- The input tensors may be contiguous: they may be contiguous in memory but not in a row-major order; or they may not be dense; or they may have internal overlapping;

- The input tensors may have arbitrary and/or different number of dimensions. And it is not always possible to view them as contiguous vectors of the same shape;

- The input tensors may have different but broadcastable shape;

- The inputs may mix tensor/non-tensor;

- Different pointwise operators share common logic to compute indices, while it is tedious to rewrite them over-and-over for each operator;

We propose a code generation based approach to solve these problems. The principles of our design are:

- Pointwise operations are generally memory-bound, so avoid copying tensors to make them contiguous vectors;

- Pointwise operators should support inputs of arbitrary and/or different ranks, sizes, strides, broadcasting inputs, mixing tensor and non-tensors;

- Different pointwise operators should share common internal facilities, either as a library, or a template-based code generation mechanism, to reduce boilerplate code.

- The common internal facilities should be configurable for adaption of different backends.

The result is a decorator `@pointwise_dynamic`. It provides a common wrapper for pointwise operator and mechanism to generate triton kernels and corresponding wrappers based on the operation and the input configureations.

## Code generation

The basic usage of pointwise_dynamic is to decorate a triton.jit function that has return value, which is used to map inputs to outputs. The jit function is similar to a function with `__device__` declaration specifier, a function that can be called from device. And we generate a triton jit function to call it, which acts like a cuda kernel(a function with `__global__` declaration specifier) that loads and stores data at global memory.

In order to support input tensors of different rank, shape, stride, we pass the shape of the output tensor (which is also the task-space for pointwise operation) , and strides of each tensor at every dimension. The shape and strides are unpacked and passed to kernel as integers. Due to the lack of support for tuple as arguments to Triton kernels, we have to generate different kernels for different number of integers in the shape and strides. Although Triton supports tuple as arguemnts since version 3.3, it does not support all operation on tuples(indexing, iteration, ...).

In the triton kernel, we map indices in task-space to the tensor multi-index according to the shape of the task space. Then we map them from tensor multi-index to memory offsets on each tensor according to its strides at each dimension. For example, for a binary add operation of tensor of shape `(2, 3)` and `(2, 3)`, the task space is `(2, 3)`, then task-id 4 is mapped to `(1, 1)` in task space. Say that the strides for the lhs are `(3, 1)`, the memory offset at the tensor is 4, and the strides for the rhs are `(1, 2)`, thus the memory offset for it is 3.

For tensors with broadcastable but different shapes, we first broadcast those shapes to get the shape of task space and view each tensors as the task shape, which returns new tensors that share the same storage, but with new strides w.r.t the new shape.

In most of the cases, you can treat the decorated Triton jit function as a scalar function that represents the operation. But keep in mind that the generated kernels call the decorated function with `tl.tensor`s as inputs. So avoid using `tl.tensor`s as conditions in control flow (`if` or `while`), since Triton does not support non-scalar tensors as condition.

In the description above, we map task indices (integer) to memory offsets of each tensors, since we view tasks in pointwise operation as a 1d-tensor and partitions it for each CTA. We also have other task-space and partitioning schema, but for briefness, it is omitted here.

In addition to kernels, we also generate wrappers for the corresponding kernel. The wrapper expect the outputs has the right shape, stride, dtype and device meta data, and is ready for the computation.

## MetaData Computation

Since pointwise operators shares similar logic at meta data computation, which has been implemented as a common function used by all `PointwiseDynamicFunction`s. It involes:

- shape inference: infer the output shape by broadcasting input tensor shape;

- ouput layout inference: infer an appropriate layout (stride order) for output tensors if necessary;

- type promotion: infer output dtypes according to prescribed rules;

- device inference: infer the output device and the device to launch the kernel.

- output allocation.

- infer the rank of the task-space. This is a factor related to the code generation which depends on the arguments. It also involes trying to reduce the dimension of task-space to 1 when all pre-allocated tensors are dense and non-overlapping and have the same size and stride for each dimension.

Pre-allocated output tensors can also be passed into `PointwiseDynamicFunctions`. In the cases where there are pre-allocated tensors in output tensors, the shape, layout, dtype and device of theses pre-allocated tensors are respected and checked.

The meta data computation can also be skipped, but in this case, you should ensure that the outputs have correct meta data and are pre-allocated. And you should provide the rank of the task-space.

## Caching and dispatching

The decorator `pointwise_dynamic` returns a `PointwiseDynamicFunction` object, which servers as the proxy to all the decorated function. It caches call the generated python modules and dispatches to them.

The dispatch result depends only on the rank of the task-space, rather than the shape of the task-space.

## Use pointwise_dynamic decorator

### Basic

Decorating the pointwise operator function with `pointwise_dynamic` can save the manual handling of tensor addressing, tensor read/write, parallel tiling, tensor broadcasting, dynamic dimensions, non-contiguous storage, type promotion, etc.

For example, in the following code, you only need to provide a Triton jit function describing the computational logic (the Payload), the decorated function can then take torch tesors as inputs and outputs, and support broadcasting, type-promotion, etc.

```python
@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)
```

Since the decorated function does not provide enough information for the code generation, we supply other necessary information by passing arguemnts to `pointwise_dynamic`.

### Tensor/Non-Tensor

By default, `pointwise_dynamic` treats each arguemnt as tensor, and generates code to load/store them. But it can be configured by passing a list of boolean values to the parameter `is_tensor` to indicate whether the corresponding argument is tensor or nor.

For non-tensor arguments, its type can be specfied by passing `dtypes` to the decorator, although it is not required. For tensor arguments, the corresponding value in `dtypes` is ignored, since its dtype is dynamic and Triton can dispatch according to it.

For example, in the following code, the `alpha` parameter is defined as a non-tensor floating point number, while the `x` and `y` parameters are defined as tensors.

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

### Ouput dtypes

For pointwise operators to allocate outputs with correct dtype, `promotion_methods` is required. Since the output dtype may be depedent on the input dtypes with some rules, specifying the rule is more expressive than providing output dtypes directly.

`promotion_methods` is a list of tuples (one per output), each of which consists of several arg indices and a promotion method. An arg index (an integer) is used to indicate the position of the argument, which is dependent by the promotion method. The promotion method (an enum or string) denotes the method of type promotion.

- DEFAULT is the default rule for type promotion, which is suitable for most numeric operations;

- NO_OPMATH means copy data type as-is, which is suitable for non-numeric operation, like data-copy.

```python
class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    NO_OPMATH = (1,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)
    BOOL_TO_LONG = (5,)
```

Examples：

- `DEFAULT` ：add
- `NO_OPMATH` ： where, nextafter, cat
- `INT_TO_FLOAT` ：sin
- `ALWAYS_BOOL` ：eq
- `COMPLEX_TO_FLOAT` ：abs
- `BOOL_TO_LONG` ：pow

### Number of outputs

For pointwise operations with multiple output tensors, we need to inform `pointwise_dynamic` about the number of outputs so it could generate code to store the output tensors. For number of inputs, it can be inferred from the length of `is_tensor` of `dtypes`.

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

## Use PointwiseDynamicFunction

### Basic

PointwiseDynamicFunction can be called with the same function signature as the decorated function, as shown in previous examples.

### Inplace Operation & Output arguments

Since `pointwise_dynamic` generates wrappers that take outputs as arguments, we can use it to implement inplace-operations. For all `PointwiseDynamicFunction`s, you can pass output parameters to it by key-word. To discriminate between input arguments and output arguments, we now follow a simple rule that all input arguments must be passed by position and all output arguments must be passed by keyword.

The output parameters are named as `out{output_index}`. Since the decorated function does not have name for return values, we simply use the naming rule by suffixing `out` with output index.

We can implement inplace operations with it. Example

```python
@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


def add_(A, B, *, alpha=1):
    return add_func(A, B, alpha, out0=A)
```

We can also pass pre-allocated outputs tensor, which is not in input tensors. Example

```python
@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


def add_(A, B, *, alpha=1, out=None):
    return add_func(A, B, alpha, out=out)
```

Note that in these cases, you have to ensure that the output has the right meta data.

### Manual Instantiation

For some operations you may want to skip the meta data computation, especially the process to reduce the rank of task space, and prepare all inputs and outputs manually. Then you can call `instantiate` method of `PointwiseDynamicFunction` with a specific task rank to get a specific cached function and call it directly.

For example, `flip` operator is not a pointwise operator in the sense that each element in the output only depends on the element in the inputs at the corresponding position. But if we can create a view of the input tensor with negative strides and shifted data pointer, it can be framed as a pointwise copy. That is how we implement it with `pointwise_dynamic`.

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
