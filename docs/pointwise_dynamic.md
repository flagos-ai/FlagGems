### Automatic Codegen

In FlagGems, we provide automatic code generation that developers can use to conveniently generate pointwise single operators and pointwise fused operators. Automatic code generation can handle various needs such as normal pointwise computations, non-tensor arguments, and specifying output data types.

#### Normal Pointwise Operator

Decorating the pointwise operator function with `pointwise_dynamic` can save the manual handling of tensor addressing, tensor read/write, parallel tiling, tensor broadcasting, dynamic dimensions, non-contiguous storage, etc. For example, in the following code, developers only need to describe the computational logic to generate flexible and efficient Triton code.

```python
@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)
```

#### Non-Tensor Argument

By default, `pointwise_dynamic` treats all parameters as tensors, and by passing a list of boolean values to the parameter `is_tensor`, developers can specify which parameters are tensors and which are not. Additionally, developers can pass in `dtypes` to indicate the data types of non-tensor parameters, but this is not required. For example, in the following code, the `alpha` parameter is defined as a non-tensor floating point number, while the `x` and `y` parameters are defined as tensors.

```python
@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0,"DEFAULT")]
)
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha
```

#### Output Data Type

Furthermore, developers MUST provide promotion_methods to specify how type promotion should be handled for the operation to achieve the correct output type during computation.

```python
@pointwise_dynamic(output_dtypes=[torch.bool])
@triton.jit
def ge(x, y):
    return x > y
```

In `promotion_methods`, an `int` is used to indicate the position of the parameter requiring type promotion, while a `str` denotes the method of type promotion. The `str` corresponds to the following enumerated types:

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
