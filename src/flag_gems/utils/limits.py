import triton
from triton import language as tl


@triton.jit
def get_dtype_max(dtype: tl.constexpr):
    """get a value which is greater that all other values of that dtype"""
    # extract the tl.dtype from tl.constexpr so as to use its methods
    dtype_ = dtype.value
    if dtype_.is_floating():
        value: tl.constexpr = float("inf")
        return value
    if dtype_.is_int_signed():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = 2 ** (width - 1) - 1
        return value
    if dtype_.is_int_unsigned():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = 2**width - 1
        return value


@triton.jit
def get_dtype_min(dtype):
    """get a value which is less that all other values of that dtype"""
    dtype_ = dtype.value  # tl.dtype
    if dtype_.is_floating():
        value: tl.constexpr = float("-inf")
        return value
    if dtype_.is_int_signed():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = -1 * 2 ** (width - 1)
        return value
    if dtype_.is_int_unsigned():
        value: tl.constexpr = 0
        return value
