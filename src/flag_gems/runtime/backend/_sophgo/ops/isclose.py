import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.all import all
from flag_gems.utils import pointwise_dynamic, tl_extra_shim

try:
    _isfinited = tl_extra_shim.isfinited
    _finitef = tl_extra_shim.finitef
except Exception:
    pass


@pointwise_dynamic(
    is_tensor=[True, True, False, False, False, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
)
@triton.jit
def isclose_func(
    x,
    y,
    rtol,
    atol,
    equal_nan,
    zero_tol,
):
    cast_x = x if x.dtype.is_fp64() else x.to(tl.float32)
    cast_y = y if x.dtype.is_fp64() else y.to(tl.float32)
    if x.dtype.is_bf16():
        close = cast_x == cast_y
    else:
        close = x == y
    # equal_nan is passed as int (0 or 1), compare with 1
    if equal_nan == 1:
        close |= (cast_x != cast_x) & (cast_y != cast_y)
    # zero_tol is passed as int (0 or 1), compare with 0
    if zero_tol == 0:
        allowed = atol + tl.abs(rtol * cast_y)
        actual = tl.abs(cast_x - cast_y)
        finite_actual = (actual - actual) == 0
        close |= finite_actual & (actual <= allowed)
    return close


def isclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> torch.Tensor:
    logging.debug("GEMS ISCLOSE")
    # note: Int8 is not supported in isclose_func, because the result of int8 == int8 is wrong
    # in triton jit function, and needs to be fixed in triton. The same is true for bool.
    if A.dtype == torch.bool:
        return A == B
    if A.dtype != B.dtype:
        raise RuntimeError("{} did not match {}".format(A.dtype, B.dtype))
    if A.is_quantized or B.is_quantized:
        raise RuntimeError("isclose is not supported for quantized inputs.")
    if rtol < 0:
        raise RuntimeError(
            "rtol must be greater than or equal to zero, but got {}".format(rtol)
        )
    if atol < 0:
        raise RuntimeError(
            "atol must be greater than or equal to zero, but got {}".format(atol)
        )
    zero_tol = (rtol == 0) and (atol == 0)
    # Convert bool to int for Sophgo TPU backend compatibility
    equal_nan_int = 1 if equal_nan else 0
    zero_tol_int = 1 if zero_tol else 0
    return isclose_func(A, B, rtol, atol, equal_nan_int, zero_tol_int)


def allclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> bool:
    logging.debug("GEMS ALLCLOSE")
    return all(isclose(A, B, rtol, atol, equal_nan)).item()
