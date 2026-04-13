import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

from ..utils.pointwise_dynamic import pointwise_dynamic
from .all import all

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
_isfinited = tl_extra_shim.isfinited
_finitef = tl_extra_shim.finitef

cluster_num = 12
core_num = 64
thread_num = core_num * cluster_num
buf_len_per_core = 2048
vector_size = 16


def get_block(n: int) -> int:
    if n < cluster_num:
        res = cluster_num
    else:
        res = cluster_num * triton.cdiv(n, cluster_num)
    return res


@triton.jit
def reduce_and(a, b):
    return a and b


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
    equal_nan: tl.constexpr,
    zero_tol: tl.constexpr,
):
    cast_x = x if x.dtype.is_fp64() else x.to(tl.float32)
    cast_y = y if x.dtype.is_fp64() else y.to(tl.float32)
    if x.dtype.is_bf16():
        close = cast_x == cast_y
    else:
        close = x == y
    if equal_nan:
        close |= (cast_x != cast_x) & (cast_y != cast_y)
    if not zero_tol:
        allowed = atol + tl.abs(rtol * cast_y)
        actual = tl.abs(cast_x - cast_y)
        actual_finite = _isfinited(actual) if x.dtype.is_fp64() else _finitef(actual)
        close |= actual_finite.to(tl.int1) & (actual <= allowed)
    return close


# Fused allclose kernel stage 1: isclose + partial reduction
@libentry()
@triton.jit
def allclose_kernel_1(
    inp_x,
    inp_y,
    mid,
    n_elements,
    rtol,
    atol,
    EQUAL_NAN: tl.constexpr,
    ZERO_TOL: tl.constexpr,
    IS_FP64: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(inp_x + offset, mask=mask, other=0.0)
    y = tl.load(inp_y + offset, mask=mask, other=0.0)

    cast_x = x if IS_FP64 else x.to(tl.float32)
    cast_y = y if IS_FP64 else y.to(tl.float32)

    if IS_BF16:
        close = cast_x == cast_y
    else:
        close = x == y

    if EQUAL_NAN:
        close |= (cast_x != cast_x) & (cast_y != cast_y)

    if not ZERO_TOL:
        allowed = atol + tl.abs(rtol * cast_y)
        actual = tl.abs(cast_x - cast_y)
        if IS_FP64:
            actual_finite = _isfinited(actual)
        else:
            actual_finite = _finitef(actual)
        close |= actual_finite.to(tl.int1) & (actual <= allowed)

    # For masked-out elements, treat as True (allclose identity)
    close = close | ~mask

    all_val = tl.reduce(close, axis=0, combine_fn=reduce_and)
    tl.store(mid + pid, all_val)


# Stage 2: reduce partial results
@libentry()
@triton.jit
def allclose_kernel_2(
    mid,
    out,
    MID_SIZE,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid_val = tl.load(mid + offset, mask=mask, other=1).to(tl.int1)
    all_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_and)
    tl.store(out, all_val)


def isclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> torch.Tensor:
    logger.debug("GEMS ISCLOSE")
    if not equal_nan:
        os.environ["XPU_cmp_nan"] = "1"
    else:
        if "XPU_cmp_nan" in os.environ:
            del os.environ["XPU_cmp_nan"]
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
    return isclose_func(A, B, rtol, atol, equal_nan, zero_tol)


def allclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> bool:
    logger.debug("GEMS ALLCLOSE")
    if A.dtype == torch.bool:
        return all(A == B).item()
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

    # Make inputs contiguous for the fused kernel
    A_contig = A.contiguous()
    B_contig = B.contiguous()

    n_elements = A_contig.numel()
    if n_elements == 0:
        return True

    is_fp64 = A_contig.dtype == torch.float64
    is_bf16 = A_contig.dtype == torch.bfloat16

    block_size = min(
        triton.cdiv(get_block(n_elements), cluster_num),
        triton.cdiv(buf_len_per_core * core_num, 4),
    )
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=A.device)
    out = torch.empty([], dtype=torch.bool, device=A.device)

    with torch_device_fn.device(A.device):
        allclose_kernel_1[(mid_size, 1)](
            A_contig,
            B_contig,
            mid,
            n_elements,
            rtol,
            atol,
            EQUAL_NAN=equal_nan,
            ZERO_TOL=zero_tol,
            IS_FP64=is_fp64,
            IS_BF16=is_bf16,
            BLOCK_SIZE=block_size,
            buffer_size_limit=2048,
        )
        if mid_size == 1:
            return mid.reshape([]).item()
        allclose_kernel_2[(1, 1)](
            mid,
            out,
            mid_size,
            BLOCK_MID=block_mid,
            buffer_size_limit=2048,
        )

    return out.item()
