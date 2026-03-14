import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_INPLACE_ERROR_DTYPES = {
    torch.bool: "Bool",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
}


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def asinh_func(x):
    x = x.to(tl.float32)
    abs_x = tl.abs(x)
    inner = tl.log(abs_x + tl.sqrt(abs_x * abs_x + 1.0))
    return tl.where(x < 0, -inner, inner)


def _can_use_triton_asinh(x, out=None):
    if x.is_complex() or x.dtype == torch.float64:
        return False
    if out is not None and (out.is_complex() or out.dtype == torch.float64):
        return False
    return True


def _raise_asinh_inplace_error(tensor):
    dtype_name = _INPLACE_ERROR_DTYPES.get(tensor.dtype)
    if dtype_name is None:
        return
    raise RuntimeError(
        f"result type Float can't be cast to the desired output type {dtype_name}"
    )


def _asinh_cpu_fallback(x, out=None):
    x_cpu = x.cpu()
    if x_cpu.dtype == torch.complex32:
        x_cpu = x_cpu.to(torch.complex64)
    target_dtype = out.dtype if out is not None else x.dtype
    res = torch.asinh(x_cpu).to(target_dtype)
    if out is None:
        out = torch.empty_like(x, dtype=target_dtype, device=x.device)
        out.copy_(res)
        return out
    out.copy_(res)
    return out


def asinh(A):
    logger.debug("GEMS ASINH")
    if _can_use_triton_asinh(A):
        return asinh_func(A)
    return _asinh_cpu_fallback(A)


def asinh_(A):
    logger.debug("GEMS ASINH_")
    if _can_use_triton_asinh(A):
        if A.dtype.is_floating_point:
            return asinh_func(A, out0=A)
        _raise_asinh_inplace_error(A)
    return _asinh_cpu_fallback(A, out=A)


def asinh_out(A, *, out=None):
    logger.debug("GEMS ASINH_OUT")
    if out is None:
        return asinh(A)
    if not (out.dtype.is_floating_point or out.is_complex()):
        _raise_asinh_inplace_error(out)
    if _can_use_triton_asinh(A, out):
        return asinh_func(A, out0=out)
    return _asinh_cpu_fallback(A, out=out)
