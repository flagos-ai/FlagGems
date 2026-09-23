import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)

_FAST_BLOCK = 16384
_FAST_WARPS = 32


@triton.jit
def trunc_fast_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)  # numel % BLOCK == 0 guaranteed by caller
    xf = x.to(tl.float32)
    a = tl.abs(xf)
    r = (a + 12582912.0) - 12582912.0
    d = tl.minimum(tl.maximum((r - a) * 1e38, 0.0), 1.0)
    t = (r - d) * tl.where(xf >= 0.0, 1.0, -1.0)
    tl.store(y_ptr + offs, t.to(y_ptr.dtype.element_ty))


@triton.jit
def trunc_masked_kernel(x_ptr, y_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    a = tl.abs(xf)
    r = (a + 12582912.0) - 12582912.0
    d = tl.minimum(tl.maximum((r - a) * 1e38, 0.0), 1.0)
    t = (r - d) * tl.where(xf >= 0.0, 1.0, -1.0)
    tl.store(y_ptr + offs, t.to(y_ptr.dtype.element_ty), mask=mask)


# Generic fallback: any dtype/layout/shape (bf16 keeps the extern truncf
# which is not collapsed here); exact via libdevice.
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def trunc_func(x):
    x_fp32 = x.to(tl.float32)
    return tl_extra_shim.trunc(x_fp32).to(x.dtype)


def _trunc_impl(A, out=None):
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32)
        and A.is_contiguous()
        and A.dim() > 0
        and numel
        > 1024  # tiny tensors: generic pointwise path is at/below the launch floor
    ):
        block = min(_FAST_BLOCK, triton.next_power_of_2(numel))
        if out is None:
            out = torch.empty_like(A)
        if numel % block == 0:
            trunc_fast_kernel[(numel // block,)](
                A, out, BLOCK=block, num_warps=_FAST_WARPS
            )
        else:
            trunc_masked_kernel[(triton.cdiv(numel, block),)](
                A, out, numel, BLOCK=block, num_warps=_FAST_WARPS
            )
        return out
    if out is None:
        return trunc_func(A)
    trunc_func(A, out0=out)
    return out


def trunc(A):
    logger.debug("GEMS_KUNLUNXIN TRUNC")
    return _trunc_impl(A)


def trunc_(A):
    logger.debug("GEMS_KUNLUNXIN TRUNC_")
    trunc_func(A, out0=A)
    return A
