import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
pow = tl_extra_shim.pow
_tanh = tl_extra_shim.tanh


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_kernel(x):
    return _tanh(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward_kernel(y, dy):
    y = y.to(tl.float32)
    return dy.to(tl.float32) * (1.0 - y * y)


_LEGACY_TANH_BACKWARD = os.environ.get("GEMS_KUNLUNXIN_TANH_BACKWARD_LEGACY") == "1"

_FAST_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False

_BF16_FAST_MAX_NUMEL = 32 << 20


def _pick_block(n_elements):
    if n_elements <= 16384:
        return 2048, 4, n_elements % 2048 != 0
    if n_elements <= 1048576:
        return 16384, 8, n_elements % 16384 != 0
    return 65536, 8, n_elements % 65536 != 0


@triton.jit
def tanh_backward_flat_kernel(y_ptr, dy_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    y = tl.load(y_ptr + offs, mask=mask, other=0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0).to(tl.float32)
    res = dy * (1.0 - y * y)
    tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def tanh_backward_flat_kernel_unmasked(y_ptr, dy_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(y_ptr + offs).to(tl.float32)
    dy = tl.load(dy_ptr + offs).to(tl.float32)
    res = dy * (1.0 - y * y)
    tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty))


def _tanh_backward_fast_eligible(grad_output, output):
    if _LEGACY_TANH_BACKWARD:
        return False
    if not (torch.is_tensor(grad_output) and torch.is_tensor(output)):
        return False
    if output.dtype is not grad_output.dtype or output.dtype not in _FAST_DTYPES:
        return False
    if output.shape != grad_output.shape:
        return False
    if output.dim() == 0 or output.numel() == 0:
        return False
    if output.dtype is torch.bfloat16 and output.numel() > _BF16_FAST_MAX_NUMEL:
        return False
    return output.is_contiguous() and grad_output.is_contiguous()


def _tanh_backward_fast(grad_output, output):
    n_elements = output.numel()
    out = torch.empty_like(output)
    block, warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block),)
        tanh_backward_flat_kernel[grid](
            output,
            grad_output,
            out,
            n_elements,
            BLOCK=block,
            num_warps=warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block,)
        tanh_backward_flat_kernel_unmasked[grid](
            output,
            grad_output,
            out,
            BLOCK=block,
            num_warps=warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    return out


def tanh(self):
    logger.debug("GEMS_KUNLUNXIN TANH")
    out = tanh_kernel(self)
    return out


def tanh_backward(grad_output, output):
    logger.debug("GEMS_KUNLUNXIN TANH_BACKWARD")
    if _tanh_backward_fast_eligible(grad_output, output):
        return _tanh_backward_fast(grad_output, output)
    in_grad = tanh_backward_kernel(output, grad_output)
    return in_grad


def tanh_(A):
    logger.debug("GEMS_KUNLUNXIN TANH_")
    out = tanh_kernel(A, out0=A)
    return out
