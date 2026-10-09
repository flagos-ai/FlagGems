import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.xpu.libdevice as xpu

from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

MIN_BLOCK = 2048
UNROLL_NUM = 8
INPLACE_UNROLL_NUM = 2
INPLACE_UNROLL_NUM_BF16 = 4
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    if n_elements <= 16384:
        return 2048, 4, True
    if n_elements <= 262144 and n_elements % 8192 == 0:
        return 8192, 4, False
    if n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements <= 65536:
        return 2048, 4, True
    return 16384, 8, True


@triton.jit
def _acos_body(x):
    t = 0.5 - 0.5 * tl.abs(x)
    s = t * xpu.rsqrt(t + 1e-30)
    p = -493.19885254
    p = p * t + 1060.03149414
    p = p * t + -941.14831543
    p = p * t + 445.70321655
    p = p * t + -121.05153656
    p = p * t + 18.99153519
    p = p * t + -1.44778073
    p = p * t + 0.39646727
    p = p * t + 1.99919987
    y = s * p
    m = tl.minimum(1.0, tl.maximum(0.0, -x * 8.50705917e37))
    return m * 3.1415927 + (1.0 - 2.0 * m) * y


@triton.jit
def acos_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    t = 0.5 - 0.5 * tl.abs(x)
    s = tl.sqrt(t)
    p = -246.59942627
    p = p * t + 530.01574707
    p = p * t + -470.57415771
    p = p * t + 222.85160828
    p = p * t + -60.52576828
    p = p * t + 9.49576759
    p = p * t + -0.72389036
    p = p * t + 0.19823363
    p = p * t + 0.99959993
    y = (s * p) * 2.0
    r = tl.where(x < 0.0, 3.1415927 - y, y)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def acos_kernel_unmasked(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    t = 0.5 - 0.5 * tl.abs(x)
    s = tl.sqrt(t)
    p = -246.59942627
    p = p * t + 530.01574707
    p = p * t + -470.57415771
    p = p * t + 222.85160828
    p = p * t + -60.52576828
    p = p * t + 9.49576759
    p = p * t + -0.72389036
    p = p * t + 0.19823363
    p = p * t + 0.99959993
    y = (s * p) * 2.0
    r = tl.where(x < 0.0, 3.1415927 - y, y)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty))


def _launch(x, out, unroll_num=UNROLL_NUM):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        acos_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=unroll_num,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        acos_kernel_unmasked[grid](
            x,
            out,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=unroll_num,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def _inplace_unroll(dtype):
    if dtype == torch.bfloat16:
        return INPLACE_UNROLL_NUM_BF16
    return INPLACE_UNROLL_NUM


def acos(x):
    logger.debug("GEMS_KUNLUNXIN ACOS")
    x = x.contiguous()
    out = torch.empty_like(x)
    _launch(x, out)
    return out


def acos_(A):
    logger.debug("GEMS_KUNLUNXIN ACOS_")
    x = A.contiguous()
    _launch(x, x, unroll_num=_inplace_unroll(x.dtype))
    if x.data_ptr() != A.data_ptr():
        A.copy_(x.view(A.shape))
    return A
