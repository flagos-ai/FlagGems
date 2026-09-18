import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

MIN_BLOCK = 2048
MAX_BLOCK = 131072
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


def _pick_block(n_elements):
    if n_elements >= 1_048_576 and n_elements % MAX_BLOCK == 0:
        return MAX_BLOCK, 32, False
    if n_elements >= 262_144 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 16384 and n_elements % 16384 == 0:
        return 16384, 8, False
    if n_elements <= 65536:
        return 2048, 4, True
    return 16384, 8, True


@triton.jit
def _atan2_poly(yc, xc, LOW_DEG: tl.constexpr):
    ay = tl.abs(yc)
    ax = tl.abs(xc)
    m = tl.maximum(ay, ax)
    mn = tl.minimum(ay, ax)
    u = mn / m
    u = tl.where(m > 0.0, u, 0.0)
    if LOW_DEG:
        p = 1.4017184409e-01
        p = p * u + -3.4245381452e-01
        p = p * u + -1.5262712340e-02
        p = p * u + 1.0031357076e00
        p = p * u + -7.7171867993e-05
    else:
        p = 5.21594798e-02
        p = p * u + -2.22082111e-01
        p = p * u + 3.16956596e-01
        p = p * u + -3.27826582e-02
        p = p * u + -3.28529690e-01
        p = p * u + -3.31425699e-04
        p = p * u + 1.00000797e00
        p = p * u + 4.05427219e-17
    t = tl.where(ay > ax, 1.5707963267948966 - p, p)
    t = tl.where(xc < 0.0, 3.141592653589793 - t, t)
    return tl.where(yc < 0.0, -t, t)


@triton.jit
def atan2_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LOW_DEG: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    yc = tl.load(x_ptr + offset, mask=mask, other=0).to(tl.float32)
    xc = tl.load(y_ptr + offset, mask=mask, other=0).to(tl.float32)
    res = _atan2_poly(yc, xc, LOW_DEG)
    tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def atan2_kernel_unmasked(
    x_ptr,
    y_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    LOW_DEG: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    yc = tl.load(x_ptr + offset).to(tl.float32)
    xc = tl.load(y_ptr + offset).to(tl.float32)
    res = _atan2_poly(yc, xc, LOW_DEG)
    tl.store(out_ptr + offset, res.to(out_ptr.dtype.element_ty))


def _launch(x, y, out):
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_block(n_elements)
    low_deg = out.dtype != torch.float32
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        atan2_kernel[grid](
            x,
            y,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            LOW_DEG=low_deg,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        atan2_kernel_unmasked[grid](
            x,
            y,
            out,
            BLOCK_SIZE=block_size,
            LOW_DEG=low_deg,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def _atan2_kernel_generic(input, other):
    input_f32 = input.to(tl.float32)
    other_f32 = other.to(tl.float32)
    result = tl_extra_shim.atan2(input_f32, other_f32)

    input_bits = input_f32.to(tl.int32, bitcast=True)
    other_bits = other_f32.to(tl.int32, bitcast=True)
    signed_pi = tl.where(input_bits < 0, -3.141592653589793, 3.141592653589793)
    negative_other = (other_f32 < 0.0) | ((other_f32 == 0.0) & (other_bits < 0))
    result = tl.where((input_f32 == 0.0) & negative_other, signed_pi, result)
    is_nan = (input_f32 != input_f32) | (other_f32 != other_f32)
    return tl.where(is_nan, float("nan"), result)


def _use_fast_path(input, other):
    return input.shape == other.shape and input.dtype == other.dtype


def atan2(input, other):
    logger.debug("GEMS_KUNLUNXIN ATAN2")
    if _use_fast_path(input, other):
        input = input.contiguous()
        other = other.contiguous()
        out = torch.empty_like(input)
        _launch(input, other, out)
        return out
    return _atan2_kernel_generic(input, other)


def atan2_(input, other):
    logger.debug("GEMS_KUNLUNXIN ATAN2_")
    if _use_fast_path(input, other):
        xc = input.contiguous()
        yc = other.contiguous()
        _launch(xc, yc, xc)
        if xc.data_ptr() != input.data_ptr():
            input.copy_(xc.view(input.shape))
        return input
    out = _atan2_kernel_generic(input, other)
    if out.shape != input.shape:
        raise RuntimeError(
            "output with shape "
            + str(tuple(out.shape))
            + " doesn't match the broadcast shape "
            + str(tuple(input.shape))
        )
    input.copy_(out)
    return input


def atan2_out(input, other, out):
    logger.debug("GEMS_KUNLUNXIN ATAN2_OUT")
    input = input.contiguous()
    other = other.contiguous()
    if out.is_contiguous() and out.dtype == input.dtype and out.shape == input.shape:
        _launch(input, other, out)
        return out
    tmp = torch.empty_like(input)
    _launch(input, other, tmp)
    out.copy_(tmp)
    return out
