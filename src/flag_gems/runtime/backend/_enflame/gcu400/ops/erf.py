"""erf / erf_ 的 GCU400 专属实现。

非原地 erf 保留后端 pointwise_dynamic codegen 语义；in-place erf_ 走固定规模
CTA + 网格跨步的 1D flat kernel，消除 codegen 的 Python 侧调度与索引分解开销。
输入输出共用同一块显存，每元素只读一次、只写一次。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# 与 GCU400 SIP 拓扑对齐的常驻 CTA 数，kernel 内以跨步循环复用
NUM_CTAS = 24
# 大 shape 下按元素字节数区分的 tile 元素数
BLOCK_LARGE_16BIT = 8192
BLOCK_LARGE_32BIT = 65536
NUM_WARPS = 4
MATH_MODE = 0

_extern_erf = tl_extra_shim.erf


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def erf_func(x):
    return tl.math.erf(x.to(tl.float32))


def erf(x):
    logger.debug("GEMS_ENFLAME ERF")
    return erf_func(x)


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def _erf_inplace_kernel(
    x_ptr, num_elements, BLOCK: tl.constexpr, MATH_MODE: tl.constexpr
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(num_elements, BLOCK)
    for tile_index in tl.range(program_id, num_tiles, num_programs):
        offsets = tile_index * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_elements
        value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if MATH_MODE == 0:
            # Triton 内置 erf，与公共 codegen 完全同语义
            result = tl.math.erf(value)
        elif MATH_MODE == 1:
            # 设备端 extern __ef_erff
            result = _extern_erf(value)
        elif MATH_MODE == 2:
            # Abramowitz-Stegun 7.1.26：单 exp + 5 次 Horner 乘加，最大误差约 1.5e-7
            sign = tl.where(value >= 0.0, 1.0, -1.0)
            abs_value = tl.abs(value)
            reciprocal = 1.0 / (1.0 + 0.3275911 * abs_value)
            poly = 1.061405429 * reciprocal - 1.453152027
            poly = poly * reciprocal + 1.421413741
            poly = poly * reciprocal - 0.284496736
            poly = poly * reciprocal + 0.254829592
            poly = poly * reciprocal
            result = sign * (1.0 - poly * tl.exp(-abs_value * abs_value))
        else:
            # 与 mode 2 同式，改用 exp2 减少超越函数开销
            sign = tl.where(value >= 0.0, 1.0, -1.0)
            abs_value = tl.abs(value)
            reciprocal = 1.0 / (1.0 + 0.3275911 * abs_value)
            poly = 1.061405429 * reciprocal - 1.453152027
            poly = poly * reciprocal + 1.421413741
            poly = poly * reciprocal - 0.284496736
            poly = poly * reciprocal + 0.254829592
            poly = poly * reciprocal
            exponent = -abs_value * abs_value * 1.4426950408889634
            result = sign * (1.0 - poly * tl.exp2(exponent))
        tl.store(x_ptr + offsets, result, mask=mask)


def _select_block_size(num_elements, is_fp32):
    """小 shape 回落到 1024；中 shape 取 2 次幂；大 shape 按 dtype 字节数分档。"""
    if num_elements <= 1024:
        return 1024
    if num_elements <= 32768:
        return triton.next_power_of_2(num_elements)
    return BLOCK_LARGE_32BIT if is_fp32 else BLOCK_LARGE_16BIT


def erf_(A):
    logger.debug("GEMS_ENFLAME ERF_")
    num_elements = A.numel()
    if num_elements == 0:
        return A
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    # 非浮点 / 非连续输入沿用 codegen 路径，保持 promotion 与 stride 语义
    if A.dtype not in supported_dtypes or not A.is_contiguous():
        return erf_func(A, out0=A)
    block_size = _select_block_size(num_elements, A.dtype == torch.float32)
    grid_size = min(triton.cdiv(num_elements, block_size), NUM_CTAS)
    with torch_device_fn.device(A.device):
        _erf_inplace_kernel[(grid_size,)](
            A,
            num_elements,
            BLOCK=block_size,
            MATH_MODE=MATH_MODE,
            num_warps=NUM_WARPS,
        )
    return A


__all__ = ["erf", "erf_"]
