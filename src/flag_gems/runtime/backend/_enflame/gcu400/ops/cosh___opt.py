"""cosh_ 的 GCU400 专属 Triton 实现：in-place 逐元素双曲余弦。

用固定规模 CTA + 网格跨步循环的 1D flat kernel 替代公共 pointwise_dynamic
codegen 的 Python 侧调度；输入与输出共用同一块显存，读写各一次。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.cosh import cosh_func as _common_cosh_func
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# 与 GCU400 SIP 拓扑对齐的常驻 CTA 数，kernel 内部以跨步循环复用
NUM_CTAS = 48
# 大 shape 下按元素字节数区分的 tile 元素数
BLOCK_LARGE_16BIT = 8192
BLOCK_LARGE_32BIT = 32768
NUM_WARPS = 4


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def _cosh_inplace_kernel(x_ptr, num_elements, BLOCK: tl.constexpr):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(num_elements, BLOCK)
    for tile_index in tl.range(program_id, num_tiles, num_programs):
        offsets = tile_index * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_elements
        value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # cosh(x) = (exp(x) + exp(-x)) / 2；fp32 中间计算保证低精度输入的精度
        result = 0.5 * (tl.exp(value) + tl.exp(-value))
        tl.store(x_ptr + offsets, result, mask=mask)


def _select_block_size(num_elements, is_fp32):
    """小 shape 用 1024，中 shape 取最接近的 2 次幂，大 shape 按 dtype 分档。"""
    if num_elements <= 1024:
        return 1024
    if num_elements <= 32768:
        return triton.next_power_of_2(num_elements)
    return BLOCK_LARGE_32BIT if is_fp32 else BLOCK_LARGE_16BIT


def cosh_(A):
    logger.debug("GEMS_ENFLAME COSH_")
    num_elements = A.numel()
    if num_elements == 0:
        return A
    # 非浮点 / 非连续输入沿用公共 codegen 路径，保持 promotion 与 stride 语义
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if A.dtype not in supported_dtypes or not A.is_contiguous():
        return _common_cosh_func(A, out0=A)
    is_fp32 = A.dtype == torch.float32
    block_size = _select_block_size(num_elements, is_fp32)
    grid_size = min(triton.cdiv(num_elements, block_size), NUM_CTAS)
    with torch_device_fn.device(A.device):
        _cosh_inplace_kernel[(grid_size,)](
            A,
            num_elements,
            BLOCK=block_size,
            num_warps=NUM_WARPS,
        )
    return A


__all__ = ["cosh_"]
