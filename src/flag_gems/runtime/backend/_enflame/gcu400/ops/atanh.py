# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""atanh / atanh_ 的 GCU400 专属实现：1D flat Triton kernel。

输入与输出可共用同一块显存（in-place），也可写入独立输出（out-of-place）；
按元素字节数选择大 shape tile，常驻 CTA 数以网格跨步循环复用。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# 常驻 CTA 数与 GCU400 SIP 拓扑对齐；大 shape tile 按元素字节数分档
NUM_CTAS = 48
BLOCK_LARGE_16BIT = 8192
BLOCK_LARGE_32BIT = 8192
NUM_WARPS = 4


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def atanh_kernel(
    x_ptr,
    out_ptr,
    num_elements,
    BLOCK: tl.constexpr,
    FORMULA_MODE: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    MULTIPLE_OF_HINT: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = (num_elements + BLOCK - 1) // BLOCK
    for tile_index in tl.range(
        program_id, num_tiles, num_programs, num_stages=LOOP_STAGES
    ):
        offsets = tile_index * BLOCK + tl.arange(0, BLOCK)
        if MULTIPLE_OF_HINT == 1:
            offsets = tl.multiple_of(offsets, BLOCK)
        x = tl.load(x_ptr + offsets, mask=offsets < num_elements, other=0.0)
        if COMPUTE_MODE == 0:
            value = x.to(tl.float32)
        else:
            value = x
        if FORMULA_MODE == 0:
            # 原始定义：atanh(x) = 0.5 * log((1+x)/(1-x))
            result = 0.5 * tl.log((1.0 + value) / (1.0 - value))
        elif FORMULA_MODE == 1:
            # 两 log 相减：近 ±1 处避免除法商舍入放大误差
            result = 0.5 * (tl.log(1.0 + value) - tl.log(1.0 - value))
        elif FORMULA_MODE == 2:
            # log2 等价改写，log(x) = log2(x) * ln(2)
            result = 0.34657359027997264 * tl.log2((1.0 + value) / (1.0 - value))
        else:
            # 单次除法：atanh(x) = 0.5 * log(1 + 2x/(1-x))
            result = 0.5 * tl.log(1.0 + 2.0 * value / (1.0 - value))
        tl.store(out_ptr + offsets, result, mask=offsets < num_elements)


def _choose_block(num_elements, is_fp32):
    """小 shape 用 1024，中 shape 取最接近的 2 次幂，大 shape 按 dtype 分档。"""
    if num_elements <= 1024:
        return 1024
    if num_elements <= 32768:
        return triton.next_power_of_2(num_elements)
    return BLOCK_LARGE_32BIT if is_fp32 else BLOCK_LARGE_16BIT


def _launch(inp, out):
    num_elements = inp.numel()
    if num_elements == 0:
        return out
    is_fp32 = inp.dtype == torch.float32
    block_size = _choose_block(num_elements, is_fp32)
    grid_size = min(triton.cdiv(num_elements, block_size), NUM_CTAS)
    with torch_device_fn.device(inp.device):
        atanh_kernel[(grid_size,)](
            inp,
            out,
            num_elements,
            BLOCK=block_size,
            FORMULA_MODE=0,
            COMPUTE_MODE=0,
            MULTIPLE_OF_HINT=0,
            LOOP_STAGES=1,
            num_warps=NUM_WARPS,
        )
    return out


def _atanh(A):
    inp = A if A.is_contiguous() else A.contiguous()
    return _launch(inp, torch.empty_like(inp))


def atanh(A):
    logger.debug("GEMS_ENFLAME ATANH")
    return _atanh(A)


def atanh_(A):
    logger.debug("GEMS_ENFLAME ATANH_")
    if not A.is_contiguous():
        A.copy_(_atanh(A))
        return A
    return _launch(A, A)
