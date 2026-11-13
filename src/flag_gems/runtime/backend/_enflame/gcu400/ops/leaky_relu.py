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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# fixed grid 匹配 GCU400 的 SIP 拓扑，核内跨步循环覆盖大 shape
NUM_SIPS = 24
GRID_MULT = 2
BLOCK_16 = 8192
BLOCK_32 = 8192
NUM_WARPS = 4


@libentry()
@triton.jit(do_not_specialize=["N_total", "negative_slope"])
def _leaky_relu_kernel(
    input_ptr,
    output_ptr,
    N_total,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
    KEEP_FP32: tl.constexpr,
    MUL_SIGN: tl.constexpr,
    EVEN_PATH: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        if EVEN_PATH:
            x = tl.load(input_ptr + off)
            if KEEP_FP32:
                x = x.to(tl.float32)
            if MUL_SIGN:
                # 把 where 的两支收敛成标量因子，只保留一次向量乘法
                factor = tl.where(x >= 0, 1.0, negative_slope)
                output = x * factor
            else:
                output = tl.where(x >= 0, x, x * negative_slope)
            tl.store(output_ptr + off, output)
        else:
            mask = off < N_total
            x = tl.load(input_ptr + off, mask=mask, other=0.0)
            if KEEP_FP32:
                x = x.to(tl.float32)
            if MUL_SIGN:
                factor = tl.where(x >= 0, 1.0, negative_slope)
                output = x * factor
            else:
                output = tl.where(x >= 0, x, x * negative_slope)
            tl.store(output_ptr + off, output, mask=mask)


def _choose_block(n_elements, is_fp32):
    if n_elements <= 1024:
        return 1024
    if n_elements <= 32768:
        return triton.next_power_of_2(n_elements)
    if is_fp32:
        return BLOCK_32
    return BLOCK_16


def _launch_leaky_relu(input_ptr, output_ptr, n_elements, negative_slope):
    if n_elements == 0:
        return
    is_fp32 = input_ptr.dtype == torch.float32
    block_size = _choose_block(n_elements, is_fp32)
    grid = min(triton.cdiv(n_elements, block_size), NUM_SIPS * GRID_MULT)
    even_path = False and (n_elements % block_size == 0)
    with torch_device_fn.device(input_ptr.device):
        _leaky_relu_kernel[(grid,)](
            input_ptr,
            output_ptr,
            n_elements,
            negative_slope,
            BLOCK_SIZE=block_size,
            KEEP_FP32=False,
            MUL_SIGN=False,
            EVEN_PATH=even_path,
            num_warps=NUM_WARPS,
        )


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS_ENFLAME LEAKY_RELU")
    if not A.is_contiguous():
        A = A.contiguous()
    output = torch.empty_like(A)
    _launch_leaky_relu(A, output, A.numel(), negative_slope)
    return output


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS_ENFLAME LEAKY_RELU_")
    if not A.is_contiguous():
        # 非连续 view 无法按 1D 平坦地址线性寻址；先在连续副本上走同一 Triton
        # kernel，再按原 stride 写回，保持 in-place 语义与返回别名不变。
        # 必须透传 negative_slope，否则会静默退化成默认 0.01。
        A.copy_(leaky_relu(A, negative_slope))
        return A
    _launch_leaky_relu(A, A, A.numel(), negative_slope)
    return A


def leaky_relu_out(A, negative_slope=0.01, *, out=None):
    logger.debug("GEMS_ENFLAME LEAKY_RELU_OUT")
    if out is None:
        return leaky_relu(A, negative_slope)
    if not A.is_contiguous():
        A = A.contiguous()
    _launch_leaky_relu(A, out, A.numel(), negative_slope)
    return out
