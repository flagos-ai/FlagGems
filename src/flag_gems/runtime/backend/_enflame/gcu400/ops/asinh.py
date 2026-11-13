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

# 常驻 CTA 数与 GCU400 SIP 拓扑对齐；跨步循环保证大输入不超 grid.x 硬件上限。
# BLOCK 8192 为实测最优值：更大 BLOCK 会拉低 fp16 大 shape 带宽利用率。
_GRID_SIZE = 48
_BLOCK_SIZE = 8192
_NUM_WARPS = 4
_NUM_STAGES = 4


@libentry()
@triton.jit
def asinh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        # 统一提升到 fp32 计算，16bit 输入不会在中间量里丢失精度。
        value = x.to(tl.float32)
        abs_value = tl.abs(value)
        # 数值稳定的 asinh：先用 where 选出 log 的输入，再只做一次 log。
        # 大数分支不依赖 x*x；被丢弃分支即使溢出也不影响结果，
        # 且 nan / inf 会自然传播到最终输出。
        sqrt_term = tl.sqrt(abs_value * abs_value + 1.0)
        log_input = tl.where(abs_value > 1e18, 2.0 * abs_value, abs_value + sqrt_term)
        magnitude = tl.log(log_input)
        # 位运算恢复符号：保留 -0.0，同时避免一次 tl.where 分支。
        sign_bits = value.to(tl.int32, bitcast=True) & -2147483648
        magnitude_bits = tl.abs(magnitude).to(tl.int32, bitcast=True)
        result = (magnitude_bits | sign_bits).to(tl.float32, bitcast=True)
        tl.store(y_ptr + offsets, result, mask=mask)


def _launch(x_contig: torch.Tensor, out_contig: torch.Tensor):
    n_elements = out_contig.numel()
    if n_elements == 0:
        return
    num_tile = triton.cdiv(n_elements, _BLOCK_SIZE)
    active_cta_count = min(_GRID_SIZE, num_tile)
    grid = (active_cta_count, 1, 1)
    with torch_device_fn.device(out_contig.device.index):
        asinh_kernel[grid](
            x_contig,
            out_contig,
            n_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
            GRID_SIZE=active_cta_count,
            NUM_STAGES=_NUM_STAGES,
            num_warps=_NUM_WARPS,
        )


def asinh(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME ASINH")
    if x.is_floating_point():
        result_dtype = x.dtype
    else:
        result_dtype = torch.float32

    x_contig = x.to(result_dtype).contiguous()
    out = torch.empty_like(x_contig)
    _launch(x_contig.view(-1), out.view(-1))
    return out.view_as(x)


def asinh_(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME ASINH_")
    if not x.is_floating_point():
        x.copy_(asinh(x))
        return x

    if x.is_contiguous():
        _launch(x.view(-1), x.view(-1))
    else:
        x_contig = x.contiguous()
        _launch(x_contig.view(-1), x_contig.view(-1))
        x.copy_(x_contig)
    return x


def asinh_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME ASINH_OUT")
    # 按 x 的原始精度计算，写出时由 Triton 自动转换到 out.dtype：
    # 既避免提前降精度，又省去一次额外的 dtype 转换拷贝。
    x_contig = x.contiguous()

    if out.is_contiguous():
        _launch(x_contig.view(-1), out.view(-1))
    else:
        out_contig = torch.empty_like(out, memory_format=torch.contiguous_format)
        _launch(x_contig.view(-1), out_contig.view(-1))
        out.copy_(out_contig)
    return out
