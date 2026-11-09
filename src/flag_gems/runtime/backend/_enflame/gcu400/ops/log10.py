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

"""GCU400 专属 log10：1D flat kernel + 固定规模 grid，替代 pointwise_dynamic codegen。"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# 固定 program 数匹配 GCU400 计算单元，核内跨步遍历全部 tile。
NUM_PROGRAMS = 48
BLOCK_SIZE = 16384


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def _log10_flat_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    BLOCK: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(program_id, tl.cdiv(num_elements, BLOCK), num_programs):
        element_offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        valid_mask = element_offsets < num_elements
        # 整数/布尔输入在 fp32 中计算，输出 dtype 由 dispatch 决定。
        input_values = tl.load(
            input_ptr + element_offsets, mask=valid_mask, other=0
        ).to(tl.float32)
        result_values = tl.log(input_values) * 0.4342944819032518
        tl.store(output_ptr + element_offsets, result_values, mask=valid_mask)


def _output_dtype(input_tensor: torch.Tensor):
    # 与公共实现的 INT_TO_FLOAT 提升规则一致。
    if input_tensor.is_floating_point():
        return input_tensor.dtype
    return torch.float32


def _launch_log10(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    num_elements = input_tensor.numel()
    if num_elements == 0:
        return output_tensor
    grid_size = min(triton.cdiv(num_elements, BLOCK_SIZE), NUM_PROGRAMS)
    with torch_device_fn.device(input_tensor.device):
        _log10_flat_kernel[(grid_size,)](
            input_tensor,
            output_tensor,
            num_elements,
            BLOCK=BLOCK_SIZE,
            num_warps=2,
        )
    return output_tensor


def log10(A):
    logger.debug("GEMS_ENFLAME LOG10")
    contiguous_input = A if A.is_contiguous() else A.contiguous()
    output = torch.empty(
        contiguous_input.shape, dtype=_output_dtype(A), device=A.device
    )
    return _launch_log10(contiguous_input, output)


def log10_(A):
    logger.debug("GEMS_ENFLAME LOG10_")
    if A.is_contiguous():
        return _launch_log10(A, A)
    # 非连续原地输入先在线性副本上计算，再按原 stride 写回。
    contiguous_copy = A.contiguous()
    _launch_log10(contiguous_copy, contiguous_copy)
    A.copy_(contiguous_copy)
    return A


def log10_out(A, out):
    logger.debug("GEMS_ENFLAME LOG10_OUT")
    contiguous_input = A if A.is_contiguous() else A.contiguous()
    if out.is_contiguous():
        return _launch_log10(contiguous_input, out)
    contiguous_output = torch.empty(out.shape, dtype=out.dtype, device=out.device)
    _launch_log10(contiguous_input, contiguous_output)
    out.copy_(contiguous_output)
    return out
