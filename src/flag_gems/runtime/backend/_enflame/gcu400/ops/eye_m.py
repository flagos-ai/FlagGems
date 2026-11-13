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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
device_ = device


@libentry()
@triton.jit
def eye_kernel(out_ptr, N, M, BLOCK_M: tl.constexpr):
    # 每个 program 负责若干整行：行内连续写，对角线判定退化为 1D 标量比较
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_M)
    col_mask = cols < M
    for row in tl.range(pid, N, num_programs):
        val = tl.where(cols == row, 1.0, 0.0)
        tl.store(out_ptr + row * M + cols, val, mask=col_mask)


def eye_m(n, m, *, dtype=None, layout=torch.strided, device=None, pin_memory=None):
    """
    Triton-based implementation of torch.eye_m(n, m), using 2D tiles to split the matrix into blocks.
    """
    logger.debug("GEMS_ENFLAME EYE_M")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    if layout != torch.strided:
        raise ValueError("Currently only strided layout is supported for eye_m.")

    out = torch.empty(
        (n, m), dtype=dtype, device=device, layout=layout, pin_memory=pin_memory
    )
    BLOCK_M = triton.next_power_of_2(m)
    grid = (min(24, n),)
    with torch_device_fn.device(device):
        eye_kernel[grid](out, n, m, BLOCK_M, num_warps=4)

    return out
