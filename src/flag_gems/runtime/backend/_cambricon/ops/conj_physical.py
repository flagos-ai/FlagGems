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

from ..utils import MAX_GRID_SIZE_X

logger = logging.getLogger(__name__)


@triton.jit
def _conj_physical_kernel(
    in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, M: tl.constexpr
):
    grid_0 = tl.num_programs(0)
    pid = tl.program_id(0)
    while pid < M:
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        base = offsets * 2
        real = tl.load(in_ptr + base, mask=mask)
        imag = tl.load(in_ptr + base + 1, mask=mask)

        tl.store(out_ptr + base, real, mask=mask)
        tl.store(out_ptr + base + 1, -imag, mask=mask)
        pid += grid_0


@triton.jit
def _copy_kernel(
    in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, M: tl.constexpr
):
    grid_0 = tl.num_programs(0)
    pid = tl.program_id(0)
    while pid < M:
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x, mask=mask)
        pid += grid_0


def conj_physical(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_CAMBRICON CONJ_PHYSICAL")

    src = input if input.is_contiguous() else input.contiguous()
    n_elements = src.numel()
    if n_elements == 0:
        return torch.empty_like(src)

    block_size = 1024
    m = triton.cdiv(n_elements, block_size)
    grid = (min(m, MAX_GRID_SIZE_X),)

    with torch_device_fn.device(input.device):
        output = torch.empty_like(src)
        if input.is_complex():
            in_real_ptr = torch.view_as_real(src)
            out_real_ptr = torch.view_as_real(output)
            _conj_physical_kernel[grid](
                in_real_ptr,
                out_real_ptr,
                n_elements,
                BLOCK_SIZE=block_size,
                M=m,
            )
        else:
            _copy_kernel[grid](src, output, n_elements, BLOCK_SIZE=block_size, M=m)

    return output
