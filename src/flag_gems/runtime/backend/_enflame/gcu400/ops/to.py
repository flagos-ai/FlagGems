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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
)
@triton.jit
def to_dtype_func(x):
    return x


@triton.jit
def i64_i32_to_dtype_func(in_ptr, out_ptr, num_elem_per_grid, num_elem):
    grid_id = tl.program_id(0)

    start = grid_id * num_elem_per_grid
    end = tl.minimum(start + num_elem_per_grid, num_elem)

    for offset in range(num_elem_per_grid):
        current_offset = start + offset

        if current_offset < end:
            current_out_ptr = out_ptr + current_offset
            current_in_ptr = in_ptr + current_offset
            x = tl.load(current_in_ptr).to(out_ptr.type.element_ty)
            tl.store(current_out_ptr, x)


def to_dtype(x, dtype, non_blocking=False, copy=False, memory_format=None):
    logger.debug("GEMS_ENFLAME TO_DTYPE")
    if not copy and x.dtype == dtype:
        return x
    out = torch.empty_like(x, dtype=dtype, memory_format=memory_format)

    cond1 = x.dtype == torch.int64 and dtype == torch.int32
    cond2 = x.dtype == torch.int32 and dtype == torch.int64
    if cond1 or cond2:
        num_elem = x.numel()
        max_grid_size = 65532
        if num_elem <= max_grid_size:
            grid_size = num_elem
            num_elem_per_grid = 1
        else:
            grid_size = max_grid_size
            num_elem_per_grid = (num_elem + max_grid_size - 1) // max_grid_size

        grid = (grid_size,)
        i64_i32_to_dtype_func[grid](x, out, num_elem_per_grid, num_elem, num_warps=1)
        return out

    return to_dtype_func(x, out0=out)
