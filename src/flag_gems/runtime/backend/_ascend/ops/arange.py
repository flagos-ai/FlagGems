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
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
MAX_GRID_DIM = 65535


@libentry()
@triton.jit
def arange_func(
    y_ptr,
    start,
    end,
    step,
    size,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = ext.program_id(0)
    num_programs = tl.num_programs(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
        num_programs = num_programs.to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    if INT64_INDEX:
        cols = cols.to(tl.int64)
    for block_start in range(pid * BLOCK_SIZE, size, num_programs * BLOCK_SIZE):
        offset = block_start + cols
        tl.store(y_ptr + offset, start + step * offset, mask=offset < size)


def arange_start(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    logger.debug("GEMS_ASCEND ARANGE")
    if dtype is None:
        integer_args = all(
            isinstance(v, (int, torch.SymInt)) for v in (start, end, step)
        )
        dtype = torch.int64 if integer_args else torch.get_default_dtype()

    if step == 0:
        raise RuntimeError("step must be nonzero")
    if (step > 0 and end < start) or (step < 0 and end > start):
        raise RuntimeError("upper bound and lower bound inconsistent with step sign")

    if dtype is torch.int64:
        start = int(start)
        end = int(end)
        step = int(step)
        if step == 0:
            raise RuntimeError("step must be nonzero")
        sgn = (step > 0) - (step < 0)
        size = max(0, (end - start + step - sgn) // step)
    else:
        size = math.ceil((end - start) / step)
    size = max(0, int(size))

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = (
            runtime.device.name
        )  # Note(Zhengzekang): Torch default value is CPU, but triton is target to GPU.

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    if size == 0:
        return result

    BLOCK_SIZE = 128
    grid = min(triton.cdiv(size, BLOCK_SIZE), MAX_GRID_DIM)
    wide = size >= 2147483648
    with torch_device_fn.device(result.device):
        arange_func[(grid,)](
            result, start, end, step, size, BLOCK_SIZE, INT64_INDEX=wide
        )
    return result


def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
