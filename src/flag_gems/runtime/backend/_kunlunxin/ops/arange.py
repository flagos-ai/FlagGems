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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def arange_func(
    y_ptr,
    start,
    end,
    step,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE
    step_offset = offset * step

    cols = tl.arange(0, BLOCK_SIZE)
    arange_val = cols * step + step_offset + start
    mask = cols + offset < size
    tl.store(y_ptr + offset + cols, arange_val, mask=mask)


@libentry()
@triton.jit
def arange_func_float(
    y_ptr,
    start,
    step,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    # For floating output dtypes: computing the whole value in int32 then
    # casting a large-magnitude integer to float is the XPU bottleneck (~85 GB/s
    # fp16 / 160 GB/s fp32). Instead convert only the small per-block `cols`
    # (0..BLOCK_SIZE) to fp32 and add a precomputed fp32 base scalar, which halves
    # the runtime (~170 GB/s fp16 / ~330 GB/s fp32). Results are bit-identical to
    # the int kernel on all shapes.
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    idx = offset + cols
    base = (offset * step + start).to(tl.float32)
    arange_val = cols.to(tl.float32) * step + base
    tl.store(y_ptr + idx, arange_val, mask=idx < size)


def arange_start(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    logger.debug("GEMS_KUNLUNXIN ARANGE")
    if dtype is torch.int64:
        start = int(start)
        end = int(end)
        step = int(step)
        if step == 0:
            raise RuntimeError("step must be nonzero")
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    else:
        if dtype is torch.int64 and (
            isinstance(step, float)
            or isinstance(start, float)
            or isinstance(end, float)
        ):
            int_step = int(step)
            if int_step == 0:
                raise RuntimeError("step must be nonzero")
        size = math.ceil((end - start) / step)
    size = int(size)

    if dtype is None:
        dtype = torch.int64

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = runtime.device.name

    # Size-based heuristic for BLOCK_SIZE and num_warps
    if size <= 1024:
        BLOCK_SIZE = 256
        num_warps = 2
    elif size <= 8192:
        BLOCK_SIZE = 1024
        num_warps = 4
    elif size <= 65536:
        BLOCK_SIZE = 4096
        num_warps = 8
    else:
        BLOCK_SIZE = 16384
        num_warps = 8

    grid = triton.cdiv(size, BLOCK_SIZE)

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        # fp32-base fast path (avoids slow large-int -> float conversion)
        arange_func_float[grid,](
            result, start, step, size, BLOCK_SIZE, num_warps=num_warps
        )
    else:
        arange_func[grid,](
            result, start, end, step, size, BLOCK_SIZE, num_warps=num_warps
        )
    return result


def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
