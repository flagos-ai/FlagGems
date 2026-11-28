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

logger = logging.getLogger(__name__)

# logging's isEnabledFor is evaluated once at import time: the host side of a single call to this op is only about 45us,
# and running the argument packing and level check of logger.debug again on every call would eat a few percentage
# points.
_LOG_DEBUG = logger.isEnabledFor(logging.DEBUG)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["start", "step", "size"])
def arange_func(
    y_ptr, start, step, size, BLOCK_SIZE: tl.constexpr, GRID_DIM: tl.constexpr
):
    pid = tl.program_id(0)
    num_tiles = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tiles, GRID_DIM):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < size
        vals = (
            (offs * step + start).to(tl.float32)
            if step != 1
            else (offs + start).to(tl.float32)
        )
        tl.store(y_ptr + offs, vals, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["start", "step", "size"])
def arange_int_func(
    y_ptr, start, step, size, BLOCK_SIZE: tl.constexpr, GRID_DIM: tl.constexpr
):
    pid = tl.program_id(0)
    num_tiles = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tiles, GRID_DIM):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < size
        vals = offs * step + start
        tl.store(y_ptr + offs, vals, mask=mask)


def arange_start(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    if _LOG_DEBUG:
        logger.debug("GEMS_ENFLAME ARANGE")

    # ATen infers int64 only when every bound is integral; a float start/end/step
    # selects the default floating dtype. Mirroring that avoids truncating a
    # fractional step (e.g. 1/70) to zero before the size calculation.
    if dtype is None:
        integer_args = all(
            isinstance(v, (int, torch.SymInt)) for v in (start, end, step)
        )
        dtype = torch.int64 if integer_args else torch.get_default_dtype()

    if dtype is torch.int64:
        start = int(start)
        end = int(end)
        step = int(step)
        if step == 0:
            raise RuntimeError("step must be nonzero")
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    else:
        if step == 0:
            raise RuntimeError("step must be nonzero")
        size = math.ceil((end - start) / step)
    size = max(int(size), 0)

    if pin_memory is None:
        pin_memory = False
    if device is None:
        device = runtime.device.name

    if size == 0:
        return torch.empty((0,), device=device, dtype=dtype, pin_memory=pin_memory)

    # Measured optimal tiling for a pure write-bandwidth kernel (gcu400): BLOCK fixed at 8192, grid capped at
    # 2*NUM_SIPS.
    # Larger BLOCKs (16384/32768) are actually slower on large shapes; small shapes use a single tile to save grid
    # overhead.
    if size <= 1024:
        BLOCK_SIZE = 1024
        grid_dim = 1
    elif size <= 8192:
        BLOCK_SIZE = triton.next_power_of_2(size)
        grid_dim = 1
    else:
        BLOCK_SIZE = 8192
        grid_dim = min(-(-size // BLOCK_SIZE), NUM_SIPS * 2)
    nw = 4

    is_int = dtype in (torch.int32, torch.int64, torch.int16, torch.int8)

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)

    # The host side of a single call is about 47us (of which device context switching is about 9us); for small shapes
    # like size=4096 this
    # is the dominant cost, so switch context only when the target device is not the current device.
    if result.device.index == torch_device_fn.current_device():
        if is_int:
            arange_int_func[(grid_dim,)](
                result,
                int(start),
                int(step),
                size,
                BLOCK_SIZE=BLOCK_SIZE,
                GRID_DIM=grid_dim,
                num_warps=nw,
            )
        else:
            arange_func[(grid_dim,)](
                result,
                float(start),
                float(step),
                size,
                BLOCK_SIZE=BLOCK_SIZE,
                GRID_DIM=grid_dim,
                num_warps=nw,
            )
    else:
        with torch_device_fn.device(result.device):
            if is_int:
                arange_int_func[(grid_dim,)](
                    result,
                    int(start),
                    int(step),
                    size,
                    BLOCK_SIZE=BLOCK_SIZE,
                    GRID_DIM=grid_dim,
                    num_warps=nw,
                )
            else:
                arange_func[(grid_dim,)](
                    result,
                    float(start),
                    float(step),
                    size,
                    BLOCK_SIZE=BLOCK_SIZE,
                    GRID_DIM=grid_dim,
                    num_warps=nw,
                )

    return result


def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ENFLAME ARANGE")
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
