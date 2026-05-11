import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["start", "step", "size"])
def arange_func(y_ptr, start, step, size,
                BLOCK_SIZE: tl.constexpr, GRID_DIM: tl.constexpr):
    pid = tl.program_id(0)
    num_tiles = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tiles, GRID_DIM):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < size
        vals = (offs * step + start).to(tl.float32) if step != 1 else (offs + start).to(tl.float32)
        tl.store(y_ptr + offs, vals, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["start", "step", "size"])
def arange_int_func(y_ptr, start, step, size,
                    BLOCK_SIZE: tl.constexpr, GRID_DIM: tl.constexpr):
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
    logger.debug("GEMS ARANGE")
    if dtype is None:
        dtype = torch.int64

    if dtype is torch.int64:
        start = int(start)
        end = int(end)
        step = int(step)
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    else:
        size = math.ceil((end - start) / step)
    size = max(int(size), 0)

    if pin_memory is None:
        pin_memory = False
    if device is None:
        device = runtime.device.name

    if size == 0:
        return torch.empty((0,), device=device, dtype=dtype, pin_memory=pin_memory)

    BLOCK_SIZE = 1024
    if size > 65536:
        BLOCK_SIZE = 65536
    elif size > 1024:
        BLOCK_SIZE = min(triton.next_power_of_2(size), 8192)

    grid_dim = min(triton.cdiv(size, BLOCK_SIZE), NUM_SIPS * 2)
    nw = 4

    is_int = dtype in (torch.int32, torch.int64, torch.int16, torch.int8)

    if dtype == torch.int64:
        compute_dtype = torch.int32
    else:
        compute_dtype = dtype

    result = torch.empty((size,), device=device, dtype=compute_dtype, pin_memory=pin_memory)

    with torch_device_fn.device(result.device):
        if is_int:
            arange_int_func[(grid_dim,)](
                result, int(start), int(step), size,
                BLOCK_SIZE=BLOCK_SIZE, GRID_DIM=grid_dim, num_warps=nw,
            )
        else:
            arange_func[(grid_dim,)](
                result, float(start), float(step), size,
                BLOCK_SIZE=BLOCK_SIZE, GRID_DIM=grid_dim, num_warps=nw,
            )

    if dtype == torch.int64:
        result = result.to(torch.int64)

    return result


def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def arange_func(y_ptr, start, end, step, size, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    y_ptr += pid * BLOCK_SIZE
    step_offset = pid * BLOCK_SIZE * step

    cols = tl.arange(0, BLOCK_SIZE)
    arange_val = cols * step + step_offset + start
    mask = cols + pid * BLOCK_SIZE
    tl.store(y_ptr + cols, arange_val, mask=mask < size)


def arange_start(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    logger.debug("GEMS ARANGE")
    if dtype is torch.int64:
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    else:
        size = math.ceil((end - start) / step)

    if size <= 65536:
        BLOCK_SIZE = min(triton.next_power_of_2(size), 8192)
        nw = 4
    else:
        BLOCK_SIZE = 65536
        nw = 1
    while triton.cdiv(size, BLOCK_SIZE) > 65535:
        BLOCK_SIZE *= 2
    grid = triton.cdiv(size, BLOCK_SIZE)

    if dtype is None:
        dtype = torch.int64

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = runtime.device.name

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    if size > 65536:
        arange_func[grid,](result, start, end, step, size, BLOCK_SIZE, num_warps=nw)
    else:
        arange_func[grid,](result, start, end, step, size, BLOCK_SIZE)
    return result


def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
