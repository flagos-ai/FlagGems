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

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Size-based BLOCK_SIZE/num_warps heuristics (mirrors the sibling generation op
# `arange` on this backend). The original hard-coded BLOCK_SIZE=128 spawned
# cdiv(steps, 128) programs (up to ~8M for 1G steps), each writing only 128
# elements -> launch-bound. 8192/8 is the measured sweet spot for the large
# store-heavy kernel; smaller steps keep smaller blocks to avoid tail waste.
_BLOCK_SMALL = 256
_WARPS_SMALL = 2
_BLOCK_MED = 1024
_WARPS_MED = 4
_BLOCK_LARGE = 4096
_WARPS_LARGE = 8
_BLOCK_BIG = 8192
_WARPS_BIG = 8
_SINGLE_BLOCK = 256


@libentry()
@triton.jit
def linspace_single_kernel(out_ptr, value, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + idx, value, mask=idx < 1)


@libentry()
@triton.jit
def linspace_fp_kernel(
    out_ptr,
    start,
    step_size,
    steps,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Single forward-formula store-only kernel for floating point dtypes.
    # `base` is a scalar so the whole value is one fma on the vector of
    # consecutive indices: value[i] = start + (pid*BLOCK_SIZE + i) * step.
    pid = ext.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base = (pid * BLOCK_SIZE * step_size + start).to(tl.float32)
    cols = tl.arange(0, BLOCK_SIZE).to(tl.float32)
    out_val = cols * step_size + base
    if NEED_MASK:
        tl.store(out_ptr + idx, out_val, mask=idx < steps)
    else:
        tl.store(out_ptr + idx, out_val)


@libentry()
@triton.jit
def linspace_int_kernel(
    out_ptr,
    start,
    mid,
    end,
    step_size,
    steps,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Two-sided (forward/backward) formula for integer dtypes, mirroring torch's
    # exact truncation semantics: fw = start + step*idx, bd = end - step*(steps-1-idx),
    # choose fw for idx < mid (so the head is exact) and bd otherwise (so the
    # tail is exact). 1-ULP forward-only drift would flip the last element by 1
    # after truncation, hence the two-sided form.
    pid = ext.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    fw_values = start + (step_size * idx)
    bd_values = end - step_size * (steps - idx - 1)
    out_val = tl.where(idx < mid, fw_values, bd_values)
    if NEED_MASK:
        tl.store(out_ptr + idx, out_val, mask=idx < steps)
    else:
        tl.store(out_ptr + idx, out_val)


def _block_config(steps):
    if steps <= 1024:
        return _BLOCK_SMALL, _WARPS_SMALL
    elif steps <= 8192:
        return _BLOCK_MED, _WARPS_MED
    elif steps <= 65536:
        return _BLOCK_LARGE, _WARPS_LARGE
    else:
        return _BLOCK_BIG, _WARPS_BIG


def linspace(
    start, end, steps, *, dtype=None, layout=None, device=None, pin_memory=None
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN LINSPACE")
    assert steps >= 1, "steps must be >= 1"

    out = torch.empty(
        steps,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    if isinstance(start, torch.Tensor):
        start = start.item()
    if isinstance(end, torch.Tensor):
        end = end.item()

    if steps == 1:
        # Single element: value is `start` (cast to out dtype on store).
        linspace_single_kernel[(1,)](
            out, float(start), BLOCK_SIZE=_SINGLE_BLOCK, num_warps=1
        )
        return out

    mid = steps // 2
    step_size = (float(end) - float(start)) / (steps - 1)
    BLOCK_SIZE, num_warps = _block_config(steps)
    NEED_MASK = (steps % BLOCK_SIZE) != 0
    grid = (triton.cdiv(steps, BLOCK_SIZE),)

    if out.dtype.is_floating_point:
        linspace_fp_kernel[grid](
            out,
            start,
            step_size,
            steps,
            BLOCK_SIZE=BLOCK_SIZE,
            NEED_MASK=NEED_MASK,
            num_warps=num_warps,
        )
    else:
        linspace_int_kernel[grid](
            out,
            start,
            mid,
            end,
            step_size,
            steps,
            BLOCK_SIZE=BLOCK_SIZE,
            NEED_MASK=NEED_MASK,
            num_warps=num_warps,
        )
    return out
