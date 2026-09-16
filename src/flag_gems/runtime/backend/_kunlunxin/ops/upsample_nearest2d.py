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
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
device = device.name

# Upper bound on the per-program BLOCK_SIZE (number of output lanes handled by
# one program). The unbounded `next_pow2(cdiv(total, 12))` produced tiles of up
# to 33.5M lanes for pathological test shapes, which both bloats the compiled IR
# and triggers the known XPU big-tile slowdown. A hard cap keeps every tile small
# and the grid bounded.
MAX_BLOCK_SIZE = 262144


# @triton.autotune(
#     configs=runtime.get_tuned_config("upsample_nearest2d"), key=["N", "C", "OH", "OW"]
# )
@triton.jit
def upsample_nearest2d_kernel(
    ptr_o,
    ptr_i,
    N: tl.constexpr,
    C: tl.constexpr,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    total,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ow = idx % OW
    oh = idx // OW % OH
    c = idx // OW // OH % C
    n = idx // OW // OH // C % N
    if SAME_H:
        ih = oh
    else:
        # tl.floor() cannot be found in 2.3.1, using int trunc
        ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)
    offset_o = ((n * C + c) * OH + oh) * OW + ow
    offset_i = ((n * C + c) * IH + ih) * IW + iw
    data = tl.load(ptr_i + offset_i)
    if NEED_MASK:
        # Tail block may extend past total output elements; guard the store.
        # (offset_i is always in-bounds: oh/ow/n/c are derived via modulo.)
        tl.store(ptr_o + offset_o, data, mask=idx < total)
    else:
        tl.store(ptr_o + offset_o, data)


def upsample_nearest2d(
    input: torch.Tensor,
    output_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_NEAREST2D")
    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    OH, OW = output_size
    N, C, IH, IW = input.shape
    if scales_h is not None:
        reciprocal_scale_h = 1 / scales_h
    else:
        reciprocal_scale_h = IH / OH
    if scales_w is not None:
        reciprocal_scale_w = 1 / scales_w
    else:
        reciprocal_scale_w = IW / OW
    # allocate output
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    total_threads = N * C * OH * OW
    # Bounded block size: keep the usual ~12-program cluster layout for small
    # outputs, but never build a huge single-program tile.
    block_size = min(
        triton.next_power_of_2(triton.cdiv(total_threads, 12)), MAX_BLOCK_SIZE
    )
    need_mask = total_threads % block_size != 0
    grid = (triton.cdiv(total_threads, block_size),)
    with torch_device_fn.device(input.device):
        upsample_nearest2d_kernel[grid](
            output,
            input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            total_threads,
            SAME_H=(OH == IH),
            SAME_W=(OW == IW),
            BLOCK_SIZE=block_size,
            NEED_MASK=need_mask,
        )
    return output
