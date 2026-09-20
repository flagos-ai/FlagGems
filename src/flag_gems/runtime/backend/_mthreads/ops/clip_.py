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

from flag_gems.ops.clip import clip_ as default_clip_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.jit
def clip_kernel(
    x_ptr,
    mini,
    maxi,
    n_elements,
    HAS_MIN: tl.constexpr,
    HAS_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    pid = tl.program_id(0)
    step = BLOCK * GRID
    for start in range(pid * BLOCK, n_elements, step):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        if HAS_MAX:
            x = tl.minimum(x, maxi)
        if HAS_MIN:
            x = tl.maximum(x, mini)
        tl.store(x_ptr + offsets, x, mask=mask)


def _use_triton_kernel(x: torch.Tensor, mini, maxi) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not x.is_contiguous() or x.numel() == 0:
        return False
    # Only scalar min/max are specialized here; tensor bounds fall back to generic.
    for v in (mini, maxi):
        if v is not None and not isinstance(v, (int, float)):
            return False
    return True


def clip_(x: torch.Tensor, mini=None, maxi=None):
    logger.debug("GEMS_MTHREADS CLIP_")
    if not _use_triton_kernel(x, mini, maxi):
        return default_clip_(x, mini, maxi)

    has_min = mini is not None
    has_max = maxi is not None
    mini_v = float(mini) if has_min else 0.0
    maxi_v = float(maxi) if has_max else 0.0
    n = x.numel()
    # Persistent grid-stride loop: BLOCK=1024, GRID=60*8 sized for the S5000
    # (60 SMs). Hardcoded (not autotuned) because this is an inplace kernel —
    # autotune would rerun on the same buffer and corrupt the data.
    BLOCK = 1024
    GRID = 60 * 8
    grid = (min(triton.cdiv(n, BLOCK), GRID),)
    with torch_device_fn.device(x.device):
        clip_kernel[grid](
            x,
            mini_v,
            maxi_v,
            n,
            HAS_MIN=has_min,
            HAS_MAX=has_max,
            BLOCK=BLOCK,
            GRID=GRID,
            num_warps=4,
        )
    return x
