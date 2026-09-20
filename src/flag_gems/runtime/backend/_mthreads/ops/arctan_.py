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

from flag_gems.ops.arctan_ import arctan_ as default_arctan_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_atan = tl_extra_shim.atan

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

# Above this many elements, streaming out-of-place beats in-place on S5000
# (working set far exceeds L2; separate write stream avoids read/write aliasing).
HUGE_THRESHOLD = 1 << 27


@libentry()
@triton.jit
def arctan_inplace_kernel(
    x_ptr, n_elements, BLOCK_SIZE: tl.constexpr, VEC: tl.constexpr, EVEN: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * (BLOCK_SIZE * VEC)
    for i in tl.static_range(VEC):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if EVEN:
            x = tl.load(x_ptr + offsets)
            y = _atan(x.to(tl.float32)).to(x_ptr.dtype.element_ty)
            tl.store(x_ptr + offsets, y)
        else:
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = _atan(x.to(tl.float32)).to(x_ptr.dtype.element_ty)
            tl.store(x_ptr + offsets, y, mask=mask)


@libentry()
@triton.jit
def arctan_oop_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy="evict_first")
    y = _atan(x.to(tl.float32)).to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + offsets, y, mask=mask, eviction_policy="evict_first")


def _use_triton_kernel(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not x.is_contiguous() or x.numel() == 0:
        return False
    return True


def arctan_(x):
    logger.debug("GEMS_MTHREADS ARCTAN_")
    if not _use_triton_kernel(x):
        return default_arctan_(x)

    n = x.numel()
    with torch_device_fn.device(x.device):
        if n >= HUGE_THRESHOLD:
            out = torch.empty_like(x)
            grid = (triton.cdiv(n, 2048),)
            arctan_oop_kernel[grid](x, out, n, BLOCK_SIZE=2048, num_warps=4)
            return out
        # dtype-tuned vectorization: fp16 uses wider VEC.
        if x.dtype == torch.float16:
            vec, block = 4, 1024
        else:
            vec, block = 2, 1024
        grid = (triton.cdiv(n, block * vec),)
        even = (n % (block * vec)) == 0
        arctan_inplace_kernel[grid](
            x, n, BLOCK_SIZE=block, VEC=vec, EVEN=even, num_warps=4
        )
        return x


__all__ = ["arctan_"]
