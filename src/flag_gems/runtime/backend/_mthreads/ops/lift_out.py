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

from flag_gems.ops.lift import lift_out as default_lift_out

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _lift_copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, value, mask=mask)


def _specialized_lift_out(A, *, out=None):
    if out is None:
        out = torch.empty_like(A)
    src = A
    if not src.is_contiguous():
        src = src.contiguous()
    n = A.numel()
    if n == 0:
        return out
    if n <= 65536:
        BLOCK_SIZE = 512
        num_warps = 4
    elif n <= 268435456:
        BLOCK_SIZE = 8192 // A.element_size()
        num_warps = 2
    else:
        BLOCK_SIZE = 1024 // A.element_size()
        num_warps = 1
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _lift_copy_kernel[grid](src, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return out


def lift_out(A, *, out=None):
    logger.debug("GEMS_MTHREADS LIFT_OUT")
    if (
        isinstance(A, torch.Tensor)
        and A.device.type == "musa"
        and A.dtype in _SUPPORTED_DTYPES
        and isinstance(out, torch.Tensor)
        and out.device.type == "musa"
        and out.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_lift_out(A, out=out)
    return default_lift_out(A, out=out)
