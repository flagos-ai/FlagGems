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

from flag_gems.ops.log10 import log10_ as default_log10_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.jit
def _log10_inplace_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    LOG10_2: tl.constexpr = 0.3010299956639812
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.log2(x.to(tl.float32)) * LOG10_2
    tl.store(x_ptr + offs, y.to(x.dtype), mask=mask)


def _use_triton_kernel(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not x.is_contiguous() or x.numel() == 0:
        return False
    return True


def log10_(x):
    logger.debug("GEMS_MTHREADS LOG10_")
    if not _use_triton_kernel(x):
        return default_log10_(x)

    n_elements = x.numel()
    # Hardcoded BLOCK_SIZE/num_warps: the draft kernel was tuned with a fixed
    # block of 1024 and 4 warps; kept static per Rule 21 (autotune on an inplace
    # kernel would require restore_value=["x_ptr"] to avoid buffer corruption).
    block = 1024
    grid = (triton.cdiv(n_elements, block),)
    with torch_device_fn.device(x.device):
        _log10_inplace_kernel[grid](x, n_elements, BLOCK=block, num_warps=4)
    return x


__all__ = ["log10_"]
