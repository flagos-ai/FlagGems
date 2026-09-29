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

from flag_gems.ops.mish import mish_ as default_mish_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.jit
def mish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    # mish(x) = x * tanh(softplus(x)) = x * u * (u + 2) / (u * u + 2 * u + 2),
    # where u = exp(x). u is computed via exp2(x * log2e) to hit the fast
    # hardware exp2 path on Moore Threads.
    LOG2E: tl.constexpr = 1.4426950408889634
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN:
        x = tl.load(x_ptr + offs)
        xf = x.to(tl.float32)
        u = tl.exp2(xf * LOG2E)
        y = xf * u * (u + 2.0) / (u * u + 2.0 * u + 2.0)
        tl.store(out_ptr + offs, y.to(x.dtype))
    else:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        u = tl.exp2(xf * LOG2E)
        y = xf * u * (u + 2.0) / (u * u + 2.0 * u + 2.0)
        tl.store(out_ptr + offs, y.to(x.dtype), mask=mask)


def _use_triton_kernel(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not x.is_contiguous() or x.numel() == 0:
        return False
    return True


def _launch_mish(x: torch.Tensor, out: torch.Tensor):
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    n = x_flat.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    with torch_device_fn.device(x.device):
        mish_kernel[grid](
            x_flat,
            out_flat,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
            EVEN=(n % BLOCK_SIZE == 0),
        )
    return out


def mish_(x):
    logger.debug("GEMS_MTHREADS MISH_")
    if not _use_triton_kernel(x):
        return default_mish_(x)
    return _launch_mish(x, x)


__all__ = ["mish_"]
