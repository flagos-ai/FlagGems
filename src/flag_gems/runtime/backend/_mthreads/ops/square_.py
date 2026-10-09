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

from flag_gems.ops.square import square_ as default_square_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=2, num_stages=1),
    ],
    key=["n_elements"],
    # Inplace: autotune reruns the kernel on the same buffer, so restore the
    # input between trials to avoid squaring repeatedly in place.
    restore_value=["x_ptr"],
)
@triton.jit
def square_kernel_full(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, eviction_policy="evict_first")
    tl.store(x_ptr + offsets, x * x, eviction_policy="evict_first")


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=2, num_stages=1),
    ],
    key=["n_elements"],
    # Inplace: autotune reruns the kernel on the same buffer, so restore the
    # input between trials to avoid squaring repeatedly in place.
    restore_value=["x_ptr"],
)
@triton.jit
def square_kernel_masked(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy="evict_first")
    tl.store(x_ptr + offsets, x * x, mask=mask, eviction_policy="evict_first")


def _use_triton_kernel(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not x.is_contiguous() or x.numel() == 0:
        return False
    return True


def square_(x: torch.Tensor):
    logger.debug("GEMS_MTHREADS SQUARE_")
    if not _use_triton_kernel(x):
        return default_square_(x)

    n = x.numel()
    with torch_device_fn.device(x.device):
        if n % 4096 == 0:
            grid = lambda META: (n // META["BLOCK_SIZE"],)
            square_kernel_full[grid](x, n)
        else:
            grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)
            square_kernel_masked[grid](x, n)
    return x
