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
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.round import round_ as default_round_  # fallback
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _round_half_to_even(x):
    """Round to nearest, ties to even. x must be fp32."""
    r = tl.floor(x)
    d = x - r  # fractional part in [0, 1)
    is_odd = tl.abs(r - 2.0 * tl.floor(r / 2.0)) > 0.5
    return tl.where((d > 0.5) | ((tl.abs(d - 0.5) < 1e-10) & is_odd), r + 1.0, r)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit
def round_inplace_kernel(
    x_ptr,
    n_elements,
    dtype_size,  # used for autotune key
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    out = _round_half_to_even(x_fp32).to(x.dtype)
    tl.store(x_ptr + offsets, out, mask=mask)


def _use_triton_kernel(x, decimals) -> Tuple[bool, int]:
    if not isinstance(x, torch.Tensor):
        return False, 0
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False, 0
    if x.numel() == 0 or not x.is_contiguous():
        return False, 0
    # Only the common decimals == 0 path is specialized here; the scaled
    # variant is rare and left to the generic implementation.
    if decimals != 0:
        return False, 0
    return True, x.element_size()


def round_(input, *, decimals=0):
    logger.debug("GEMS_MTHREADS ROUND_")
    use_triton, dtype_size = _use_triton_kernel(input, decimals)
    if not use_triton:
        return default_round_(input, decimals=decimals)

    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(input.device):
        round_inplace_kernel[grid](input, n_elements, dtype_size)
    return input
