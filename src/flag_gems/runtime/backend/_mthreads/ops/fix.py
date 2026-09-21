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

from flag_gems.ops.fix import fix as default_fix
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256, "VEC": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 4}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048, "VEC": 1}, num_warps=8, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit
def fix_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dtype_size,  # used for autotune key
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(0)
    BLOCK_ELEMS: tl.constexpr = BLOCK_SIZE * VEC
    offsets = (pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)).to(tl.int64)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # torch.fix == truncate toward zero; compute in fp32 for stability.
    # Moore Threads hardware does not support fp64.
    xf = x.to(tl.float32)
    y = tl.where(xf < 0.0, tl.ceil(xf), tl.floor(xf))
    out = y.to(x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)


def _use_triton_kernel(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not x.is_contiguous() or x.numel() == 0:
        return False
    return True


def _launch_fix(x: torch.Tensor, out: torch.Tensor):
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    n_elements = out_flat.numel()
    dtype_size = out_flat.element_size()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"] * META["VEC"]),)
    with torch_device_fn.device(out.device):
        fix_kernel[grid](x_flat, out_flat, n_elements, dtype_size)
    return out


def fix(input):
    logger.debug("GEMS_MTHREADS FIX")
    if not _use_triton_kernel(input):
        return default_fix(input)

    out = torch.empty_like(input)
    return _launch_fix(input, out)
