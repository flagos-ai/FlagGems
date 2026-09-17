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

from flag_gems.ops.mish import mish as default_mish
from flag_gems.ops.mish import mish_ as default_mish_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

exp = tl_extra_shim.exp
log = tl_extra_shim.log
fast_tanh = tl_extra_shim.fast_tanh


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256, "VEC": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 4}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048, "VEC": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096, "VEC": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096, "VEC": 2}, num_warps=16, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit
def mish_kernel(
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
    x = tl.load(x_ptr + offsets, mask=mask)

    # mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    # compute in fp32 (mthreads has no fp64); tails are stable since for
    # large |x| mish -> x (pos) or 0 (neg) and tanh saturates accordingly.
    x_fp32 = x.to(tl.float32)
    out = (x_fp32 * fast_tanh(log(1.0 + exp(x_fp32)))).to(x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)


def _use_triton_kernel(x: torch.Tensor) -> Tuple[bool, int]:
    if not isinstance(x, torch.Tensor):
        return False, 0
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False, 0
    if x.numel() == 0 or not x.is_contiguous():
        return False, 0
    return True, x.element_size()


def _launch_mish(x: torch.Tensor, out: torch.Tensor, dtype_size: int):
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    n_elements = out_flat.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"] * META["VEC"]),)
    with torch_device_fn.device(out.device):
        mish_kernel[grid](x_flat, out_flat, n_elements, dtype_size)
    return out


def mish(A):
    logger.debug("GEMS_MTHREADS MISH")
    use_triton, dtype_size = _use_triton_kernel(A)
    if not use_triton:
        return default_mish(A)

    out = torch.empty_like(A)
    return _launch_mish(A, out, dtype_size)


def mish_(A):
    logger.debug("GEMS_MTHREADS MISH_")
    use_triton, dtype_size = _use_triton_kernel(A)
    if not use_triton:
        return default_mish_(A)

    return _launch_mish(A, A, dtype_size)
