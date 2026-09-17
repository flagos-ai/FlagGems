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

from flag_gems.ops.fmod import fmod_scalar_ as default_fmod_scalar_
from flag_gems.ops.fmod import fmod_tensor_ as default_fmod_tensor_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _fmod_fp32(x, y):
    # fmod rounds the quotient toward zero (C fmod / torch.fmod semantics),
    # unlike the modulo operator which floors. Compute in fp32 so half/bf16
    # inputs keep full intermediate precision before narrowing back.
    q = x / y
    q_trunc = tl.where(q >= 0, tl.floor(q), tl.ceil(q))
    return x - y * q_trunc


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256, "VEC": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 4}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 2}, num_warps=8, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit
def fmod_inplace_tensor_kernel(
    x_ptr,
    y_ptr,
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
    y = tl.load(y_ptr + offsets, mask=mask)

    out = _fmod_fp32(x.to(tl.float32), y.to(tl.float32)).to(x.dtype)

    tl.store(x_ptr + offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256, "VEC": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 4}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 2}, num_warps=8, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit(do_not_specialize=["scalar"])
def fmod_inplace_scalar_kernel(
    x_ptr,
    scalar,
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

    y = tl.full((1,), scalar, tl.float32)
    out = _fmod_fp32(x.to(tl.float32), y).to(x.dtype)

    tl.store(x_ptr + offsets, out, mask=mask)


def _use_triton_kernel(A) -> bool:
    if not isinstance(A, torch.Tensor):
        return False
    if A.device.type != "musa" or A.dtype not in _SUPPORTED_DTYPES:
        return False
    if not A.is_contiguous() or A.numel() == 0:
        return False
    return True


def _launch_tensor(A: torch.Tensor, B: torch.Tensor):
    x_flat = A.view(-1)
    y_flat = B.view(-1)
    n_elements = x_flat.numel()
    dtype_size = x_flat.element_size()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"] * META["VEC"]),)
    with torch_device_fn.device(A.device):
        fmod_inplace_tensor_kernel[grid](x_flat, y_flat, n_elements, dtype_size)
    return A


def _launch_scalar(A: torch.Tensor, scalar: float):
    x_flat = A.view(-1)
    n_elements = x_flat.numel()
    dtype_size = x_flat.element_size()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"] * META["VEC"]),)
    with torch_device_fn.device(A.device):
        fmod_inplace_scalar_kernel[grid](x_flat, scalar, n_elements, dtype_size)
    return A


def fmod_tensor_(A, B):
    logger.debug("GEMS_MTHREADS FMOD_")
    if (
        _use_triton_kernel(A)
        and isinstance(B, torch.Tensor)
        and B.shape == A.shape
        and B.dtype == A.dtype
        and B.is_contiguous()
    ):
        return _launch_tensor(A, B)
    return default_fmod_tensor_(A, B)


def fmod_scalar_(A, B):
    logger.debug("GEMS_MTHREADS FMOD_")
    if _use_triton_kernel(A) and not isinstance(B, torch.Tensor):
        try:
            scalar = float(B)
        except Exception:
            return default_fmod_scalar_(A, B)
        return _launch_scalar(A, scalar)
    return default_fmod_scalar_(A, B)


def fmod_(A, B):
    logger.debug("GEMS_MTHREADS FMOD_")
    if isinstance(B, torch.Tensor):
        return fmod_tensor_(A, B)
    return fmod_scalar_(A, B)
