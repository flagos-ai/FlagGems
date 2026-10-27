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

from flag_gems.ops.renorm_ import renorm_ as default_renorm_  # fallback
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

pow = tl_extra_shim.pow

# Largest slice length handled by the single-pass kernel (one tile per row).
_MAX_SINGLE_PASS_N = 8192


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
    ],
    key=["N", "dtype_size"],
)
@triton.jit(do_not_specialize=["p", "maxnorm"])
def renorm_kernel(X, N, p, maxnorm, dtype_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = X + pid * N

    inv_p = 1.0 / p

    # Pass 1: accumulate the p-norm of this slice in fp32 (mthreads has no
    # fp64). Drop the loaded tiles from L2 (``evict_first``) since they will be
    # re-read in pass 2 -- this keeps the cache warm for the rescaling store.
    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(
            row_ptr + idx, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        acc += tl.sum(pow(tl.abs(x), p))

    norm = pow(acc, inv_p)
    # Slices whose norm already fits within maxnorm are left unchanged.
    scale = tl.where(norm > maxnorm, maxnorm / norm, 1.0)

    # Pass 2: rescale every element of the slice in place.
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(row_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        tl.store(row_ptr + idx, x * scale, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
    ],
    key=["N", "dtype_size"],
)
@triton.jit(do_not_specialize=["p", "maxnorm"])
def renorm_kernel_single_pass(X, N, p, maxnorm, dtype_size, BLOCK_SIZE: tl.constexpr):
    # Single-pass fast path: the whole slice fits in one tile, so we can load
    # every element exactly once, reduce it in registers, and rescale it in
    # place -- halving the memory traffic of the two-pass kernel.
    pid = tl.program_id(0)
    row_ptr = X + pid * N

    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(row_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    norm = pow(tl.sum(pow(tl.abs(x), p)), 1.0 / p)
    # Slices whose norm already fits within maxnorm are left unchanged.
    scale = tl.where(norm > maxnorm, maxnorm / norm, 1.0)

    tl.store(row_ptr + idx, x * scale, mask=mask)


def _use_triton_kernel(x: torch.Tensor) -> Tuple[bool, int]:
    if not isinstance(x, torch.Tensor):
        return False, 0
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False, 0
    if x.numel() == 0 or not x.is_contiguous():
        return False, 0
    return True, x.element_size()


def renorm_(x, p, dim, maxnorm):
    logger.debug("GEMS_MTHREADS RENORM_")
    use_triton, dtype_size = _use_triton_kernel(x)
    if not use_triton:
        return default_renorm_(x, p, dim, maxnorm)

    dim = dim % x.ndim
    # Normalize scalars to float once; mthreads kernels compute in fp32 only.
    p = float(p)
    maxnorm = float(maxnorm)

    def _launch(x_flat, M, N):
        grid = (M,)
        with torch_device_fn.device(x.device):
            if N <= _MAX_SINGLE_PASS_N:
                renorm_kernel_single_pass[grid](x_flat, N, p, maxnorm, dtype_size)
            else:
                renorm_kernel[grid](x_flat, N, p, maxnorm, dtype_size)

    if dim == x.ndim - 1:
        # Fast in-place path: the reduced dimension is the last (contiguous)
        # axis, so each slice is already a contiguous run in memory and no
        # transpose / materialization / copy-back is required.
        N = x.shape[-1]
        M = x.numel() // N
        _launch(x.reshape(M, N), M, N)
        return x

    # General path: move the reduced dimension to the front so that each slice
    # becomes a contiguous row, run the kernel, then scatter the result back.
    perm = [dim] + [i for i in range(x.ndim) if i != dim]
    x_perm = x.permute(perm).contiguous()
    M = x_perm.shape[0]
    N = x_perm[0].numel()
    x_flat = x_perm.reshape(M, N)
    _launch(x_flat, M, N)

    inv_perm = [0] * x.ndim
    for i, pi in enumerate(perm):
        inv_perm[pi] = i
    x.copy_(x_flat.reshape(x_perm.shape).permute(inv_perm))

    return x
