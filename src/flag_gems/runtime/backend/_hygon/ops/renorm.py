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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def renorm_kernel_norms_hygon(
    X,
    norms_out,
    M,
    N,
    p_val,
    BLOCK_SIZE: tl.constexpr,
):
    """Hygon-optimized kernel to compute p-norms."""
    pid = tle.program_id(0)

    if tl.constexpr(X.dtype.element_ty == tl.float16) or tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    row_offset = pid * N
    x_ptr_row = X + row_offset
    norm_ptr = norms_out + pid

    _sum = tl.zeros([BLOCK_SIZE], dtype=cdtype)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_vals = tl.load(x_ptr_row + cols, mask=mask, other=0.0).to(cdtype)
        abs_vals = tl.abs(x_vals)
        if p_val == 2.0:
            powered = x_vals * x_vals
        else:
            powered = tl_extra_shim.pow(abs_vals, p_val)
        _sum += powered

    sum_val = tl.sum(_sum)
    if p_val == 2.0:
        norm = tl_extra_shim.sqrt(sum_val)
    else:
        norm = tl_extra_shim.pow(sum_val, 1.0 / p_val)

    tl.store(norm_ptr, norm)


@libentry()
@triton.jit
def renorm_kernel_scale_hygon(
    X,
    norms_in,
    Y,
    M,
    N,
    p_val,
    maxnorm,
    BLOCK_SIZE: tl.constexpr,
):
    """Hygon-optimized kernel to apply scaling."""
    pid = tle.program_id(0)

    if tl.constexpr(X.dtype.element_ty == tl.float16) or tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    row_offset = pid * N
    x_ptr_row = X + row_offset
    y_ptr_row = Y + row_offset
    norm = tl.load(norms_in + pid)

    if norm <= maxnorm:
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x_vals = tl.load(x_ptr_row + cols, mask=mask, other=0.0)
            tl.store(y_ptr_row + cols, x_vals, mask=mask)
    else:
        scale = maxnorm / norm
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x_vals = tl.load(x_ptr_row + cols, mask=mask, other=0.0).to(cdtype)
            y_vals = x_vals * scale
            tl.store(y_ptr_row + cols, y_vals.to(X.dtype.element_ty), mask=mask)


def renorm(input, p, dim, maxnorm):
    logger.debug("GEMS_HYGON RENORM")

    if dim < 0:
        dim = input.ndim + dim

    # Handle dim 0 case efficiently with single-kernel-per-row approach
    if dim == 0:
        M = input.shape[0]
        N = input.numel() // M

        input = input.contiguous()
        norms = torch.empty((M,), dtype=torch.float32, device=input.device)

        # Hygon-optimized block size: use 256 for better occupancy
        BLOCK = min(triton.next_power_of_2(N), 256)
        grid = (M,)

        with torch_device_fn.device(input.device):
            renorm_kernel_norms_hygon[grid](
                input,
                norms,
                M,
                N,
                p,
                BLOCK_SIZE=BLOCK,
            )

        output = torch.empty_like(input)

        with torch_device_fn.device(input.device):
            renorm_kernel_scale_hygon[grid](
                input,
                norms,
                output,
                M,
                N,
                p,
                maxnorm,
                BLOCK_SIZE=BLOCK,
            )

        return output
    else:
        # For non-zero dim, use permute to make dim=0
        ndim = input.ndim
        perm = list(range(ndim))
        perm.remove(dim)
        perm.insert(0, dim)
        inv_perm = [perm.index(i) for i in range(ndim)]

        x_perm = input.permute(perm)
        result = renorm(x_perm, p, 0, maxnorm)
        return result.permute(inv_perm)


def renorm_(input, p, dim, maxnorm):
    logger.debug("GEMS_HYGON RENORM_")

    if dim < 0:
        dim = input.ndim + dim

    if dim == 0:
        M = input.shape[0]
        N = input.numel() // M

        input = input.contiguous()
        norms = torch.empty((M,), dtype=torch.float32, device=input.device)

        # Hygon-optimized block size
        BLOCK = min(triton.next_power_of_2(N), 256)
        grid = (M,)

        with torch_device_fn.device(input.device):
            renorm_kernel_norms_hygon[grid](
                input,
                norms,
                M,
                N,
                p,
                BLOCK_SIZE=BLOCK,
            )

        with torch_device_fn.device(input.device):
            renorm_kernel_scale_hygon[grid](
                input,
                norms,
                input,
                M,
                N,
                p,
                maxnorm,
                BLOCK_SIZE=BLOCK,
            )

        return input
    else:
        # For non-zero dim, use permute to make dim=0
        ndim = input.ndim
        perm = list(range(ndim))
        perm.remove(dim)
        perm.insert(0, dim)
        inv_perm = [perm.index(i) for i in range(ndim)]

        x_perm = input.permute(perm)
        result = renorm_(x_perm, p, 0, maxnorm)
        input.copy_(result.permute(inv_perm))
        return input
