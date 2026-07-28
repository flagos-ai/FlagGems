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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def aminmax_kernel_1(
    inp,
    min_out,
    max_out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype == tl.bfloat16 else dtype

    min_fill = get_dtype_max(inp.type.element_ty)
    max_fill = get_dtype_min(inp.type.element_ty)
    min_val = tl.load(inp_ptrs, mask=mask, other=min_fill).to(acc_type)
    max_val = tl.load(inp_ptrs, mask=mask, other=max_fill).to(acc_type)

    min_val = tl.min(min_val)
    max_val = tl.max(max_val)

    min_ptr = min_out + pid
    max_ptr = max_out + pid
    tl.store(min_ptr, min_val.to(dtype))
    tl.store(max_ptr, max_val.to(dtype))


@libentry()
@triton.jit
def aminmax_kernel_2(
    min_inp, max_inp, min_out, max_out, mid_size, BLOCK_MID: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    min_ptrs = min_inp + offset
    max_ptrs = max_inp + offset
    mask = offset < mid_size

    dtype = min_inp.type.element_ty
    acc_type = tl.float32 if dtype == tl.bfloat16 else dtype

    min_fill = get_dtype_max(min_inp.type.element_ty)
    max_fill = get_dtype_min(max_inp.type.element_ty)
    min_val = tl.load(min_ptrs, mask=mask, other=min_fill).to(acc_type)
    max_val = tl.load(max_ptrs, mask=mask, other=max_fill).to(acc_type)

    min_val = tl.min(min_val)
    max_val = tl.max(max_val)

    tl.store(min_out, min_val.to(dtype))
    tl.store(max_out, max_val.to(dtype))


def _aminmax_block_n(n):
    # Restrict BLOCK_N to a power of two that does not exceed N or 512.
    # This avoids Triton XPU bugs triggered when BLOCK_N == 1024 with
    # small N (e.g. shape (1, 2), dim=0 produces N=1 in the dim path).
    return max(1, min(triton.next_power_of_2(n), 512))


def _aminmax_block_m(m):
    # Heuristic for the row-block size; choose the largest BLOCK_M that does
    # not exceed M (capped to one of the tune-space values).
    if m <= 1:
        return 1
    if m <= 2:
        return 2
    if m <= 4:
        return 4
    return 8


@libentry()
@triton.jit
def aminmax_kernel(
    inp,
    min_out,
    max_out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    max_value = get_dtype_max(dtype)

    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    min_out = min_out + rows
    max_out = max_out + rows
    row_mask = rows < M

    acc_type = tl.float32 if dtype == tl.bfloat16 else dtype
    _min = tl.full([BLOCK_M, BLOCK_N], value=max_value, dtype=acc_type)
    _max = tl.full([BLOCK_M, BLOCK_N], value=min_value, dtype=acc_type)
    # Build acc_type-typed sentinel values so tl.where does not promote the
    # tensor operands to float32 (which would change _min/_max dtype inside
    # the loop on the Triton XPU backend).
    max_value_t = tl.full([], value=max_value, dtype=acc_type)
    min_value_t = tl.full([], value=min_value, dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(inp + cols, mask=mask, other=0.0).to(acc_type)
        a_min = tl.where(mask, a, max_value_t)
        a_max = tl.where(mask, a, min_value_t)
        _min = tl.minimum(_min, a_min)
        _max = tl.maximum(_max, a_max)
    min_result = tl.min(_min, axis=1)[:, None]
    max_result = tl.max(_max, axis=1)[:, None]
    tl.store(min_out, min_result, row_mask)
    tl.store(max_out, max_result, row_mask)


def aminmax(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS_KUNLUNXIN AMINMAX")

    if dim is None:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        min_mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        max_mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)

        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
            if not keepdim:
                min_out = min_out.squeeze()
                max_out = max_out.squeeze()
        else:
            if not keepdim:
                min_out = torch.empty([], dtype=dtype, device=inp.device)
                max_out = torch.empty([], dtype=dtype, device=inp.device)
            else:
                shape = [1] * inp.dim()
                min_out = torch.empty(shape, dtype=dtype, device=inp.device)
                max_out = torch.empty(shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            aminmax_kernel_1[(mid_size, 1)](
                inp,
                min_mid,
                max_mid,
                M,
                block_size,
            )
            aminmax_kernel_2[(1, 1)](
                min_mid, max_mid, min_out, max_out, mid_size, block_mid
            )
        return min_out, max_out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
        else:
            min_out = torch.empty(shape, dtype=dtype, device=inp.device)
            max_out = torch.empty(shape, dtype=dtype, device=inp.device)

        block_m = _aminmax_block_m(M)
        block_n = _aminmax_block_n(N)
        grid = (triton.cdiv(M, block_m),)
        with torch_device_fn.device(inp.device):
            aminmax_kernel[grid](
                inp, min_out, max_out, M, N, BLOCK_M=block_m, BLOCK_N=block_n
            )
        if not keepdim:
            min_out = min_out.squeeze(dim=dim)
            max_out = max_out.squeeze(dim=dim)
        return min_out, max_out
