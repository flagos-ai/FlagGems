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

import builtins
import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

TOTAL_CORE_NUM = 16
F32_INT_MAX = (1 << 24) - 1
_MAX_COL_TILE = 2048


def col_tile_size(n, max_tile=2048):
    return min(max_tile, triton.next_power_of_2(builtins.max(n, 1)))


def d0_chunk_size(stride_d0):
    if stride_d0 <= 0:
        return 1 << 30
    return builtins.max(1, F32_INT_MAX // stride_d0)


logger = logging.getLogger(__name__)


def _is_reduce_last_dim(dim: int, ndim: int) -> bool:
    return dim % ndim == ndim - 1


@libentry()
@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


@libentry()
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


@libentry()
@triton.jit
def max_kernel_gsl(
    inp,
    out_value,
    out_index,
    M,
    N,
    iters,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tl.num_programs(0)
    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    min_value = get_dtype_min(dtype)

    for iter_i in range(iters):
        row = iter_i * num_programs + pid
        if row < M:
            result_value = tl.full((), min_value, dtype=acc_type)
            result_index = tl.zeros((), dtype=tl.int64)
            for i in range(0, N, BLOCK_N):
                n_offset = i + tl.arange(0, BLOCK_N)
                offset = row * N + n_offset
                mask = n_offset < N
                inp_vals = tl.load(inp + offset, mask=mask, other=min_value)
                max_value, max_index = tl.max(inp_vals, axis=0, return_indices=True)
                update_mask = max_value > result_value
                result_value = tl.where(update_mask, max_value, result_value)
                result_index = tl.where(update_mask, i + max_index, result_index)
            tl.store(out_value + row, result_value)
            tl.store(out_index + row, result_index)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    min_value = get_dtype_min(dtype)
    result_value = tl.full([BLOCK_M], value=min_value, dtype=acc_type)
    result_index = tl.zeros([BLOCK_M], dtype=tl.int64)
    for i in range(0, N, BLOCK_N):
        n_offset = i + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        max_value, max_index = tl.max(inp_vals, axis=1, return_indices=True)
        update_mask = max_value > result_value
        result_value = tl.where(update_mask, max_value, result_value)
        result_index = tl.where(update_mask, i + max_index, result_index)
    mask1 = m_offset < M
    tl.store(out_value + m_offset, result_value, mask=mask1)
    tl.store(out_index + m_offset, result_index, mask=mask1)


def _launch_max_dim_last(
    inp: torch.Tensor,
    out_value: torch.Tensor,
    out_index: torch.Tensor,
    n: int,
) -> None:
    d0 = inp.shape[0]
    chunk = d0_chunk_size(inp.stride(0))
    if d0 <= chunk:
        m = inp.numel() // n
        row_iters = triton.cdiv(m, TOTAL_CORE_NUM)
        grid = (min(m, TOTAL_CORE_NUM),)
        block_n = col_tile_size(n, _MAX_COL_TILE)
        max_kernel_gsl[grid](
            inp.view(m, n),
            out_value.reshape(m),
            out_index.reshape(m),
            m,
            n,
            row_iters,
            BLOCK_N=block_n,
        )
        return

    for d0_start in range(0, d0, chunk):
        sl = slice(d0_start, min(d0_start + chunk, d0))
        sub = inp[sl]
        m_sub = sub.numel() // n
        row_iters = triton.cdiv(m_sub, TOTAL_CORE_NUM)
        grid = (min(m_sub, TOTAL_CORE_NUM),)
        block_n = col_tile_size(n, _MAX_COL_TILE)
        max_kernel_gsl[grid](
            sub.view(m_sub, n),
            out_value[sl].reshape(m_sub),
            out_index[sl].reshape(m_sub),
            m_sub,
            n,
            row_iters,
            BLOCK_N=block_n,
        )


def max(inp):
    logger.debug("GEMS_TSINGMICRO MAX")
    inp = inp.contiguous()
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        max_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        max_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def max_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_TSINGMICRO MAX DIM")
    assert dim is not None, "dim must be specified"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"

    shape = list(inp.shape)
    dim = dim % inp.ndim
    n = shape[dim]
    shape[dim] = 1
    use_fast = inp.is_contiguous() and _is_reduce_last_dim(dim, inp.ndim)

    out_value = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    if not use_fast:
        inp = dim_compress(inp, dim)
    m = inp.numel() // n

    if m == 0 or n == 0:
        Max_out = namedtuple("max", ["values", "indices"])
        return Max_out(values=out_value, indices=out_index)

    with torch_device_fn.device(inp.device):
        if use_fast:
            _launch_max_dim_last(inp, out_value, out_index, n)
        else:
            grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),)
            max_kernel[grid](inp, out_value, out_index, m, n)

    Max_out = namedtuple("max", ["values", "indices"])
    return Max_out(values=out_value, indices=out_index)
