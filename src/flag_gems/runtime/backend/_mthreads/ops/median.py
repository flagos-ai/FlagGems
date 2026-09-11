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

from flag_gems.ops.median import (
    MedianResult,
    _canonical_dim,
    _copy_out,
    _has_names,
    _name_to_dim,
)
from flag_gems.ops.median import median as default_median
from flag_gems.ops.median import median_dim as default_median_dim
from flag_gems.ops.topk import _get_iinfo_val
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64/int64 compute. The in-register
# sort kernel below handles the fixed-width dtypes listed here; any other dtype
# falls back to the generic implementation for correctness.
_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
}

# Above this reduction width an in-register sort becomes too costly; hand those
# rows to the generic implementation (which switches to a binary-search select).
_SORT_SELECT_LIMIT = 8192


@libentry()
@triton.jit
def median_sort_select_kernel(
    inp,
    values,
    indices,
    M,
    N,
    RANK,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    cols = tl.arange(0, BLOCK_N)
    valid = (cols[None, :] < N) & row_mask[:, None]
    ptrs = inp + rows[:, None] * N + cols[None, :]

    is_float: tl.constexpr = inp.dtype.element_ty.is_floating()
    rank_mask = cols[None, :] == RANK

    if is_float:
        # Sort in fp32: the mthreads LLVM backend cannot lower bf16 comparisons.
        high = tl.full((), float("inf"), dtype=tl.float32)
        data = tl.load(ptrs, mask=valid, other=0.0)
        fdata = data.to(tl.float32)
        nan_mask = valid & (fdata != fdata)
        sortable = tl.where(valid & ~nan_mask, fdata, high)
        ordered = tl.sort(sortable, dim=1, descending=False)
        median_key = tl.sum(
            tl.where(rank_mask, ordered, tl.zeros_like(ordered)), axis=1
        )
        first_match = tl.argmax(
            (valid & ~nan_mask & (fdata == median_key[:, None])).to(tl.int32), axis=1
        )
        nan_i32 = nan_mask.to(tl.int32)
        has_nan = tl.max(nan_i32, axis=1) != 0
        first_nan = tl.argmax(nan_i32, axis=1)
        first_match = tl.where(has_nan, first_nan, first_match)
    else:
        high = tl.full(
            (),
            _get_iinfo_val(inp.dtype.element_ty, return_max=True),
            dtype=inp.dtype.element_ty,
        )
        data = tl.load(ptrs, mask=valid, other=high)
        sortable = tl.where(valid, data, high)
        ordered = tl.sort(sortable, dim=1, descending=False)
        median_key = tl.sum(
            tl.where(rank_mask, ordered, tl.zeros_like(ordered)), axis=1
        )
        first_match = tl.argmax(
            (valid & (data == median_key[:, None])).to(tl.int32), axis=1
        )

    # Gather the original element at the selected index so the stored value keeps
    # the input dtype exactly (also preserves signed zero / NaN payloads).
    selected_value = tl.load(inp + rows * N + first_match, mask=row_mask, other=0)

    tl.store(values + rows, selected_value, mask=row_mask)
    tl.store(indices + rows, first_match.to(tl.int64), mask=row_mask)


def _use_triton_kernel(inp):
    if not isinstance(inp, torch.Tensor):
        return False
    if inp.device.type != "musa":
        return False
    if inp.dtype not in _SUPPORTED_DTYPES:
        return False
    if _has_names(inp):
        return False
    return True


def _median_select(work_2d):
    """Lower-median value and a selecting index for each row of a (M, N) tensor."""
    m, n = work_2d.shape
    device = work_2d.device

    values = torch.empty((m,), dtype=work_2d.dtype, device=device)
    indices = torch.empty((m,), dtype=torch.int64, device=device)

    block_n = triton.next_power_of_2(n)
    if block_n >= 512:
        block_m = 1
    elif block_n >= 128:
        block_m = 4
    else:
        block_m = 16
    num_warps = min(8, max(4, block_n // 512)) if block_n >= 128 else 1

    rank = (n - 1) // 2
    grid = (triton.cdiv(m, block_m),)
    with torch_device_fn.device(device):
        median_sort_select_kernel[grid](
            work_2d,
            values,
            indices,
            m,
            n,
            rank,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
    return values, indices


def median(inp):
    logger.debug("GEMS_MTHREADS MEDIAN")
    if (
        not _use_triton_kernel(inp)
        or inp.numel() <= 1
        or inp.numel() > _SORT_SELECT_LIMIT
    ):
        return default_median(inp)

    work_2d = inp.contiguous().reshape(1, inp.numel())
    values, _ = _median_select(work_2d)
    return values.reshape(())


def median_out(inp, *, out):
    logger.debug("GEMS_MTHREADS MEDIAN.OUT")
    return _copy_out(median(inp), out, "out")


def median_dim(inp, dim=0, keepdim=False):
    logger.debug("GEMS_MTHREADS MEDIAN.DIM")
    if not _use_triton_kernel(inp) or inp.ndim == 0:
        return default_median_dim(inp, dim=dim, keepdim=keepdim)

    # Resolve/validate the reduction dim exactly like the generic path so
    # out-of-range dims still raise IndexError before we launch any kernel.
    if isinstance(dim, str):
        dim = _name_to_dim(inp, dim)
    dim = _canonical_dim(inp.ndim, dim)

    if inp.shape[dim] == 0 or inp.numel() == 0 or inp.shape[dim] > _SORT_SELECT_LIMIT:
        return default_median_dim(inp, dim=dim, keepdim=keepdim)

    work = torch.movedim(inp, dim, -1).contiguous()
    batch_shape = work.shape[:-1]
    n = work.shape[-1]
    m = work.numel() // n
    work_2d = work.reshape(m, n)

    values, indices = _median_select(work_2d)
    values = values.reshape(batch_shape)
    indices = indices.reshape(batch_shape)

    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)

    return MedianResult(values=values, indices=indices)


def median_dim_values(inp, dim=0, keepdim=False, *, values, indices):
    logger.debug("GEMS_MTHREADS MEDIAN.DIM_VALUES")
    result = median_dim(inp, dim=dim, keepdim=keepdim)
    _copy_out(result.values, values, "values")
    _copy_out(result.indices, indices, "indices")
    return MedianResult(values=values, indices=indices)
