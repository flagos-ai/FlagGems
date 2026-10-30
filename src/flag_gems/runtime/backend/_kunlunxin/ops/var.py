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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("var"))
@triton.jit(do_not_specialize=["correction"])
def _var_dim_kernel(
    out,
    x,
    N,
    correction,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    row = tl.program_id(0)
    if ONE_TILE_PER_CTA:
        offsets = tl.arange(0, TILE_N)
        mask = offsets < N
        safe_offsets = tl.minimum(offsets, N - 1)
        values = tl.load(x + row * N + safe_offsets).to(tl.float32)
        values = tl.where(mask, values, 0.0)
        total_sum = tl.sum(values, axis=0)
        total_sum_sq = tl.sum(values * values, axis=0)
    else:
        sum_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        sum_sq_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for start in range(0, N, TILE_N):
            offsets = start + tl.arange(0, TILE_N)
            mask = offsets < N
            safe_offsets = tl.minimum(offsets, N - 1)
            values = tl.load(x + row * N + safe_offsets).to(tl.float32)
            values = tl.where(mask, values, 0.0)
            sum_acc += values
            sum_sq_acc += values * values
        total_sum = tl.sum(sum_acc, axis=0)
        total_sum_sq = tl.sum(sum_sq_acc, axis=0)

    mean = total_sum / N
    denominator = N - correction
    variance = (total_sum_sq / N) - (mean * mean)
    result = tl.where(denominator > 0, variance * N / denominator, float("nan"))
    tl.store(out + row, result)


@libentry()
@triton.jit(do_not_specialize=["correction"])
def _var_dim_row_kernel(
    out,
    x,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program reduces BLOCK_M rows at once (inner reduction along the N
    # axis), so the many-row / small-N dim-reduction cases are no longer
    # launch-bound by one program per row.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    rows_c = tl.minimum(rows, M - 1)

    sum_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    sum_sq_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        cols_c = tl.minimum(cols, N - 1)
        a = tl.load(x + rows_c[:, None] * N + cols_c[None, :]).to(tl.float32)
        a = tl.where(row_mask[:, None] & col_mask[None, :], a, 0.0)
        sum_acc += a
        sum_sq_acc += a * a

    total_sum = tl.sum(sum_acc, axis=1)
    total_sum_sq = tl.sum(sum_sq_acc, axis=1)
    mean = total_sum / N
    denominator = N - correction
    variance = (total_sum_sq / N) - (mean * mean)
    result = tl.where(denominator > 0, variance * N / denominator, float("nan"))
    tl.store(out + rows, result.to(out.dtype.element_ty), mask=row_mask)


def _normalize_dims(x, dim):
    if dim is None:
        return tuple(range(x.ndim))
    if isinstance(dim, int):
        dim = (dim,)
    else:
        dim = tuple(dim)
    if not dim:
        return tuple(range(x.ndim))

    ndim = max(x.ndim, 1)
    normalized = []
    for axis in dim:
        if axis < -ndim or axis >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {axis})"
            )
        axis %= ndim
        if axis in normalized:
            raise RuntimeError(f"dim {axis} appears multiple times in the list of dims")
        normalized.append(axis)
    return () if x.ndim == 0 else tuple(normalized)


def _output_shape(x, dims, keepdim):
    if keepdim:
        return tuple(1 if index in dims else size for index, size in enumerate(x.shape))
    return tuple(size for index, size in enumerate(x.shape) if index not in dims)


def _var_impl(x, dim, correction, keepdim):
    dims = _normalize_dims(x, dim)
    N = math.prod(x.shape[index] for index in dims)
    if N == 0:
        return torch.full(
            _output_shape(x, dims, keepdim),
            float("nan"),
            dtype=x.dtype,
            device=x.device,
        )

    work = dim_compress(x, dims).reshape(-1, N)
    M = work.shape[0]
    result = torch.empty(M, dtype=x.dtype, device=x.device)
    with torch_device_fn.device(x.device):
        if M >= 128 and N <= 512:
            # Many rows with a short inner reduction: amortize the per-program
            # launch/reduction overhead over BLOCK_M rows.
            BLOCK_M = 128
            BLOCK_N = max(64, triton.next_power_of_2(N))
            grid = (triton.cdiv(M, BLOCK_M),)
            _var_dim_row_kernel[grid](
                result, work, M, N, float(correction), BLOCK_M, BLOCK_N
            )
        else:
            _var_dim_kernel[(M,)](result, work, N, float(correction))
    return result.reshape(_output_shape(x, dims, keepdim))


def var(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN VAR")
    effective_correction = 1 if correction is None else correction
    return _var_impl(x, dim, effective_correction, keepdim)


def var_dim(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN VAR_DIM")
    effective_correction = 1 if correction is None else correction
    return _var_impl(x, dim, effective_correction, keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN VAR_CORRECTION")
    effective_correction = 1 if correction is None else correction
    return _var_impl(x, dim, effective_correction, keepdim)
