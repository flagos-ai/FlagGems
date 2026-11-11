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

logger = logging.getLogger(__name__)

_ROW_BLOCK = 32
_IN_TILE = 1024
_TARGET_ELEMS = 4096
_SPLIT_ELEMS = 32768
_MAX_SPLIT = 1024
_SINGLE_MIN_ROWS = 64
_SINGLE_MAX_REDUCE = 1 << 18
_ROW_TILE = 4096


def _warps_for(block_n, block_m):
    # Two float32 accumulators per element of the tile: keep the tile per thread
    # small enough that they stay in registers.
    elems = block_n * block_m
    if elems <= 512:
        return 4
    if elems <= 1024:
        return 8
    return 16


@libentry()
@triton.jit(do_not_specialize=["correction", "N"])
def _var_rows_kernel(
    out,
    x,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = row < M
    x = x + row * N

    sum_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask & (cols < N)
        values = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
        sum_acc += values
        sq_acc += values * values

    total = tl.sum(sum_acc, axis=1)
    total_sq = tl.sum(sq_acc, axis=1)
    count = N.to(tl.float32)
    mean = total / count
    denominator = count - correction
    variance = (total_sq - total * mean) / denominator
    variance = tl.where(denominator > 0, variance, float("nan"))
    tl.store(out + row, variance[:, None], row_mask)


@libentry()
@triton.jit(do_not_specialize=["N", "nsplit", "chunk"])
def _var_partial_kernel(
    part_sum,
    part_sq,
    x,
    M,
    N,
    nsplit,
    chunk,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    base = x + row.to(tl.int64) * N + pid * chunk
    limit = N - pid * chunk

    sum_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sq_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, chunk, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < limit
        values = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        sum_acc += values
        sq_acc += values * values

    off = row.to(tl.int64) * nsplit + pid
    tl.store(part_sum + off, tl.sum(sum_acc))
    tl.store(part_sq + off, tl.sum(sq_acc))


@libentry()
@triton.jit(do_not_specialize=["correction", "N", "nsplit"])
def _var_combine_kernel(
    out,
    part_sum,
    part_sq,
    M,
    N,
    nsplit,
    correction,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * nsplit

    sum_acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    sq_acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for start in range(0, nsplit, BLOCK_S):
        idx = start + tl.arange(0, BLOCK_S)
        mask = idx < nsplit
        sum_acc += tl.load(part_sum + base + idx, mask=mask, other=0.0)
        sq_acc += tl.load(part_sq + base + idx, mask=mask, other=0.0)

    total = tl.sum(sum_acc)
    total_sq = tl.sum(sq_acc)
    count = N.to(tl.float32)
    mean = total / count
    denominator = count - correction
    variance = (total_sq - total * mean) / denominator
    variance = tl.where(denominator > 0, variance, float("nan"))
    tl.store(out + row, variance)


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


def _resolve_correction(correction, unbiased):
    if isinstance(correction, bool):
        return 1 if correction else 0
    if isinstance(unbiased, bool):
        return 1 if unbiased else 0
    return 1 if correction is None else correction


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
    out = torch.empty(M, dtype=x.dtype, device=x.device)

    with torch_device_fn.device(x.device):
        if M >= _SINGLE_MIN_ROWS or N <= _SINGLE_MAX_REDUCE:
            block_m = max(1, min(_ROW_BLOCK, _TARGET_ELEMS // max(N, 1)))
            block_m = triton.next_power_of_2(block_m)
            block_n = min(_ROW_TILE, triton.next_power_of_2(N))
            warps = _warps_for(block_n, block_m)
            _var_rows_kernel[(triton.cdiv(M, block_m),)](
                out,
                work,
                M,
                N,
                correction,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=warps,
            )
        else:
            block_n = min(_IN_TILE, triton.next_power_of_2(N))
            nsplit = min(_MAX_SPLIT, max(1, triton.cdiv(M * N, _SPLIT_ELEMS)))
            chunk = triton.cdiv(triton.cdiv(N, nsplit), block_n) * block_n
            nsplit = triton.cdiv(N, chunk)
            part_sum = torch.empty(M * nsplit, dtype=torch.float32, device=x.device)
            part_sq = torch.empty(M * nsplit, dtype=torch.float32, device=x.device)
            _var_partial_kernel[(nsplit, M)](
                part_sum,
                part_sq,
                work,
                M,
                N,
                nsplit,
                chunk,
                BLOCK_N=block_n,
            )
            _var_combine_kernel[(M,)](
                out,
                part_sum,
                part_sq,
                M,
                N,
                nsplit,
                correction,
                BLOCK_S=min(_IN_TILE, triton.next_power_of_2(nsplit)),
            )

    return out.reshape(_output_shape(x, dims, keepdim))


def var(x, dim=None, correction=None, keepdim=False, unbiased=None):
    logger.debug("GEMS_KUNLUNXIN VAR")
    if isinstance(dim, bool):
        unbiased = dim
        dim = None
    return _var_impl(x, dim, _resolve_correction(correction, unbiased), keepdim)


def var_dim(x, dim=None, correction=None, keepdim=False, unbiased=None):
    logger.debug("GEMS_KUNLUNXIN VAR_DIM")
    return _var_impl(x, dim, _resolve_correction(correction, unbiased), keepdim)


def var_correction(x, dim=None, correction=None, keepdim=False, unbiased=None):
    logger.debug("GEMS_KUNLUNXIN VAR_CORRECTION")
    return _var_impl(x, dim, _resolve_correction(correction, unbiased), keepdim)
