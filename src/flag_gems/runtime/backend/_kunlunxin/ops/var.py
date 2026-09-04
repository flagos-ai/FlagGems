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
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
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
        values = tl.load(x + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(values, axis=0) / N
        diff = values - mean
        squared_sum = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0)
    else:
        sum_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for start in range(0, N, TILE_N):
            offsets = start + tl.arange(0, TILE_N)
            mask = offsets < N
            values = tl.load(x + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_acc += values
        mean = tl.sum(sum_acc, axis=0) / N

        squared_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for start in range(0, N, TILE_N):
            offsets = start + tl.arange(0, TILE_N)
            mask = offsets < N
            values = tl.load(x + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
            diff = values - mean
            squared_acc += tl.where(mask, diff * diff, 0.0)
        squared_sum = tl.sum(squared_acc, axis=0)

    denominator = N - correction
    variance = squared_sum / denominator
    result = tl.where(denominator > 0, variance, float("nan"))
    tl.store(out + row, result)


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
    result = torch.empty(work.shape[0], dtype=x.dtype, device=x.device)
    with torch_device_fn.device(x.device):
        _var_dim_kernel[(work.shape[0],)](result, work, N, float(correction))
    return result.reshape(_output_shape(x, dims, keepdim))


def var(x, unbiased=True):
    logger.debug("GEMS_KUNLUNXIN VAR")
    return _var_impl(x, None, 1 if unbiased else 0, False)


def var_dim(x, dim=None, unbiased=True, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN VAR_DIM")
    return _var_impl(x, dim, 1 if unbiased else 0, keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN VAR_CORRECTION")
    effective_correction = 1 if correction is None else correction
    return _var_impl(x, dim, effective_correction, keepdim)
