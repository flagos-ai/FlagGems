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

logger = logging.getLogger(__name__)


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


def _chunked_row_sum(values):
    rows, columns = values.shape
    chunk_size = 65536
    full_chunks = columns // chunk_size
    if full_chunks:
        result = (
            values[:, : full_chunks * chunk_size]
            .reshape(rows, full_chunks, chunk_size)
            .sum(dim=2)
            .sum(dim=1)
        )
    else:
        result = torch.zeros(rows, dtype=values.dtype, device=values.device)
    if full_chunks * chunk_size < columns:
        result += values[:, full_chunks * chunk_size :].sum(dim=1)
    return result


def _var_impl(x, dim, correction, keepdim):
    dims = _normalize_dims(x, dim)
    count = math.prod(x.shape[index] for index in dims)
    if count == 0:
        return torch.full(
            _output_shape(x, dims, keepdim),
            float("nan"),
            dtype=x.dtype,
            device=x.device,
        )

    remaining_dims = tuple(index for index in range(x.ndim) if index not in dims)
    work = x.permute(remaining_dims + dims).contiguous().to(torch.float32)
    rows = x.numel() // count
    work = work.reshape(rows, count)
    mean = _chunked_row_sum(work) / count
    squared_deviation = (work - mean[:, None]) * (work - mean[:, None])
    result = _chunked_row_sum(squared_deviation) / (count - correction)
    return result.reshape(_output_shape(x, dims, keepdim)).to(x.dtype)


def var(x, unbiased=True):
    logger.debug("GEMS_KUNLUNXIN VAR")
    return _var_impl(x, None, 1 if unbiased else 0, False)


def var_dim(x, dim=None, unbiased=True, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN VAR_DIM")
    return _var_impl(x, dim, 1 if unbiased else 0, keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN VAR_CORRECTION")
    if correction is None:
        correction = 1
    return _var_impl(x, dim, correction, keepdim)
