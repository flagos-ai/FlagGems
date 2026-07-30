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

from flag_gems.ops.nanmedian import (
    MAX_BLOCK_N,
    NanMedian,
    _check_supported_dtype,
    _count_block_n,
    _full_nan_result,
)
from flag_gems.ops.nanmedian import _nanmedian_dim_impl as _generic_nanmedian_dim_impl
from flag_gems.ops.nanmedian import _nanmedian_flat_impl as _generic_nanmedian_flat_impl
from flag_gems.ops.nanmedian import _normalize_dim, count_valid_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _native_kthvalue(inp, k, dim=-1, keepdim=False):
    return torch.ops.aten.kthvalue.default.redispatch(
        _FALLBACK_KEYSET, inp, k, dim, keepdim
    )


def _nanmedian_kthvalue_fallback(inp, M, N):
    inp = inp.reshape(M, N)
    if not torch.is_floating_point(inp):
        values, indices = _native_kthvalue(inp, (N + 1) // 2, dim=1)
        return NanMedian(values=values, indices=indices)

    valid_count = torch.empty((M,), dtype=torch.long, device=inp.device)
    block_n = _count_block_n(inp, N)
    with torch_device_fn.device(inp.device):
        count_valid_kernel[(M,)](inp, valid_count, M, N, block_n, inp.is_cuda)
    kth_inp = torch.where(
        torch.isnan(inp),
        torch.tensor(float("inf"), dtype=inp.dtype, device=inp.device),
        inp,
    )
    min_count = int(torch.min(valid_count).item())
    max_count = int(torch.max(valid_count).item())
    if min_count == max_count:
        if max_count == 0:
            return _full_nan_result((M,), inp.dtype, inp.device)
        values, indices = _native_kthvalue(kth_inp, (max_count + 1) // 2, dim=1)
        return NanMedian(values=values, indices=indices)

    if max_count - min_count <= 1:
        min_k = (min_count + 1) // 2 if min_count > 0 else 0
        max_k = (max_count + 1) // 2
        if min_k == max_k:
            values, indices = _native_kthvalue(kth_inp, max_k, dim=1)
            if min_count > 0:
                return NanMedian(values=values, indices=indices)
            fallback = _full_nan_result((M,), inp.dtype, inp.device)
            positive = valid_count > 0
            return NanMedian(
                values=torch.where(positive, values, fallback.values),
                indices=torch.where(positive, indices, fallback.indices),
            )

        result = _full_nan_result((M,), inp.dtype, inp.device)
        if min_count > 0:
            values, indices = _native_kthvalue(kth_inp, min_k, dim=1)
            mask = valid_count == min_count
            result = NanMedian(
                values=torch.where(mask, values, result.values),
                indices=torch.where(mask, indices, result.indices),
            )

        values, indices = _native_kthvalue(kth_inp, max_k, dim=1)
        mask = valid_count == max_count
        return NanMedian(
            values=torch.where(mask, values, result.values),
            indices=torch.where(mask, indices, result.indices),
        )

    result = _full_nan_result((M,), inp.dtype, inp.device)
    for count in torch.unique(valid_count).tolist():
        count = int(count)
        if count == 0:
            continue
        row_indices = torch.nonzero(valid_count == count).flatten()
        rows = torch.index_select(kth_inp, 0, row_indices)
        values, indices = _native_kthvalue(rows, (count + 1) // 2, dim=1)
        result.values[row_indices] = values
        result.indices[row_indices] = indices
    return result


def _uses_kthvalue_fallback(inp, dim):
    if inp.ndim == 0:
        return False
    dim = _normalize_dim(dim, inp.ndim)
    shape = list(inp.shape)
    reduction_size = shape[dim]
    output_size = math.prod(shape[:dim] + shape[dim + 1 :])
    return (
        reduction_size > 0
        and output_size > 0
        and (reduction_size > MAX_BLOCK_N or inp.dtype is torch.float64)
    )


def _nanmedian_dim_impl(inp, dim, keepdim, out=None):
    if not _uses_kthvalue_fallback(inp, dim):
        return _generic_nanmedian_dim_impl(inp, dim, keepdim, out=out)

    dim = _normalize_dim(dim, inp.ndim)
    shape = list(inp.shape)
    N = shape[dim]
    out_shape = shape[:dim] + shape[dim + 1 :]
    M = math.prod(out_shape)
    keepdim_shape = shape.copy()
    keepdim_shape[dim] = 1
    output_shape = keepdim_shape if keepdim else out_shape
    compute_shape = output_shape if out is not None else keepdim_shape

    result = _nanmedian_kthvalue_fallback(dim_compress(inp, dim), M, N)
    computed_values = result.values.reshape(compute_shape)
    computed_indices = result.indices.reshape(compute_shape)
    if out is None:
        values = computed_values
        indices = computed_indices
    else:
        values, indices = out
        values.copy_(computed_values)
        indices.copy_(computed_indices)

    if out is None and not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)
    return NanMedian(values=values, indices=indices)


def _nanmedian_flat_impl(inp, out=None):
    if inp.numel() == 0:
        return _generic_nanmedian_flat_impl(inp, out=out)

    flat = inp.reshape(-1)
    if out is None:
        return _nanmedian_dim_impl(flat, 0, False).values

    indices = torch.empty((), dtype=torch.long, device=inp.device)
    _nanmedian_dim_impl(flat, 0, False, out=(out, indices))
    return out


def nanmedian(inp):
    logger.debug("GEMS_METAX NANMEDIAN")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp)


def nanmedian_out(inp, *, out):
    logger.debug("GEMS_METAX NANMEDIAN OUT")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp, out=out)


def nanmedian_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_METAX NANMEDIAN DIM")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim)


def nanmedian_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_METAX NANMEDIAN DIM VALUES")
    return _nanmedian_dim_impl(
        inp,
        dim,
        keepdim,
        out=(values, indices),
    )
