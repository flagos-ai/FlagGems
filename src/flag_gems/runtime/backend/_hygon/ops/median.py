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

from flag_gems.ops.median import (
    _DIRECT_REDUCTION_LIMIT,
    MedianResult,
    _anonymous,
    _canonical_dim,
    _copy_out,
    _has_names,
    _median_direct_dim,
    _name_to_dim,
    _use_float_key_select,
)
from flag_gems.ops.median import median_dim as _general_median_dim

logger = logging.getLogger(__name__)

# Floating dtypes for which the direct-reduction sort kernel is both correct
# and cheaper than the binary-search key-select path when the reduction width
# is small.
_SMALL_FLOAT_DIRECT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def median_dim(inp, dim=0, keepdim=False):
    logger.debug("GEMS_HYGON MEDIAN.DIM")

    # On Hygon the general layer routes small-width float reductions through the
    # per-element binary-search key-select kernel (32/64 static iterations),
    # which is markedly slower than the single-pass direct sort kernel for these
    # sizes. For strided (non-last-dim) reductions with a small reduction width,
    # dispatch to `_median_direct_dim` instead; it is numerically identical
    # (including NaN handling) and ~2.5x faster on the affected shapes. All other
    # cases fall through to the proven general implementation.
    if isinstance(dim, str):
        canonical = _name_to_dim(inp, dim)
    else:
        canonical = _canonical_dim(inp.ndim, dim)

    work = _anonymous(inp)
    if (
        not _has_names(inp)
        and work.ndim > 0
        and work.numel() > 0
        and canonical != work.ndim - 1
        and work.dtype in _SMALL_FLOAT_DIRECT_DTYPES
        and work.shape[canonical] <= _DIRECT_REDUCTION_LIMIT
        and _use_float_key_select(work.dtype, work.shape[canonical])
    ):
        output_shape = list(work.shape)
        if keepdim:
            output_shape[canonical] = 1
        else:
            del output_shape[canonical]
        values, indices = _median_direct_dim(work.contiguous(), canonical, output_shape)
        return MedianResult(values=values, indices=indices)

    return _general_median_dim(inp, dim=dim, keepdim=keepdim)


def median_dim_values(inp, dim=0, keepdim=False, *, values, indices):
    logger.debug("GEMS_HYGON MEDIAN.DIM_VALUES")
    result = median_dim(inp, dim=dim, keepdim=keepdim)
    _copy_out(result.values, values, "values")
    _copy_out(result.indices, indices, "indices")
    return MedianResult(values=values, indices=indices)
