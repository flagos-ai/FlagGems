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

"""MTHREADS boolean OR-reductions (any / any_dim / any_dims).

Thin wrapper over the shared ``_boolean_reduction`` core (single codegen,
``IS_ALL=False`` polarity); see that module for the execution
architecture.
"""

import logging

import torch

from flag_gems.runtime.backend._mthreads.ops import _boolean_reduction as _boolean

logger = logging.getLogger(__name__)

# Merge strategy used when the reduction is split across CTAs along K
# (see _boolean_reduction for the strategy catalogue).
_MERGE_STRATEGY = "partial"


def _any_dims_impl(inp, reduce_dims, keepdim):
    """Single-launch fused multi-dim any over `reduce_dims` (normalized)."""
    return _boolean.boolean_dims_impl(
        inp, reduce_dims, keepdim, is_all=False, merge_strategy=_MERGE_STRATEGY
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _check_dim_range(dim, ndim):
    lo = -max(ndim, 1)
    hi = max(ndim, 1) - 1
    if not (lo <= dim <= hi):
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{lo}, {hi}], but got {dim})"
        )
    return dim % max(ndim, 1)


def _bool_like(inp):
    out = inp != 0
    if inp.dtype == torch.uint8:
        return out.to(torch.uint8)
    return out


def any(inp):
    logger.debug("GEMS_MTHREADS ANY")
    n_elements = inp.numel()
    out_dtype = torch.uint8 if inp.dtype == torch.uint8 else torch.bool
    if n_elements == 0:
        # any over an empty tensor is False; do not launch any kernel.
        return torch.zeros([], dtype=out_dtype, device=inp.device)
    if not inp.is_contiguous():
        # Strided inputs cannot use linear addressing; reduce all dims with
        # the fused kernel, which handles arbitrary strides.
        return _any_dims_impl(inp, list(range(inp.ndim)), keepdim=False)

    return _boolean.boolean_whole_impl(inp, is_all=False)


def any_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_MTHREADS ANY_DIM")
    if dim is None:
        out = any(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
        return out
    if not isinstance(dim, int):
        raise TypeError(f"dim(): argument 'dim' must be int, not {type(dim).__name__}")
    ndim = inp.ndim
    dim = _check_dim_range(dim, ndim)
    if ndim == 0:
        # torch native accepts dim 0/-1 on 0-dim tensors: no-op reduction.
        return _bool_like(inp)
    return _any_dims_impl(inp, [dim], keepdim)


def any_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_MTHREADS ANY_DIMS")
    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)
    if isinstance(dim, tuple):
        dim = list(dim)
    if not isinstance(dim, list):
        raise TypeError(
            f"any(): argument 'dim' must be int or list of ints, not {type(dim).__name__}"
        )
    ndim = inp.ndim
    if len(dim) == 0:
        # Matches upstream CPU semantics: convert to bool, no reduction
        # (MUSA native incorrectly performs a full reduction here).
        return _bool_like(inp)
    reduce_dims = []
    for d in dim:
        if not isinstance(d, int):
            raise TypeError(
                f"dim(): argument 'dim' must be int, not {type(d).__name__}"
            )
        d = _check_dim_range(d, ndim)
        if d in reduce_dims:
            raise RuntimeError(f"dim {d} appears multiple times in the list of dims")
        reduce_dims.append(d)
    if ndim == 0:
        return _bool_like(inp)
    return _any_dims_impl(inp, sorted(reduce_dims), keepdim)


__all__ = ["any", "any_dim", "any_dims"]
