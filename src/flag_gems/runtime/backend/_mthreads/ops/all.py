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

"""MTHREADS boolean AND-reductions (all / all_dim / all_dims).

Thin wrapper over the shared ``_boolean_reduction`` core (single codegen,
``IS_ALL=True`` polarity); see that module for the execution
architecture.
"""

import logging
import operator
import os

import torch

from flag_gems.runtime.backend._mthreads.ops import _boolean_reduction as _boolean

logger = logging.getLogger(__name__)

# Split-K merge strategy for the fused reduction kernel (ablation knob):
#   "partials" - int8 partials workspace + AND-merge kernel (default)
#   "atomic"   - int32 zero-found flags merged via atomic_max
#   "none"     - force splits=1 (single kernel, degenerate for M=1)
_strategy = os.environ.get("FLAGGEMS_ALL_SPLIT_STRATEGY", "partials").lower()
if _strategy not in ("partials", "atomic", "none"):
    _strategy = "partials"
_MERGE_STRATEGY = {
    "partials": "partial",
    "atomic": "atomic",
    "none": "none",
}[_strategy]


def _out_dtype(inp: torch.Tensor) -> torch.dtype:
    # ATen keeps uint8 for all/any (legacy bool storage); everything else -> bool.
    return torch.uint8 if inp.dtype == torch.uint8 else torch.bool


def _all_dims_impl(inp, reduce_dims, keepdim):
    """Single-launch fused multi-dim AND over `reduce_dims` (normalized)."""
    return _boolean.boolean_dims_impl(
        inp, reduce_dims, keepdim, is_all=True, merge_strategy=_MERGE_STRATEGY
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def all(inp):
    logger.debug("GEMS_MTHREADS ALL")
    n_elements = inp.numel()

    if n_elements == 0:
        # all() over an empty domain is vacuously True; never launch kernels.
        return torch.ones([], dtype=_out_dtype(inp), device=inp.device)

    if not torch.ops.aten.is_non_overlapping_and_dense(inp):
        # Gapped (slice/pad) or broadcast (stride-0) storage: linear addressing
        # would read the wrong element set. Use the stride-aware fused path.
        return _all_dims_impl(inp, tuple(range(inp.ndim)), False)

    # Dense non-overlapping layout: the addressed offsets are exactly
    # {0 .. n_elements-1}, so the two-kernel linear reduction is valid.
    return _boolean.boolean_whole_impl(inp, is_all=True)


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_MTHREADS ALL_DIM")
    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
        return out

    dim = operator.index(dim)
    ndim = inp.ndim
    if ndim == 0:
        # 0-dim tensors accept dim 0 / -1 (ATen semantics): result = bool(x).
        if dim == 0 or dim == -1:
            return (inp != 0).to(_out_dtype(inp))
        raise IndexError(
            f"Dimension out of range (expected to be in range of [-1, 0], but got {dim})"
        )
    if not (-ndim <= dim < ndim):
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
        )
    return _all_dims_impl(inp, (dim % ndim,), keepdim)


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_MTHREADS ALL_DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)

    dims = [operator.index(d) for d in dim]
    ndim = inp.ndim
    if ndim == 0:
        seen = set()
        for d in dims:
            if d not in (0, -1):
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [-1, 0], but got {d})"
                )
            if 0 in seen:
                raise RuntimeError("dim 0 appears multiple times in the reduction")
            seen.add(0)
        return (inp != 0).to(_out_dtype(inp))

    seen = set()
    for d in dims:
        if not (-ndim <= d < ndim):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})"
            )
        d = d % ndim
        if d in seen:
            raise RuntimeError(f"dim {d} appears multiple times in the reduction")
        seen.add(d)

    if not seen:
        # dim=[] reduces nothing: elementwise bool conversion (CPU-native rule).
        return (inp != 0).to(_out_dtype(inp))

    return _all_dims_impl(inp, tuple(sorted(seen)), keepdim)


__all__ = ["all", "all_dim", "all_dims"]
