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

from flag_gems.ops.as_strided_copy import _launch_as_strided_copy

from ..utils.tle_copy import tle_copy
from .expand_copy import _launch_bcast

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# The flat gather kernel behind `_launch_bcast` addresses 6 dimensions; a
# deeper layout takes the rank-general strided copy instead.
_MAX_FLAT_DIM = 6


def _tle_tile_copy(src, dst):
    """The flat, contiguous, same-dtype transfer, when tle can serve it.

    `tle_copy` only reaches its `tle.gpu` tile path for that layout. Its strided,
    transposed and dtype-converting paths are built on `tle.dsa`, which this
    triton does not ship, and reaching them raises `AttributeError` instead of
    the documented `return False` -- the caller's fallback would never run. So
    the transfer is only offered to tle when its own tile-path precondition
    holds (same dtype, both sides contiguous); alignment and dtype support stay
    tle's call, and a `False` from it means the fallback takes over.
    """
    if src.dtype != dst.dtype or not src.is_contiguous() or not dst.is_contiguous():
        return False
    return tle_copy(src, dst)


def lift_out(A, *, out=None):
    """Implements aten::lift.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!).

    Copies ``A`` into ``out`` and returns ``out``. A lift is a pure move, so it
    rides the copy-family recipe (same as ``alias_copy`` / ``copy_``): tle takes
    the whole transfer for the layout it can express, and gems' own Triton
    kernels do the rest -- a real-stride move through ``A``'s and ``out``'s own
    strides, at any alignment.  ``aten::_copy_from`` is not used (gems
    operators may not delegate their work to it).
    """
    logger.debug("GEMS_KUNLUNXIN LIFT_OUT")
    if out is None:
        out = torch.empty_like(A, memory_format=torch.contiguous_format)
    # `out=` contract, matching the sibling `alias_copy_out`: refuse a mismatched
    # destination loudly instead of writing a flat run into it. ATen's
    # `_resize_output` would resize `out` instead, but the gems `resize_`
    # override resets `storage_offset`, so this backend refuses rather than
    # resizes; a flat write into an equal-element-count destination of a
    # different shape used to be accepted and silently returned `out` under its
    # own shape.
    if A.dtype != out.dtype:
        raise RuntimeError("lift_out: dtype of input and output must match.")
    if list(A.shape) != list(out.shape):
        raise RuntimeError(
            "lift_out: input and output must have the same shape, but got "
            f"{tuple(A.shape)} and {tuple(out.shape)}."
        )
    if A.device != out.device:
        raise RuntimeError("lift_out: input and output must be on the same device.")
    if out.numel() == 0:
        return out
    if _tle_tile_copy(A, out):
        return out
    # Strided fallback.  A contiguous destination keeps the flat, block-tiled
    # gather (`_launch_bcast`, the kernel behind `expand_copy`), which decodes
    # the flat index against `A`'s real strides -- a transposed / stepped /
    # narrowed source therefore needs no layout normalisation.  Otherwise the
    # destination is written through its own strides by the gems strided copy.
    if out.is_contiguous() and A.dim() <= _MAX_FLAT_DIM:
        _launch_bcast(A.shape, A.stride(), A, out, out.numel())
    else:
        _launch_as_strided_copy(A, out)
    return out
