# Copyright 2026, The FlagOS Contributors.
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
#
"""GCU400 ``swapaxes`` (zero-copy view, host-bound).

``swapaxes`` launches no kernel: the entire cost is host-side dispatch plus one
``torch.as_strided`` view construction (~1.8 us measured on GCU400). The common
implementation delegates to ``swapdims``, which is imported eagerly and
therefore resolves to the *common* variant; that path re-validates both dims
through a generic normalizer and always builds two python lists.

This specialization:

* serves already-normalized in-range dims straight from a branch that only
  touches ``shape``/``stride`` (2-D swaps use tuple unpacking, no list round
  trip, because 2-D transpose is by far the most common call);
* keeps the generic normalize/raise path for negative or out-of-range axes,
  including aten's ``[-nd, nd - 1]`` reporting for 0-d and 1-d operands;
* always returns a *fresh* view object, even when both axes are equal, because
  PyTorch's view semantics reject returning ``self``.
"""

import torch

_AS_STRIDED = torch.as_strided


def _swapaxes_generic(self: torch.Tensor, axis0: int, axis1: int) -> torch.Tensor:
    """Normalize negative/out-of-range axes exactly like ATen, then swap."""
    ndim = len(self.shape)
    # A 0-d tensor accepts only 0 and -1 (both denote the sole virtual axis),
    # so the accepted range is clamped to [-1, 0] instead of the rank-based one.
    max_axis = ndim if ndim > 1 else 1
    raw0, raw1 = axis0, axis1
    if axis0 < 0:
        axis0 += max_axis
    if axis1 < 0:
        axis1 += max_axis
    if not (0 <= axis0 < max_axis):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{max_axis}, {max_axis - 1}], but got {raw0})"
        )
    if not (0 <= axis1 < max_axis):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{max_axis}, {max_axis - 1}], but got {raw1})"
        )

    shape = list(self.shape)
    stride = list(self.stride())
    if axis0 != axis1:
        shape[axis0], shape[axis1] = shape[axis1], shape[axis0]
        stride[axis0], stride[axis1] = stride[axis1], stride[axis0]
    return _AS_STRIDED(self, shape, stride)


def swapaxes(self: torch.Tensor, axis0: int, axis1: int) -> torch.Tensor:
    """Return a view of ``self`` with ``axis0`` and ``axis1`` exchanged."""
    shape = self.shape
    ndim = len(shape)
    if 0 <= axis0 < ndim and 0 <= axis1 < ndim:
        stride = self.stride()
        if axis0 == axis1 or ndim < 2:
            # Same axis (including every 1-D call) keeps the layout but must
            # still produce a distinct view object.
            return _AS_STRIDED(self, shape, stride)
        if ndim == 2:
            return _AS_STRIDED(self, (shape[1], shape[0]), (stride[1], stride[0]))
        sizes = list(shape)
        strides = list(stride)
        sizes[axis0], sizes[axis1] = sizes[axis1], sizes[axis0]
        strides[axis0], strides[axis1] = strides[axis1], strides[axis0]
        return _AS_STRIDED(self, sizes, strides)
    return _swapaxes_generic(self, axis0, axis1)
