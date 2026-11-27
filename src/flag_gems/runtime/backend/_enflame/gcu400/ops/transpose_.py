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
"""GCU400 ``transpose_`` (in-place metadata swap, host-bound).

``transpose_`` launches no kernel, so its cost is one ``as_strided_`` metadata
write (~1.75 us) plus the host-side validation. The common implementation pays
the validation cost on every call, including the two branches that are by far
the most frequent in practice (2-D transpose and "same dim" no-ops).

This specialization:

* handles already-normalized in-range dims first, so the negative-dim wrap and
  the IndexError machinery only run for the rare invalid call;
* keeps the 2-D tuple fast path of the common version;
* advances the version counter for the same-dim no-op, matching ATen's in-place
  bookkeeping instead of silently returning ``self`` untouched.
"""

import torch

_AS_STRIDED_INPLACE = torch.Tensor.as_strided_
_INCREMENT_VERSION = torch.autograd.graph.increment_version


def _transpose_generic(self: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Normalize negative dims and validate against aten's [-nd, nd - 1]."""
    ndim = len(self.shape)
    # A 0-d tensor accepts dim 0 and -1 (aten wraps against ndim=1).
    wrap_dim = ndim if ndim > 0 else 1
    dim0_wrapped = dim0 + wrap_dim if dim0 < 0 else dim0
    dim1_wrapped = dim1 + wrap_dim if dim1 < 0 else dim1
    if not 0 <= dim0_wrapped < wrap_dim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{wrap_dim}, {wrap_dim - 1}], but got {dim0})"
        )
    if not 0 <= dim1_wrapped < wrap_dim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{wrap_dim}, {wrap_dim - 1}], but got {dim1})"
        )
    if ndim < 2 or dim0_wrapped == dim1_wrapped:
        _INCREMENT_VERSION(self)
        return self

    shape = list(self.shape)
    stride = list(self.stride())
    shape[dim0_wrapped], shape[dim1_wrapped] = (
        shape[dim1_wrapped],
        shape[dim0_wrapped],
    )
    stride[dim0_wrapped], stride[dim1_wrapped] = (
        stride[dim1_wrapped],
        stride[dim0_wrapped],
    )
    _AS_STRIDED_INPLACE(self, shape, stride)
    return self


def transpose_(self: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Swap two dimensions of ``self`` in place and return ``self``."""
    shape = self.shape
    ndim = len(shape)
    if 0 <= dim0 < ndim and 0 <= dim1 < ndim:
        if dim0 == dim1:
            # Swapping an axis with itself keeps the layout; ATen still counts
            # this as an in-place write on the version counter.
            _INCREMENT_VERSION(self)
            return self
        stride = self.stride()
        if ndim == 2:
            _AS_STRIDED_INPLACE(self, (shape[1], shape[0]), (stride[1], stride[0]))
            return self
        sizes = list(shape)
        strides = list(stride)
        sizes[dim0], sizes[dim1] = sizes[dim1], sizes[dim0]
        strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
        _AS_STRIDED_INPLACE(self, sizes, strides)
        return self
    return _transpose_generic(self, dim0, dim1)
