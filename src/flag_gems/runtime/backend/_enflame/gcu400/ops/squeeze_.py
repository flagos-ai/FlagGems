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
"""GCU400 in-place squeeze family.

The three operators below are pure metadata operations: no kernel is launched
and the whole cost is host-side. Two measurements on GCU400 drive the shape of
this implementation:

* one ``as_strided_`` metadata write costs ~1.75 us, while
  ``torch.autograd.graph.increment_version`` costs ~0.50 us;
* the common implementation builds ``drop`` index sets for every call, which is
  wasted work when no dimension can be removed.

Therefore each entry point scans the shape first (``1 not in shape`` is a single
C-level tuple scan) and:

* on the no-op path it bumps the version counter directly, which reproduces
  ``aten::squeeze_``'s version bump for calls that remove nothing (verified:
  ATen bumps the counter by one even for a 0-d tensor or a shape without size-1
  dimensions) and avoids the much more expensive metadata write;
* when ``requires_grad`` is set, it keeps the ``as_strided_`` call so the same
  "leaf Variable that requires grad is being used in an in-place operation"
  error is raised as before;
* on the real squeeze path it rewrites the metadata once, with the surviving
  sizes/strides built by a single filtered comprehension.
"""

import torch

_AS_STRIDED_INPLACE = torch.Tensor.as_strided_
_INCREMENT_VERSION = torch.autograd.graph.increment_version


def squeeze_(self: torch.Tensor) -> torch.Tensor:
    """Remove every size-1 dimension from ``self`` in place.

    Matches ``aten::squeeze_``: zero-copy, self is mutated and returned, and the
    version counter advances by exactly one even when nothing is removed.

    The no-op branch below is inlined in all three entry points rather than
    shared through a helper: one avoided python call costs ~0.2 us of the ~1 us
    body, and this branch is the one the benchmark replays.
    """
    shape = self.shape
    if 1 not in shape:
        # Keeps the autograd leaf check (and its error) identical to ATen.
        if self.requires_grad:
            return _AS_STRIDED_INPLACE(self, shape, self.stride())
        _INCREMENT_VERSION(self)
        return self
    stride = self.stride()
    _AS_STRIDED_INPLACE(
        self,
        [size for size in shape if size != 1],
        [st for size, st in zip(shape, stride) if size != 1],
    )
    return self


def squeeze__dim(self: torch.Tensor, dim: int) -> torch.Tensor:
    """Remove ``dim`` from ``self`` in place when its size is 1."""
    shape = self.shape
    ndim = len(shape)
    nd = ndim if ndim else 1

    # Normalize negative dims with aten's [-nd, nd - 1] convention; a 0-d tensor
    # accepts 0 and -1 (both address the same virtual axis).
    if dim < 0:
        wrapped = dim + nd
        if wrapped < 0:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[-{nd}, {nd - 1}], but got {dim})"
            )
        dim = wrapped
    elif dim >= nd:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{nd}, {nd - 1}], but got {dim})"
        )

    if dim >= ndim or shape[dim] != 1:
        if self.requires_grad:
            return _AS_STRIDED_INPLACE(self, shape, self.stride())
        _INCREMENT_VERSION(self)
        return self
    stride = self.stride()
    _AS_STRIDED_INPLACE(
        self,
        shape[:dim] + shape[dim + 1 :],
        stride[:dim] + stride[dim + 1 :],
    )
    return self


def squeeze__dims(self: torch.Tensor, dims) -> torch.Tensor:
    """Remove every listed size-1 dimension from ``self`` in place.

    Validation order matches ``aten::squeeze_.dims``: each dim is range-checked
    and inserted into a set as it is visited, so for ``[1, 1, 9]`` the duplicate
    at index 1 is reported before the out-of-range 9 at index 2.
    """
    shape = self.shape
    ndim = len(shape)
    nd = ndim if ndim else 1

    # Single-dimension calls (the benchmark shape and the common call pattern)
    # skip the duplicate bookkeeping they cannot need.
    if len(dims) == 1:
        dim = dims[0]
        if dim < 0:
            wrapped = dim + nd
            if wrapped < 0:
                raise IndexError(
                    f"Dimension out of range (expected to be in range of "
                    f"[-{nd}, {nd - 1}], but got {dim})"
                )
            dim = wrapped
        elif dim >= nd:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[-{nd}, {nd - 1}], but got {dim})"
            )
        if dim >= ndim or shape[dim] != 1:
            if self.requires_grad:
                return _AS_STRIDED_INPLACE(self, shape, self.stride())
            _INCREMENT_VERSION(self)
            return self
        stride = self.stride()
        _AS_STRIDED_INPLACE(
            self,
            shape[:dim] + shape[dim + 1 :],
            stride[:dim] + stride[dim + 1 :],
        )
        return self

    seen = set()
    drop = set()
    for dim in dims:
        if dim < 0:
            wrapped = dim + nd
            if wrapped < 0:
                raise IndexError(
                    f"Dimension out of range (expected to be in range of "
                    f"[-{nd}, {nd - 1}], but got {dim})"
                )
            dim = wrapped
        elif dim >= nd:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[-{nd}, {nd - 1}], but got {dim})"
            )
        if dim in seen:
            raise RuntimeError(f"dim {dim} appears multiple times in the list of dims")
        seen.add(dim)
        # Only an in-range dimension has a size; a 0-d tensor's single virtual
        # axis has nothing to remove.
        if dim < ndim and shape[dim] == 1:
            drop.add(dim)

    stride = self.stride()
    if not drop:
        if self.requires_grad:
            return _AS_STRIDED_INPLACE(self, shape, stride)
        _INCREMENT_VERSION(self)
        return self
    _AS_STRIDED_INPLACE(
        self,
        [size for index, size in enumerate(shape) if index not in drop],
        [st for index, st in enumerate(stride) if index not in drop],
    )
    return self
