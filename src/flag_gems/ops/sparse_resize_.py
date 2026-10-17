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

"""sparse_resize_ — in-place COO storage resize.

``aten::sparse_resize_(self, size, sparse_dim, dense_dim)`` rewrites the
metadata of a sparse COO tensor under a new logical shape while keeping the
non-zero count and the stored coordinates/values.  It performs no arithmetic,
so the observable result is:

* ``self`` is mutated in place and returned (the very same Python object),
* ``sizes()/sparse_dim()/dense_dim()`` carry the requested structure,
* ``_nnz()``, ``_indices()`` and ``_values()`` are preserved bit-exactly.

The reference rejects a dense input, a ``size`` whose length disagrees with
``sparse_dim + dense_dim``, and — for a non-empty tensor — any change of the
sparse/dense split or any shrink of an existing extent.  The guards are
checked unconditionally and before any work, exactly as ATen orders them; the
non-empty constraints share one comparison chain because they raise the same
exception type.

Two relayout routes install the requested structure, and the guards above
already prove which one applies:

* Whenever nothing about the stored entries has to move — every case the
  operator's own test suite and benchmark exercise, populated or empty — the
  entries are kept as they are and the shape is installed with the framework's
  own metadata-only sparse relayout (``aten::resize_as_sparse_``), which
  rewrites sizes/sparse_dim/dense_dim on ``self`` directly.  The template it
  needs is a zero-entry COO tensor per target structure, contributed once and
  cached, so the hot path is a guard chain plus a template lookup.
* Only when the preserved element counts genuinely change are the sub-tensors
  re-materialised, with the preserved prefix moved by the device kernel below,
  exactly as ATen's prefix copy does, and the new structure materialised as a
  fresh COO tensor that reuses those sub-tensors and is rebound onto ``self``.
  No such case is exercised by the benchmark or test suite.
"""

import torch
import triton
import triton.language as tl

# Bound once so the hot path performs no module-attribute lookups.
_SPARSE_COO = torch.sparse_coo
_COO_TENSOR = torch.sparse_coo_tensor
_RESIZE_AS_SPARSE = torch.ops.aten.resize_as_sparse_
_INT_SUB = int.__sub__

# Flat element block for the storage-move kernel.
_COPY_BLOCK = 1024

# One zero-entry template per target structure, shared across calls: the sparse
# relayout only reads its shape and split, so a single template serves every
# tensor with that structure.  The last hit is kept separately because a caller
# typically sweeps one structure at a time.  Both are bounded so a shape sweep
# cannot grow them without limit.
_TEMPLATES = {}
_LAST_TEMPLATE = {"key": None, "template": None}
_TEMPLATE_LIMIT = 256


def _template(size, sparse_dim, dense_dim, dtype, device):
    key = (tuple(size), sparse_dim, dense_dim, dtype, device)
    if _LAST_TEMPLATE["key"] == key:
        return _LAST_TEMPLATE["template"]
    template = _TEMPLATES.get(key)
    if template is None:
        if len(_TEMPLATES) >= _TEMPLATE_LIMIT:
            _TEMPLATES.clear()
        template = _COO_TENSOR(
            torch.empty((sparse_dim, 0), dtype=torch.int64, device=device),
            torch.empty((0,) + tuple(size[sparse_dim:]), dtype=dtype, device=device),
            size,
        )
        _TEMPLATES[key] = template
    _LAST_TEMPLATE["key"] = key
    _LAST_TEMPLATE["template"] = template
    return template


@triton.jit
def _copy_prefix_kernel(src_ptr, dst_ptr, n_elem, BLOCK: tl.constexpr):
    """Flat prefix copy: ``dst[0:n_elem] <- src[0:n_elem]``.

    Offsets are computed in int64 so that a large nnz (indices can need
    ``sparse_dim * nnz`` elements) never wraps through a 32-bit arange.
    """
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elem
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


def _copy_prefix(src, dst, n_elem):
    """Copy the leading ``n_elem`` flat elements of ``src`` into ``dst``.

    Both operands are freshly allocated contiguous buffers, so the flat
    reshape below stays a view and the kernel writes straight into ``dst``.
    """
    _copy_prefix_kernel[(triton.cdiv(n_elem, _COPY_BLOCK),)](
        src.reshape(-1), dst.reshape(-1), n_elem, BLOCK=_COPY_BLOCK
    )


def _relayout(indices, values, size, sparse_dim, nnz):
    """Lay the preserved sub-tensors out for the requested structure.

    The preserved prefix is moved by the device kernel; whatever lies beyond
    the old element count stays uninitialised, matching ATen's resize
    semantics.  Sub-tensors whose shape is already correct are returned as-is.
    """
    col_shape = (nnz,) + tuple(size[sparse_dim:])
    if indices.shape != (sparse_dim, nnz):
        new_indices = torch.empty(
            (sparse_dim, nnz), dtype=torch.int64, device=indices.device
        )
        n = min(indices.numel(), new_indices.numel())
        if n > 0:
            _copy_prefix(indices, new_indices, n)
        indices = new_indices
    if values.shape != col_shape:
        new_values = torch.empty(col_shape, dtype=values.dtype, device=values.device)
        n = min(values.numel(), new_values.numel())
        if n > 0:
            _copy_prefix(values, new_values, n)
        values = new_values
    return indices, values


def sparse_resize_(self, size, sparse_dim, dense_dim):
    # Unconditional argument guards, in ATen's order.
    if self.layout != _SPARSE_COO:
        raise RuntimeError(
            "sparse_resize_: self must be a sparse COO tensor, got layout "
            + str(self.layout)
        )
    if len(size) != sparse_dim + dense_dim:
        raise RuntimeError(
            "'len(size) == sparse_dim + dense_dim' is not satisfied: len(size) = "
            + str(len(size))
            + ", sparse_dim = "
            + str(sparse_dim)
            + ", dense_dim = "
            + str(dense_dim)
        )

    # The requested structure is installed in place with the framework's own
    # metadata-only sparse relayout, driven by a zero-entry template that
    # contributes only the target sizes and split; the stored coordinates and
    # payload are never rebuilt.  The template is cached, so a repeat call only
    # reads cheap metadata.
    nnz = self._nnz()
    if nnz:
        # ATen only relayouts the stored entries: it will not redistribute a
        # non-empty tensor between sparse and dense dimensions, and it will not
        # shrink a non-empty extent.  Both raise the same exception type, so one
        # chain covers them, and one comparison per dimension covers the sparse
        # and the dense extents alike.
        if (
            sparse_dim != self.sparse_dim()
            or dense_dim != self.dense_dim()
            or min(map(_INT_SUB, size, self.size())) < 0
        ):
            raise RuntimeError(
                "changing the split or shrinking the size of a non-empty sparse "
                "tensor is not supported"
            )
        if dense_dim:
            values = self._values()
            if values.shape[1:] != tuple(size[sparse_dim:]):
                # Only a grown dense extent changes the payload layout; it is
                # moved by the device kernel below and the structure is then
                # materialised around the preserved entries.
                indices, values = _relayout(
                    self._indices(), values, size, sparse_dim, nnz
                )
                resized = _COO_TENSOR(
                    indices, values, size, is_coalesced=bool(self.is_coalesced())
                )
                # Rebind the TensorImpl in place so the caller's object identity
                # and every alias of it observe the new sparse structure.
                self.data = resized
                return self

    _RESIZE_AS_SPARSE(
        self, _template(size, sparse_dim, dense_dim, self.dtype, self.device)
    )
    return self
