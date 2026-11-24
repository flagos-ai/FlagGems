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
"""Clear and resize a sparse COO tensor in place (GCU400 host-side override).

The shared implementation is already a pure metadata/storage swap (no Triton
kernel involved: the operator never touches elements), so the only lever left
is host dispatch cost.  The benchmark harness times the whole call, and on this
backend each ``torch.empty`` costs ~7.8us of dispatcher time while
``Tensor.new_empty`` -- same allocator, but no dtype/device re-validation from
Python -- costs ~5.4us.  Both buffers are therefore created through the source
tensors' own ``new_empty``.

Semantics are unchanged: a non-empty tensor gets *fresh* empty index/value
buffers (aliases taken before the call keep the old entries), an already-empty
tensor only gets its metadata updated, and the dispatcher still bumps the
version counter once.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def sparse_resize_and_clear_(
    self: torch.Tensor, size, sparse_dim: int, dense_dim: int
) -> torch.Tensor:
    """Drop all stored entries of a sparse COO tensor and resize it in place."""
    logger.debug("GEMS_ENFLAME SPARSE_RESIZE_AND_CLEAR_")
    if self.layout != torch.sparse_coo:
        raise NotImplementedError(
            "sparse_resize_and_clear_ supports sparse COO tensors only"
        )
    # The dispatcher rejects in-place ops on a leaf requiring grad before any
    # backend kernel runs; keep that contract for direct (undispatched) calls.
    if torch.is_grad_enabled() and self.is_leaf and self.requires_grad:
        raise RuntimeError(
            "a leaf Variable that requires grad is being used in an in-place operation"
        )

    size = tuple(size)
    if len(size) != sparse_dim + dense_dim:
        raise RuntimeError(
            f"number of dimensions must be sparse_dim ({sparse_dim}) + "
            f"dense_dim ({dense_dim}), but got {len(size)}"
        )

    if self._nnz() == 0:
        # Metadata-only resize: the native empty-tensor path keeps the existing
        # (empty) index/value buffers, exactly like the reference.
        self.sparse_resize_(size, sparse_dim, dense_dim)
        return self

    # Non-empty: install fresh empty buffers of the exact reference shapes -
    # indices (sparse_dim, 0) in int64, values (0, *dense size) in self.dtype,
    # same device - while preserving this tensor object.
    empty_indices = self._indices().new_empty((sparse_dim, 0))
    empty_values = self._values().new_empty((0, *size[sparse_dim:]))
    # check_invariants=False skips the sortedness/range validation (the buffers
    # are empty by construction) and no device/dtype kwargs are passed so the
    # constructor infers both from the two buffers instead of resolving and
    # comparing them against a Python-side device object.
    self.data = torch.sparse_coo_tensor(
        empty_indices, empty_values, size, check_invariants=False
    )
    # No explicit version-counter bump: the dispatcher's ADInplaceOrView key
    # already bumps it once for the aten call, exactly like the native kernel.
    return self
