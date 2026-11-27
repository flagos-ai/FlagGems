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

logger = logging.getLogger(__name__)

# GCU400 has no SparseCOO TensorImpl, and aten's sparse COO constructor is not registered for
# SparsePrivateUse1, so tests of sparse metadata operators raise NotImplementedError while building inputs.
# This adds a CPU-side constructor: metadata queries do not read storage, so indices/values are copied to CPU
# to build an equivalent COO tensor with sparse_dim / dense_dim / shape / coalesced unchanged.
# Registration is idempotent and reusable by _dimV and similar operators.
_SPARSE_COO_FALLBACK_FLAG = "_flag_gems_gcu400_sparse_coo_fallback_registered"
_SPARSE_COO_LIBRARY = None


def _dimI(self: torch.Tensor) -> int:
    """Return the number of sparse dimensions of a sparse COO tensor ``self``.

    Mirrors ``aten::_dimI(Tensor self) -> int``. This is a pure layout-metadata
    query: the answer is a property of the tensor's index layout, never of the
    stored element data, so there is no data movement and no Triton kernel.
    The result is read from TensorImpl/sparse-COO metadata without
    re-dispatching the ``_dimI`` operator itself (re-entering the dispatcher
    from inside a kernel registered on the Sparse dispatch key would recurse
    into this implementation).

    Semantics by layout (verified against native ATen on H20, torch
    2.9.0+cu128):
    - ``torch.sparse_coo``: ``_dimI`` is the sparse dimension count, i.e. the
      leading dimension of the index matrix, ``self._indices().shape[0]``
      (the native COO TensorImpl keeps the index matrix at shape
      ``(sparse_dim, nnz)``, so this equals ``self.sparse_dim()``). Plain 2-D
      COO gives 2, fully sparse rank-N COO gives N, and hybrid COO tensors
      (values carrying trailing dense dimensions) give the number of sparse
      dims only. Zero-nnz and uncoalesced tensors report the same value as
      their coalesced counterparts, because coalescing state does not change
      the layout.
    - Every other layout: native ATen registers ``aten::_dimI`` only for the
      Sparse (COO) backends -- strided, CSR/CSC (and BSR/BSC) raise
      ``NotImplementedError`` ("Could not run 'aten::_dimI' ... from the
      'CUDA'/'SparseCsrCUDA' backend"). Mirror that contract here instead of
      guessing a formula.
    logger.debug("GEMS_ENFLAME _DIMI")

    Host-path notes (gcu400 hot-body form; do not regress):
    - The gate is ``self.is_sparse`` (ATen defines it as ``layout() ==
      kSparse``), so it accepts exactly the COO layout and every other layout
      keeps raising, identical to a ``self.layout is torch.sparse_coo`` gate;
      it reads the cached bool straight from TensorImpl and is a few ns
      cheaper per call than the bound-layout enum comparison.
    - ``_indices().shape[0]`` is already a plain Python int, so no ``int()``
      wrapper is needed. ``self.sparse_dim()`` is NOT used here: this backend
      registers a SparseCPU Python trampoline for it (~1.6us/call), while
      ``aten::_indices`` has no such trampoline and stays a native read.
    - There is deliberately no per-call ``logger.debug`` trace line: this
      op's whole cost is host-side metadata access (~0.2us) and a disabled
      debug call would be its single largest component (dense_dim/dimV/dim
      precedent on the enflame side).
    """
    if self.is_sparse:
        return self._indices().shape[0]
    raise NotImplementedError(
        f"Could not run 'aten::_dimI' with arguments from the '{self.layout}' "
        "layout; aten::_dimI is only registered for the Sparse (COO) backends"
    )
