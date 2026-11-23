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

# GCU400's compressed sparse backend (SparseCsr/PrivateUse1) only registers to_sparse_* constructors,
# with no metadata accessors such as col_indices / row_indices / values: calling them on GCU tensors
# raises RuntimeError or reports "Could not run 'aten::_nnz' ... from the
# 'SparseCsrgcu' backend". _nnz is a pure metadata query whose answer depends only on the shape of the index buffers,
# not on stored element data, so follow the COO approach in _dimI.py in the same directory: register
# a CPU-side fallback for compressed sparse constructors on PrivateUse1, move dense inputs to CPU,
# then call the CPU native constructor to obtain a semantically equivalent compressed sparse tensor with complete
# accessors.
# CPU inputs do not hit PrivateUse1, so there is no recursion.
_COMPRESSED_SPARSE_FALLBACK_FLAG = (
    "_flag_gems_gcu400_compressed_sparse_fallback_registered"
)
_COMPRESSED_SPARSE_LIBRARY = None
_COMPRESSED_SPARSE_CONSTRUCTORS = (
    "to_sparse_csr",
    "to_sparse_csc",
    "to_sparse_bsr",
    "to_sparse_bsc",
)


def ensure_compressed_sparse_gcu_fallback():
    """Idempotently register the PrivateUse1 compressed-sparse constructor fallback for sparse metadata operators."""
    global _COMPRESSED_SPARSE_LIBRARY
    if getattr(torch, _COMPRESSED_SPARSE_FALLBACK_FLAG, False):
        return

    def make_fallback(constructor_name):
        def build_cpu_compressed(self, *args, **kwargs):
            cpu_self = self.detach().to("cpu")
            return getattr(cpu_self, constructor_name)(*args, **kwargs)

        return build_cpu_compressed

    library = torch.library.Library("aten", "IMPL")
    for constructor_name in _COMPRESSED_SPARSE_CONSTRUCTORS:
        try:
            library.impl(
                constructor_name, make_fallback(constructor_name), "PrivateUse1"
            )
        except RuntimeError as error:
            logger.warning(
                "compressed sparse gcu400 fallback %s skipped: %s",
                constructor_name,
                error,
            )
    _COMPRESSED_SPARSE_LIBRARY = library
    setattr(torch, _COMPRESSED_SPARSE_FALLBACK_FLAG, True)


ensure_compressed_sparse_gcu_fallback()


def _nnz(self: torch.Tensor) -> int:
    """Return the number of stored entries of a sparse tensor ``self``.

    Mirrors ``aten::_nnz(Tensor self) -> int``. This is a pure layout-metadata
    query: the answer is a property of the index buffers, never of the stored
    element data, so there is no data movement and no Triton kernel. The
    result is read from tensor metadata without re-dispatching the ``_nnz``
    operator itself (re-entering the dispatcher from inside a kernel
    registered on the Sparse/SparseCsr dispatch keys would recurse into this
    implementation).

    Semantics by layout (verified against native ATen on H20, torch
    2.9.0+cu128, and ATen native_functions.yaml, where ``_nnz`` dispatches to
    ``_nnz_sparse`` for the Sparse (COO) backends and ``_nnz_sparse_csr`` for
    the SparseCsr backends):
    - ``torch.sparse_coo``: the stored-entry count,
      ``self._values().shape[0]`` (native COO ``TensorImpl::nnz`` is
      ``values_.size(0)``). This is the raw storage count, not the number of
      unique coordinates: uncoalesced tensors with duplicated indices report
      every stored entry, zero-nnz tensors report 0, and hybrid COO tensors
      (values with trailing dense dimensions) report the leading values
      dimension only. ``_values()`` (unlike ``values()``) does not require a
      coalesced tensor.
    - Compressed layouts: the native compressed ``TensorImpl::nnz`` is
      ``plain_indices_.size(-1)`` -- the trailing dimension of the per-row /
      per-column index buffer, i.e. ``col_indices().shape[-1]`` for CSR/BSR
      and ``row_indices().shape[-1]`` for CSC/BSC. Batched compressed tensors
      keep one index row per batch and report the per-batch entry count
      (e.g. values of shape ``(B, nnz, ...)`` give ``nnz``), and zero-nnz
      tensors report 0.
    - Every other layout: native ATen registers ``aten::_nnz`` only for the
      Sparse (COO) and SparseCsr backends -- strided input raises
      ``NotImplementedError`` ("Could not run 'aten::_nnz' ... from the
      'CUDA' backend"). Mirror that contract here instead of guessing a
      formula.
    """
    logger.debug("GEMS_ENFLAME_GCU400 _NNZ")
    if self.layout == torch.sparse_coo:
        # Native COO TensorImpl::nnz is values_.size(0); _values() (unlike
        # values()) is available on uncoalesced tensors.
        return int(self._values().shape[0])
    if self.layout in (torch.sparse_csr, torch.sparse_bsr):
        # Native compressed TensorImpl::nnz is plain_indices_.size(-1).
        return int(self.col_indices().shape[-1])
    if self.layout in (torch.sparse_csc, torch.sparse_bsc):
        return int(self.row_indices().shape[-1])
    raise NotImplementedError(
        f"Could not run 'aten::_nnz' with arguments from the '{self.layout}' "
        "layout; aten::_nnz is only registered for the Sparse (COO) and "
        "SparseCsr backends"
    )
