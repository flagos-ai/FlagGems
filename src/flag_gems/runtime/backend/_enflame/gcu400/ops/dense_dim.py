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

from flag_gems.runtime import device as runtime_device

logger = logging.getLogger(__name__)


def dense_dim(self: torch.Tensor) -> int:
    """Return the number of dense dimensions of ``self``.

    Mirrors ``aten::dense_dim(Tensor self) -> int``. This is a pure
    layout-metadata query: the result never depends on stored element data,
    so it is computed from TensorImpl properties without re-dispatching the
    ``dense_dim`` operator itself (re-entering the dispatcher from inside a
    registered CUDA kernel would recurse into this implementation).

    Semantics by layout (verified against native ATen on H20):
    - ``torch.strided``: every dimension is dense, ``dense_dim == dim()``.
    - ``torch.sparse_coo``: ``sparse_dim`` equals ``self._indices().shape[0]``
      (the leading index-matrix dimension), so ``dense_dim == dim() -
      indices.shape[0]``. Hybrid COO tensors (values with trailing dense
      dimensions) are covered by the same formula.
    - Sparse compressed layouts (CSR/CSC/BSR/BSC): the two sparse dims are
      fixed by the ATen contract, and any leading batch dimensions are
      independent of the sparse/dense classification, so ``dense_dim == 0``
      for every compressed layout (verified on H20: a batched CSR tensor
      with shape [B, M, N] reports ``dense_dim() == 0`` natively).
    """
    logger.debug("GEMS DENSE_DIM")
    if self.layout == torch.strided:
        return len(self.shape)
    if self.layout == torch.sparse_coo:
        return len(self.shape) - self._indices().shape[0]
    return 0


# ---------------------------------------------------------------------------
# GCU sparse-compressed dispatch gap workaround (values_copy.py precedent)
#
# torch_gcu registers no SparseCsr kernels for its PrivateUse1 backend, so the
# SparseCsrPrivateUse1 slot of aten::dense_dim falls through to the
# CompositeExplicitAutograd shim, which accepts only strided/sparse-COO
# layouts and raises "dense_dim expected sparse or strided tensor layout but
# got SparseCsr" for every GCU CSR/CSC/BSR/BSC tensor. The call dies inside
# torch's dispatcher before any FlagGems code runs, which also breaks test
# references that query ``tensor.dense_dim()`` on device-resident compressed
# tensors. Registering the operator above on that key serves the query from
# TensorImpl metadata, matching the native ATen contract for compressed
# layouts exactly (dense_dim == 0 for plain and batched CSR/CSC/BSR/BSC;
# verified against native CPU). The sparse-COO key is deliberately left to
# the composite: it already answers COO from TensorImpl metadata, and GCU has
# no COO TensorImpl anyway (those tensors stay CPU-resident).
#
# The registration is idempotent across every module name this file is
# imported under (the arch ops tree loads under several aliases at once): the
# library handle is anchored on torch, so exactly one instance performs it.
# ---------------------------------------------------------------------------

_SPARSE_CSR_KEY = "SparseCsr" + runtime_device.dispatch_key
_KEY_STATE_FLAG = "_flag_gems_gcu400_dense_dim_key_state"


def _install_gcu_sparse_csr_dense_dim_kernel():
    if getattr(torch, _KEY_STATE_FLAG, None) is not None:
        return
    library = torch.library.Library("aten", "IMPL")
    setattr(torch, _KEY_STATE_FLAG, library)
    try:
        library.impl("dense_dim", dense_dim, _SPARSE_CSR_KEY)
    except RuntimeError as error:
        logger.warning(
            "dense_dim: sparse key registration dense_dim@%s skipped: %s",
            _SPARSE_CSR_KEY,
            error,
        )


_install_gcu_sparse_csr_dense_dim_kernel()
