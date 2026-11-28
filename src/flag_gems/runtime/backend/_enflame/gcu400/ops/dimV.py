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

# Resolved once at import time (dense_dim.py / _dimI.py precedent): the
# per-call ``torch.sparse_coo`` attribute walk is removable host slack, and
# layout enum members are singletons, so ``is`` is equivalent to ``==``.
_SPARSE_COO = torch.sparse_coo


def _dimV(self: torch.Tensor) -> int:
    """Return the number of dense dimensions of a sparse COO tensor ``self``.

    Mirrors ``aten::_dimV(Tensor self) -> int``. This is a pure layout-metadata
    query: the result never depends on stored element data, so it is computed
    from TensorImpl properties without re-dispatching the ``_dimV`` operator
    itself (re-entering the dispatcher from inside a registered Sparse kernel
    would recurse into this implementation).

    Semantics by layout (verified against native ATen on H20, torch
    2.8.0a0+5228986c39.nv25.05):
    - ``torch.sparse_coo``: ``sparse_dim`` equals ``self._indices().shape[0]``
      (the leading index-matrix dimension), so ``_dimV == dim() -
      indices.shape[0]``. Plain 2-D COO gives 0, hybrid COO tensors (values
      with trailing dense dimensions) give ``dim() - 1``, zero-nnz and
      uncoalesced COO behave identically to coalesced plain COO (metadata is
      unaffected by coalescing state).
    - Every other layout: native ATen registers ``aten::_dimV`` only for the
      Sparse (COO) backends — strided, CSR/CSC (and BSR/BSC) raise
      ``NotImplementedError`` natively ("Could not run 'aten::_dimV' ... from
      the 'CUDA'/'SparseCsrCUDA' backend"). Mirror that contract here instead
      of guessing a formula.
    logger.debug("GEMS_ENFLAME _DIMV")

    Enflame-local hot-path deviation (gcu400 copy): the operator has no
    device kernel (GCU400 has no sparse-COO TensorImpl; benchmark/test_dimV
    inputs are CPU-resident COO built through the shared construction
    fallback), so its whole cost is host-side Python. Interleaved timeit
    (GCU400, torch 2.11, 5 rounds x 30x2000) puts the previous body at
    ~574ns per direct call, dominated by ``self._indices().shape[0]``
    (~470ns: one dispatcher round-trip plus a fresh tensor-object wrap per
    call; ``self.sparse_dim()`` is worse at ~1.58us). The COO value is
    instead read via ``self.dense_dim()`` (~258ns, one dispatcher
    round-trip returning a plain int) — the registered FlagGems sibling
    metadata op that shares the exact COO semantics (FlagGems op
    composition per the host-function contract; no recursion: the dense_dim
    implementation never re-enters ``_dimV``), while this wrapper keeps the
    ``_dimV``-only layout contract (non-COO raises; dense_dim alone would
    answer 0 for compressed layouts). With the layout constant resolved at
    import time and the singleton ``is`` compare, the direct call measures
    ~322ns (-44%). Note that the official KERNEL-mode harness floors at
    ~2.1us for any host-only call (an empty lambda measures the same), so
    this gain is visible only in wall-clock timeit, not in the event-based
    kernel-mode record (dense_dim.py / _unsafe_view.py precedent).
    """
    if self.layout is _SPARSE_COO:
        return self.dense_dim()
    raise NotImplementedError(
        f"Could not run 'aten::_dimV' with arguments from the '{self.layout}' layout; "
        "aten::_dimV is only registered for the Sparse (COO) backends"
    )
