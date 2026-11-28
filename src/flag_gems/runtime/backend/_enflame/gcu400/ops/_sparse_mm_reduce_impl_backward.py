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

from flag_gems.ops._sparse_mm_reduce_impl_backward import (
    _sparse_mm_reduce_impl_backward as _generic_sparse_mm_reduce_impl_backward,
)

logger = logging.getLogger(__name__)


def _to_int32_indices(sparse):
    """Re-key a 2-D CSR tensor to int32 indices, keeping the values buffer.

    torch_gcu stores an int64 tensor with a true 8-byte element stride only when
    TORCH_GCU_ENABLE_INT64_AND_UINT64 selects the int64 layout; with the default
    Long->Int replacement the same tensor holds 4-byte elements.  A Triton load
    of ``crow_indices``/``col_indices`` into an int64 value therefore either
    reads interleaved halves or trips a device SIP assert (measured on GCU400
    with the env unset: the generic implementation aborts the process, while the
    same tensors re-keyed to int32 run to completion).  The index arithmetic
    itself is layout-only, so the width change is free and the values buffer is
    shared, not copied.
    """
    crow = sparse.crow_indices()
    col = sparse.col_indices()
    if crow.dtype == torch.int32 and col.dtype == torch.int32:
        return sparse
    return torch.sparse_csr_tensor(
        crow.to(torch.int32),
        col.to(torch.int32),
        sparse.values(),
        size=sparse.shape,
        dtype=sparse.dtype,
        device=sparse.device,
    )


def _sparse_mm_reduce_impl_backward(
    self, grad_out, weight, reduce="sum", arg_out=None, output_mask=None
):
    """GCU400 backend of ``aten::_sparse_mm_reduce_impl_backward``.

    The reduction kernels are index-agnostic, so this only normalises the CSR
    index width for the GCU400 int64 layout regime (see
    ``_to_int32_indices``) and defers every argument check, shape rule and
    kernel launch to the generic implementation.  Non-CSR inputs keep the
    generic error paths.
    """
    logger.debug("GEMS_ENFLAME _SPARSE_MM_REDUCE_IMPL_BACKWARD")

    if self.layout == torch.sparse_csr and self.dim() == 2 and self.dense_dim() == 0:
        self = _to_int32_indices(self)
    return _generic_sparse_mm_reduce_impl_backward(
        self, grad_out, weight, reduce, arg_out, output_mask
    )
