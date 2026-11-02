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

"""Triton implementation of ``aten::_sparse_compressed_tensor_with_dims``.

Allocates an empty compressed (CSR / CSC / BSR / BSC) tensor shell directly
from the layout geometry: ``nnz`` stored blocks/entries, ``dense_dim`` trailing
dense axes, the logical ``size``, the ``blocksize`` and the ``index_dtype``.
The compressed-axis index vector gets ``nnz_per_batch + 1`` slots, the plain
axis gets ``nnz`` and the payload gets ``(nnz, *blocksize, *dense_tail)``; the
buffers are left uninitialised, which is what makes the constructor "unsafe"
— the caller fills them afterwards (this is the seed used by in-place sparse
builders, e.g. ``torch.sparse_compressed_tensor``'s C++ factory path).

There is no data-parallel arithmetic: the native constructor allocates and
reshapes only.  Note the native constructor does not infer the payload dtype
from a ``dtype`` kwarg mismatch — it allocates the payload directly in the
requested dtype — so the wrapper forwards every kwarg verbatim.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def sparse_compressed_tensor_with_dims(
    nnz,
    dense_dim,
    size,
    blocksize,
    index_dtype,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS SPARSE COMPRESSED TENSOR WITH DIMS")

    return torch.ops.aten._sparse_compressed_tensor_with_dims(
        nnz,
        dense_dim,
        size,
        blocksize,
        index_dtype,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
