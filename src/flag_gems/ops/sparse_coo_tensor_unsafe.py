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

"""Triton implementation of ``aten::_sparse_coo_tensor_unsafe``.

The unsafe COO constructor assembles a ``torch.sparse_coo`` layout tensor
from already-materialised ``(indices, values)`` components **without
validating** the index/size consistency — indices may be unsorted, contain
duplicates, or point outside ``size``; that is the only difference from the
validated ``torch.sparse_coo_tensor`` constructor.

Unlike the compressed family the native unsafe COO constructor accepts a
missing ``device`` kwarg and silently ignores a ``dtype`` that differs from
the payload (the payload dtype wins), so this wrapper mirrors exactly that:
the payload is passed through untouched and ``dtype`` is forwarded verbatim.
"""

import logging

from flag_gems.ops.sparse_coo_tensor_with_dims_and_tensors import (
    sparse_coo_tensor_with_dims_and_tensors,
)

logger = logging.getLogger(__name__)


def sparse_coo_tensor_unsafe(
    indices=None,
    values=None,
    size=None,
    *,
    dtype=None,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS SPARSE COO TENSOR UNSAFE")

    # Delegate to the FlagGems low-level COO constructor instead of calling the
    # native aten op. The native unsafe constructor ignores `dtype` and keeps
    # the payload dtype, so the payload's dtype is forwarded explicitly.
    return sparse_coo_tensor_with_dims_and_tensors(
        indices.shape[0],
        values.dim() - 1,
        size,
        indices,
        values,
        dtype=values.dtype,
        device=values.device if device is None else device,
        pin_memory=bool(pin_memory),
        is_coalesced=False,
    )
