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

# aten::_has_compatible_shallow_copy_type(Tensor self, Tensor from) -> bool
#
# This is a metadata-only operator: it reports whether the TensorImpl of
# ``self`` can shallow-copy the TensorImpl type of ``from``. It performs no
# element-wise computation, so there is no Triton kernel involved. The result
# depends solely on the tensor layout category and is independent of device,
# dtype or shape:
#   * dense (strided) tensors
#   * sparse COO tensors
#   * sparse compressed tensors (CSR / CSC / BSR / BSC)
# Two tensors are shallow-copy compatible iff they fall into the same layout
# category.

# Layouts that share the SparseCsrTensorImpl backing store.
_SPARSE_COMPRESSED_LAYOUTS = (
    torch.sparse_csr,
    torch.sparse_csc,
    torch.sparse_bsr,
    torch.sparse_bsc,
)


def _layout_group(t: torch.Tensor) -> int:
    layout = t.layout
    if layout == torch.sparse_coo:
        return 1
    if layout in _SPARSE_COMPRESSED_LAYOUTS:
        return 2
    # torch.strided (dense) and any other dense-backed layout.
    return 0


def _has_compatible_shallow_copy_type(self: torch.Tensor, from_: torch.Tensor) -> bool:
    """Return True if ``self`` can shallow-copy the TensorImpl type of ``from_``.

    Args:
        self: The destination tensor.
        from_: The source tensor whose TensorImpl type is checked.

    Returns:
        bool: True when both tensors share the same TensorImpl layout category.
    """
    logger.debug("GEMS _HAS_COMPATIBLE_SHALLOW_COPY_TYPE")
    return _layout_group(self) == _layout_group(from_)
