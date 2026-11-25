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

"""GCU300 specialization of ``diagonal_copy``.

The generic implementation copies the diagonal view with a rank-N
``pointwise_dynamic`` kernel.  That generator emits block pointers whose
``order`` is derived from the view's stride order, and this backend's compiler
rejects the result for a diagonal view:

    error: result shape must match the permutation of the source shape

so the operator cannot compile at all for every offset and dimensionality.

The copy goes through ``Tensor.copy_`` instead, which is stride-aware and
already handled by this backend's ``copy_`` kernel.  The view, the empty-output
early return and the dimension validation are the generic ones.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def diagonal_copy(
    self: torch.Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> torch.Tensor:
    """
    Performs the same operation as torch.diagonal, but returns a copy instead of a view.
    """
    logger.debug("GEMS_ENFLAME DIAGONAL_COPY")

    ndim = self.ndim
    dim1 = dim1 if dim1 >= 0 else dim1 + ndim
    dim2 = dim2 if dim2 >= 0 else dim2 + ndim

    if dim1 == dim2:
        raise ValueError("dim1 and dim2 must be different")

    diag_view = torch.diagonal(self, offset=offset, dim1=dim1, dim2=dim2)

    output = torch.empty_like(diag_view)
    if output.numel() == 0:
        return output

    output.copy_(diag_view)
    return output
