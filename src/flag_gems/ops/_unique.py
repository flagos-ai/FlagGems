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

from flag_gems.ops.unique import _unique2

logger = logging.getLogger(__name__)


def _unique(inp: torch.Tensor, sorted: bool = True, return_inverse: bool = False):
    """Returns the unique elements of the input tensor.

    Args:
        inp: Input tensor
        sorted: Whether to sort the unique elements in ascending order
        return_inverse: Whether to return the indices for reconstructing the original tensor

    Returns:
        A tuple of (unique, inverse_indices) where:
        - unique: The unique elements tensor
        - inverse_indices: If return_inverse=True, indices to reconstruct original tensor.
                          If return_inverse=False, an empty tensor.
    """
    logger.debug("GEMS _UNIQUE")
    unique, inverse_indices, _ = _unique2(
        inp, sorted=sorted, return_inverse=return_inverse, return_counts=False
    )
    if inverse_indices is None:
        inverse_indices = torch.empty(0, dtype=torch.int64, device=inp.device)
    return unique, inverse_indices
