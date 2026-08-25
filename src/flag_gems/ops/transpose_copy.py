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

from flag_gems.ops.transpose import transpose

logger = logging.getLogger(__name__)


def transpose_copy(input: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Return a contiguous copy with ``dim0`` and ``dim1`` swapped."""
    logger.debug("GEMS TRANSPOSE_COPY")

    if input.ndim == 0:
        for dim in (dim0, dim1):
            if dim < -1 or dim > 0:
                raise IndexError(
                    "Dimension out of range (expected to be in range of "
                    f"[-1, 0], but got {dim})"
                )
        transposed = input.view(input.shape)
    else:
        transposed = transpose(input, dim0, dim1)

    return transposed.clone(memory_format=torch.contiguous_format)
