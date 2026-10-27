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
from typing import List, Union

import torch

logger = logging.getLogger(__name__)


def dsplit(
    A: torch.Tensor, indices_or_sections: Union[int, List[int]]
) -> List[torch.Tensor]:
    """Split a tensor along the third dimension (depth-wise) into views.

    ``torch.dsplit`` is a view operation: it shares storage with the input and
    does no data movement. The generic implementation routes to
    ``torch.tensor_split`` which (under use_gems) launches a copy kernel per
    split. Here we return zero-copy views via Python slicing instead.

    ``torch.dsplit`` requires the input to have at least 3 dimensions and always
    splits along ``dim=2``.
    """
    logger.debug("GEMS_KUNLUNXIN DSPLIT")

    if A.ndim < 3:
        raise RuntimeError(
            f"dsplit expects a tensor with at least 3 dimensions, "
            f"but got a tensor with {A.ndim} dimensions"
        )

    dim = 2
    dim_size = A.shape[dim]

    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        base = dim_size // n
        remainder = dim_size % n
        split_sizes = [base + (1 if i < remainder else 0) for i in range(n)]
    else:
        indices = [int(i) for i in indices_or_sections]
        split_sizes = [indices[0]]
        split_sizes.extend(indices[i] - indices[i - 1] for i in range(1, len(indices)))
        split_sizes.append(dim_size - indices[-1])

    outputs = []
    start = 0
    for size in split_sizes:
        outputs.append(A[(slice(None),) * dim + (slice(start, start + size),)])
        start += size
    return outputs
