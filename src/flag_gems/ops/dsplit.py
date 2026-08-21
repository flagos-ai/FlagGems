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


def dsplit(input: torch.Tensor, indices_or_sections: Union[int, List[int]]):
    """Split a tensor along the third axis (depth-wise).

    This is equivalent to torch.tensor_split with dim=2 for 3D+ tensors.
    Returns a tuple of views (zero-copy).
    """
    logger.debug("GEMS DSPLIT")

    # dsplit splits along dim=2 (depth)
    if isinstance(indices_or_sections, int):
        # Equal splits
        return torch.split(input, input.shape[2] // indices_or_sections, dim=2)
    else:
        # Custom split indices
        return torch.tensor_split(input, indices_or_sections, dim=2)
