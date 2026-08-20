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
from typing import List, Tuple, Union

import torch

from flag_gems.runtime.backend._enflame.gcu400.ops.cat import cat

logger = logging.getLogger(__name__)


def vstack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME VSTACK")

    n = len(tensors)
    if n == 0:
        raise RuntimeError("vstack expected a non-empty TensorList")

    t0 = tensors[0]
    if t0.ndim < 2:
        aligned = list(torch.atleast_2d(tensors))
    else:
        aligned = list(tensors) if not isinstance(tensors, list) else tensors

    return cat(aligned, dim=0)
