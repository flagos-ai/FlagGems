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
from typing import Tuple

import torch

from flag_gems.runtime.backend._kunlunxin.ops.adaptive_max_pool2d import (
    adaptive_max_pool2d,
)

logger = logging.getLogger(__name__)


def adaptive_max_pool1d(
    input: torch.Tensor, output_size
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_MAX_POOL1D")

    assert input.ndim == 3, f"adaptive_max_pool1d expects 3D input, got {input.ndim}D"

    input_4d = input.unsqueeze(2)

    if isinstance(output_size, int):
        output_size_2d = (1, output_size)
    elif isinstance(output_size, (list, tuple)):
        output_size_2d = (1, output_size[0])
    else:
        raise TypeError(
            f"output_size must be int or list/tuple, got {type(output_size)}"
        )

    output_4d, indices_4d = adaptive_max_pool2d(input_4d, output_size_2d)

    output = output_4d.squeeze(2)
    indices = indices_4d.squeeze(2)

    return output, indices
