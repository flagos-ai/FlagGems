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

from flag_gems.ops.adaptive_avg_pool2d import adaptive_avg_pool2d

logger = logging.getLogger(__name__)


def adaptive_avg_pool1d(input: torch.Tensor, output_size):
    """Adaptive average pooling operation over 1D input.

    Implemented by reshaping to 2D and calling adaptive_avg_pool2d with output_size=(1, output_size).
    """
    logger.debug("GEMS ADAPTIVE_AVG_POOL1D")

    # Input shape: (N, C, L) -> reshape to (N, C, 1, L)
    assert input.ndim == 3, f"adaptive_avg_pool1d expects 3D input, got {input.ndim}D"

    input_4d = input.unsqueeze(2)  # (N, C, L) -> (N, C, 1, L)

    # Convert 1D output_size to 2D: output_size -> (1, output_size)
    if isinstance(output_size, int):
        output_size_2d = (1, output_size)
    else:
        output_size_2d = (1, output_size[0])

    # Call adaptive_avg_pool2d
    output_4d = adaptive_avg_pool2d(input_4d, output_size_2d)

    # Reshape back: (N, C, 1, L_out) -> (N, C, L_out)
    return output_4d.squeeze(2)
