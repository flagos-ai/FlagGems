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


def adaptive_avg_pool2d(input: torch.Tensor, output_size):
    logger.debug("GEMS KUNLUNXIN ADAPTIVE_AVG_POOL2D")
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    return torch.ops.aten._adaptive_avg_pool2d(input, output_size)
