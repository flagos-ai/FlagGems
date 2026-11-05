# Copyright 2026, The FlagOS Contributors.
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
#
import logging

import torch

logger = logging.getLogger(__name__)


def is_same_size(self: torch.Tensor, other: torch.Tensor) -> bool:
    """Compare logical tensor metadata without device work or synchronization."""
    logger.debug("GEMS IS_SAME_SIZE")
    return self.shape == other.shape
