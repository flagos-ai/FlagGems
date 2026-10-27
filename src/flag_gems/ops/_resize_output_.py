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
from typing import List

import torch

from flag_gems.ops._resize_output import _resize_output

logger = logging.getLogger(__name__)


def _resize_output_(inp: torch.Tensor, size: List[int], device: torch.device):
    logger.debug("GEMS _RESIZE_OUTPUT_")
    if inp.device == device:
        inp.resize_(size)
        return inp
    else:
        out = _resize_output(inp, size, device)
        inp.resize_(size)
        inp.copy_(out)
        return inp
