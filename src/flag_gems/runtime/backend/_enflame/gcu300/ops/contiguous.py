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

from .copy import copy

logger = logging.getLogger(__name__)


def contiguous(inp, memory_format=torch.contiguous_format):
    assert memory_format == torch.contiguous_format
    logger.debug("GEMS_ENFLAME CONTIGUOUS")
    if inp.is_contiguous(memory_format=memory_format):
        return inp
    out = torch.empty_like(inp, memory_format=memory_format)
    return copy(inp, out0=out)
