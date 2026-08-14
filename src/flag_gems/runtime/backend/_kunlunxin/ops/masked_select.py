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


def masked_select(inp, mask):
    logger.debug("GEMS_KUNLUNXIN MASKED_SELECT")
    # The custom compaction path needs a separate nonzero pass plus a gather and
    # is about 10x slower on large P800 inputs.  Redispatch to XDNN's fused
    # implementation, which also preserves the exact broadcast/error contract.
    # Only the default overload is replaced by FlagGems.  The out overload
    # remains XDNN-backed, so an initially empty destination reaches the fused
    # vendor implementation without re-entering this function.
    out = torch.empty((0,), dtype=inp.dtype, device=inp.device)
    return torch.ops.aten.masked_select.out(inp, mask, out=out)
