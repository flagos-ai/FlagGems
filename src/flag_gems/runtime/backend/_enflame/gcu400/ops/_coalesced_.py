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

_SPARSE_COO_LAYOUT = torch.sparse_coo
_NATIVE_COALESCED = torch.ops.aten._coalesced_.default
_DISPATCH_BELOW_AUTOGRAD = torch._C._AutoDispatchBelowAutograd
_ERROR_MESSAGE = (
    "_coalesced_ only supports sparse coordinate (coo) tensors, "
    "but got unsupported layout"
)

logger = logging.getLogger(__name__)


def _coalesced_(self: torch.Tensor, coalesced: bool) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME_GCU400 _COALESCED_")
    if self.layout != _SPARSE_COO_LAYOUT:
        raise RuntimeError(_ERROR_MESSAGE)
    # Construct the guard per call: ``_AutoDispatchBelowAutograd`` activates the
    # Autograd-key exclusion in __init__ (RAII), so a module-level instance
    # would disable autograd process-wide from import onward and break every
    # backward test (regression of 890f0b7cd, re-fixed).
    with _DISPATCH_BELOW_AUTOGRAD():
        _NATIVE_COALESCED(self, coalesced)
    return self
