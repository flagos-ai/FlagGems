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
# One-hop redispatch onto the native SparseCPU kernel: the keyset pins the
# dispatch to exactly that key, skipping the Autograd/ADInplaceOrView layers
# (and therefore this override) without any per-call TLS guard object. This is
# the same landing point the previous ``_AutoDispatchBelowAutograd`` re-entry
# used for every COO tensor reachable on this platform: FlagGems/GCU sparse COO
# tensors are CPU-impl tensors (components may live on GCU), so their native
# kernel is SparseCPU.
_NATIVE_REDISPATCH = _NATIVE_COALESCED.redispatch
_SPARSE_CPU_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.SparseCPU)
_ERROR_MESSAGE = (
    "_coalesced_ only supports sparse coordinate (coo) tensors, "
    "but got unsupported layout"
)

logger = logging.getLogger(__name__)


def _coalesced_(self: torch.Tensor, coalesced: bool) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME _COALESCED_")
    if self.layout != _SPARSE_COO_LAYOUT:
        raise RuntimeError(_ERROR_MESSAGE)
    _NATIVE_REDISPATCH(_SPARSE_CPU_KEYSET, self, coalesced)
    return self
