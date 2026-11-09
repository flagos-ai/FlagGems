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
"""Kunlunxin ``_conj``: zero-copy conjugate view.

``aten::_conj`` only toggles the lazy conjugate bit on a storage-sharing
alias; nothing is read or written. Delegating to ATen's backend-agnostic
CompositeExplicitAutograd implementation keeps that view semantics (and its
``ConjBackward0`` autograd rule) while bypassing FlagGems' own registration
for this one call.
"""

import logging

import torch

from flag_gems import runtime

logger = logging.getLogger(__name__)

# NOTE: parse_dispatch_key also resolves renamed PrivateUse1 keys.
_BACKEND_KEYSET = torch._C.DispatchKeySet(
    torch._C._parse_dispatch_key(runtime.device.dispatch_key)
)


def _conj(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN CONJ")
    if not input.is_complex():
        raise RuntimeError("_conj only supports complex tensors")

    with torch._C._ExcludeDispatchKeyGuard(_BACKEND_KEYSET):
        return torch._conj(input)
