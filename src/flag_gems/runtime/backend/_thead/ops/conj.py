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
"""Thead ``_conj``: zero-copy conjugate view.

``torch._conj`` (aten::_conj) is a pure view operation: eager PyTorch returns
a storage-sharing alias with the lazy conjugate bit toggled (~0.00016 ms, no
data traffic). Materializing it with a Triton kernel is a semantic deviation
and an O(N) regression. Delegating to ATen's backend-agnostic implementation
under a dispatch guard keeps the view semantics, the shared storage and the
``ConjBackward0`` autograd rule.
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
    logger.debug("GEMS_THEAD CONJ")

    # Real tensors: conjugate is the identity; mirror torch._conj's
    # zero-copy alias behavior.
    if not input.is_complex():
        return input.as_strided(input.size(), input.stride(), input.storage_offset())

    # Complex: native lazy conjugate view (zero copy, shared storage).
    with torch._C._ExcludeDispatchKeyGuard(_BACKEND_KEYSET):
        return torch._conj(input)
