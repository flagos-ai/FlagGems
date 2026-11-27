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
"""``aten::_conj``: a pure metadata op that returns a zero-copy view with the conjugate bit set.

ATen's ``_conj`` moves no data; it only sets the conjugate bit and returns a view.
Materializing the negated imaginary part belongs to ``conj_physical`` / ``resolve_conj``.
This used to run a Triton flat copy kernel: 48.8us for 1024 elements, while the native
view path needs only 5.3us (about 9x).
"""

import logging

import torch

logger = logging.getLogger(__name__)

_PRIVATE_USE1 = torch._C.DispatchKey.PrivateUse1


def _conj(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME CONJ")
    if not input.is_complex():
        raise RuntimeError("_conj only supports complex tensors")

    # Calling ``input.conj()`` directly re-enters this function (FlagGems registers ``aten::conj`` /
    # ``aten::_conj`` on the same dispatch key), so temporarily exclude PrivateUse1 first,
    # letting the call fall through to the native composite implementation, which returns a view with the conjugate bit
    # set.
    # This switch is thread-local and must be set/restored in pairs.
    torch._C._dispatch_tls_set_dispatch_key_excluded(_PRIVATE_USE1, True)
    try:
        return torch.ops.aten.conj(input)
    finally:
        torch._C._dispatch_tls_set_dispatch_key_excluded(_PRIVATE_USE1, False)
