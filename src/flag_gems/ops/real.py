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

from flag_gems import runtime

# Parsing also resolves renamed PrivateUse1 keys such as MUSA.
_BACKEND_KEYSET = torch._C.DispatchKeySet(
    torch._C._parse_dispatch_key(runtime.device.dispatch_key)
)
_SELECT = torch.Tensor.select
_VIEW_AS_REAL = torch.view_as_real
_FLAG_GEMS_LOGGER = logging.getLogger("flag_gems")
logger = logging.getLogger(__name__)


def real(input: torch.Tensor) -> torch.Tensor:
    r"""Return the real component of ``input`` as a zero-copy view.

    Real-valued inputs are returned unchanged. Complex inputs are reinterpreted
    as their interleaved real storage and select the real component.
    """
    if input.is_conj():
        return real_conjugate(input)

    return real_device(input)


def real_device(input: torch.Tensor) -> torch.Tensor:
    """Device-dispatch fast path for tensors without the conjugate bit."""
    # Keep record=True observable without paying Logger.debug's disabled-path
    # cost in the metadata-only wrapper benchmark.
    if _FLAG_GEMS_LOGGER.handlers:
        logger.debug("GEMS REAL")

    if not input.dtype.is_complex:
        return input

    return _SELECT(_VIEW_AS_REAL(input), -1, 0)


def real_conjugate(input: torch.Tensor) -> torch.Tensor:
    """Conjugate-dispatch implementation for :func:`real`."""
    if _FLAG_GEMS_LOGGER.handlers:
        logger.debug("GEMS REAL")

    # Redispatch past FlagGems' device `_conj` kernel so aten can create the
    # conjugate-clearing alias with its native autograd ViewInfo. Clearing the
    # bit directly would lose view-replay metadata for inplace updates.
    with torch._C._ExcludeDispatchKeyGuard(_BACKEND_KEYSET):
        source = torch._conj(input)

    return _SELECT(_VIEW_AS_REAL(source), -1, 0)
