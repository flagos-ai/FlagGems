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
"""Public conjugate entry.

``aten::conj`` is a CompositeImplicitAutograd operator that decomposes to
``aten::_conj``, which is a zero-copy view. The public entry therefore only
needs to mirror ATen's non-complex identity path and delegate the complex
case to ``_conj.py``; no Triton kernel is involved.
"""

import logging

import torch

from flag_gems.ops._conj import _conj

logger = logging.getLogger(__name__)


def conj(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS CONJ")
    # ATen's `conj` is an identity for non-complex dtypes.
    if not input.is_complex():
        return input
    return _conj(input)
