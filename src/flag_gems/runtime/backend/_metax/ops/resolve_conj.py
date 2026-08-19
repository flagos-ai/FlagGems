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


def resolve_conj(A: torch.Tensor):
    logger.debug("GEMS_METAX RESOLVE_CONJ")
    if not A.is_conj():
        return A
    # A has the conj bit set: logical value = conj(storage).
    # resolve_conj materializes this into a contiguous tensor without the conj bit.
    #
    # We cannot use A.real / A.imag because those trigger operator dispatch back
    # into _conj → resolve_conj, causing infinite recursion.
    #
    # torch.clone() on a conjugated tensor physically resolves the conjugation
    # at the C++ aten::clone level (copies data with conj applied) and returns
    # a new tensor without the conj bit. This does not recurse through resolve_conj.
    #
    # The shared empty.py skips the triton kernel for complex dtypes,
    # so clone() → _to_copy → empty no longer fails with KeyError on complex64.
    return A.clone()
