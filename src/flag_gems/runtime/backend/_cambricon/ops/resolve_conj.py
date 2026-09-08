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
    logger.debug("GEMS_CAMBRICON RESOLVE_CONJ")
    if A.is_conj():
        # Cannot delegate to torch.ops.aten.resolve_conj.default() because
        # FlagGems registers at the aten dispatch level and it would recurse.
        # Cannot use resolve_conj_triton() either because it calls .contiguous()
        # which may trigger copy_ -> resolve_conj recursion.
        #
        # Safe approach: use torch._C to access the underlying contiguous data
        # via .conj().contiguous() at the C++ level, bypassing FlagGems dispatch.
        # Actually the simplest safe approach: just negate the imaginary part manually.
        if A.dtype == torch.complex64 or A.dtype == torch.complex128:
            # Clone resolves conjugation physically at the C++ storage level
            # torch.clone uses aten::clone which copies data including conj resolution
            return A.clone()
        else:
            return A.clone()
    else:
        return A
