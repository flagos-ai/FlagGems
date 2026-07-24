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
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def conj_func(x):
    return x ^ -(1 << 63)


def resolve_conj(A: torch.Tensor):
    logger.debug("GEMS_CAMBRICON RESOLVE_CONJ")
    if A.is_conj():
        assert (
            A.dtype == torch.cfloat
        ), "The `resolve_conj` operation in FlagGems currently only supports the `torch.cfloat` type"
        typed_view = torch.view_as_real(A.conj()).view(torch.int64)
        out = conj_func(typed_view)
        return torch.view_as_complex(out.view(torch.float))
    else:
        return A
