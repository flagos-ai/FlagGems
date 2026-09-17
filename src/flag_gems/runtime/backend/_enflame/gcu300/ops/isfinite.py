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
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

try:
    # _isfinited = tl_extra_shim.isfinited
    _finitef = tl_extra_shim.finitef
except Exception:
    pass
logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isfinite_func(x):
    # return _isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32))
    return _finitef(x.to(tl.float32))


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME ISFINITE")
    if A.is_floating_point():
        assert A.dtype != torch.float64, "Currently do not support fp64"
        return isfinite_func(A)
    else:
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
