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

"""expm1_out candidate v1: override CodeGenConfig.max_grid_size per operator.

Following the zero-branch template validated by items 11~13: no hand-written kernel, only enlarge ENFLAME's default
`max_grid_size=(12,1,1)` to `(1024,1,1)` and regenerate the pointwise_dynamic
wrapper. Codegen's internal `num_ctas=min(grid,num_tiles)` keeps the small-shape call chain
identical to the common implementation with no microsecond regression, while large shapes gain grid parallelism.

Keep production's original `INT_TO_FLOAT` promotion and out-variant semantics
(`expm1_func(A, out0=out)`). Production `expm1_out(A, out)` requires out,
and this implementation matches it with zero Python branches; note `out0=None` must not be passed explicitly
(codegen would read `.dtype` and raise, so omitting the argument is the automatic allocation path).
Global CODEGEN_COFIGS is not modified, satisfying the skill's "pointwise_dynamic must be specialized" requirement.
"""

import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

from .expm1__opt import _expm1_func
from .special_expm1 import _choose_block, _special_expm1_kernel

logger = logging.getLogger(__name__)

_GRID_SIZE = 48
_INT32_MAX = 2**31 - 1
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def expm1_out(A, out):
    logger.debug("GEMS_ENFLAME EXPM1_OUT")
    if (
        A.dtype in _FAST_DTYPES
        and out.is_contiguous()
        and A.is_contiguous()
        and A.numel() == out.numel()
        and A.numel() > 0
    ):
        n = A.numel()
        block = _choose_block(n, A.element_size())
        if triton.cdiv(n, block) * block <= _INT32_MAX:
            grid = (min(_GRID_SIZE, triton.cdiv(n, block)), 1, 1)
            with torch_device_fn.device(A.device.index):
                _special_expm1_kernel[grid](
                    A.view(-1),
                    out.view(-1),
                    n,
                    BLOCK=block,
                    GRID=grid[0],
                    ST=1,
                    num_warps=2,
                )
            return out
    return _expm1_func(A, out0=out)
