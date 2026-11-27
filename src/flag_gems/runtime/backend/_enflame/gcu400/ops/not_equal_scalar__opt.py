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

"""not_equal_scalar candidate v1: override CodeGenConfig.max_grid_size per operator.

Following the zero-branch template validated by item 11's floor: no hand-written kernel, only enlarge ENFLAME's default
`max_grid_size=(12,1,1)` to `(1024,1,1)` and regenerate the pointwise_dynamic
wrapper. Codegen's internal `num_ctas=min(grid,num_tiles)` keeps the small-shape call chain
identical to the common implementation with no microsecond regression, while large shapes gain parallelism.

Keep production's `is_tensor=[True, False]` and `ALWAYS_BOOL` promotion semantics
(bool output, scalar B compared in fp32). Global CODEGEN_COFIGS is not modified.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config

logger = logging.getLogger(__name__)
device = device.name

_BASE = get_codegen_config()
_OPT_CONFIG = CodeGenConfig(
    max_tile_size=_BASE.max_tile_size,
    max_grid_size=(1024, 1, 1),
    max_num_warps_per_cta=_BASE.max_num_warps_per_cta,
    prefer_block_pointer=_BASE.prefer_block_pointer,
    prefer_1d_tile=_BASE.prefer_1d_tile,
)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_INT32_LIMIT = 0x7FFFFFFF
_GRID_LIMIT = 65535


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=_OPT_CONFIG,
)
@triton.jit
def _not_equal_func_scalar(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


@triton.jit(do_not_specialize=["N"])
def _not_equal_scalar_flat_kernel(
    X, Out, Y, N, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(X + offs)
        tl.store(Out + offs, x != Y.to(x.dtype))
    else:
        mask = offs < N
        x = tl.load(X + offs, mask=mask, other=0)
        tl.store(Out + offs, x != Y.to(x.dtype), mask=mask)


def not_equal_scalar(A, B):
    logger.debug("GEMS_ENFLAME NOT_EQUAL_SCALAR")
    if (
        isinstance(B, (int, float))
        and A.device.type == device
        and A.dtype in _FLOAT_DTYPES
        and A.is_contiguous()
    ):
        try:
            y = float(B)
        except OverflowError:
            return _not_equal_func_scalar(A, B)
        n = A.numel()
        if n == 0:
            return torch.empty_like(A, dtype=torch.bool)
        es = A.element_size()
        if es <= 2:
            if n <= 8192:
                block, num_warps = max(16, triton.next_power_of_2(n)), 4
            elif n <= (1 << 17):
                block, num_warps = 8192, 4
            elif n <= (1 << 19):
                block, num_warps = 16384, 4
            elif n <= (1 << 20):
                block, num_warps = 65536, 4
            elif n <= (1 << 24):
                block, num_warps = 131072, 4
            else:
                block, num_warps = 131072, 2
        else:
            if n <= 8192:
                block, num_warps = max(16, triton.next_power_of_2(n)), 4
            elif n <= (1 << 18):
                block, num_warps = 8192, 4
            elif n <= (1 << 20):
                block, num_warps = 32768, 4
            else:
                block, num_warps = 65536, 4
        grid = (triton.cdiv(n, block),)
        if n < _INT32_LIMIT and grid[0] <= _GRID_LIMIT:
            out = torch.empty_like(A, dtype=torch.bool)
            _not_equal_scalar_flat_kernel[grid](
                A,
                out,
                y,
                n,
                BLOCK=block,
                EVEN=(n % block == 0),
                num_warps=num_warps,
            )
            return out
    return _not_equal_func_scalar(A, B)
