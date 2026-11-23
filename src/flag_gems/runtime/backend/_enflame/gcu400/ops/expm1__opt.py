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

"""expm1 candidate v1: override CodeGenConfig.max_grid_size per operator.

Same semantics as item 15's expm1_out, only outplace (out0 omitted).
Following the zero-branch template: no hand-written kernel, only enlarge ENFLAME's default
`max_grid_size=(12,1,1)` to `(1024,1,1)` and regenerate the pointwise_dynamic
wrapper, keeping `INT_TO_FLOAT` promotion unchanged.

The baseline reuses `base_eoN`: collected in the same pytest process and time window as expm1_out,
the log contains expm1's 3 rounds of 15 cases, filterable by op_name.
Global CODEGEN_COFIGS is not modified, satisfying the skill's "pointwise_dynamic must be specialized" requirement.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config

logger = logging.getLogger(__name__)


_BASE = get_codegen_config()
_OPT_CONFIG = CodeGenConfig(
    max_tile_size=_BASE.max_tile_size,
    max_grid_size=(1024, 1, 1),
    max_num_warps_per_cta=_BASE.max_num_warps_per_cta,
    prefer_block_pointer=_BASE.prefer_block_pointer,
    prefer_1d_tile=_BASE.prefer_1d_tile,
)

_GRID_SIZE = 48
_INT32_MAX = 2**31 - 1


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=_OPT_CONFIG)
@triton.jit
def _expm1_func(x):
    return tl.exp(x.to(tl.float32)) - 1.0


def _choose_block(n, itemsize):
    # 128KB-tile baseline widened only in the n>=2^28 large band (256KB tile:
    # 2B -> 131072, 4B -> 65536); mid/small bands keep the round-1 tiles.
    # Calibrated on the identical compute chain (exp+sub) in special_expm1
    # round 2 (2026-11-16, interleaved A/B 4 pairs + boundary re-scan):
    #   fp16 512M +3.7% / bf16 512M +2.8% / fp32 512M +0.8-1.0%;
    #   B131072 at 16.7M is ~2% slower; B65536@fp32 at 2^25+1 is ~11% slower
    #   (2^27 parity) — fp32 crossover sits at n>=2^28.
    if itemsize <= 2:
        block = 131072 if n > (1 << 28) else 65536
    else:
        block = 65536 if n >= (1 << 28) else 32768
    if n < block:
        block = max(1024, triton.next_power_of_2(n))
    return block


@triton.jit
def _expm1_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    ST: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tile = (n_elements + BLOCK - 1) // BLOCK
    for tile_id in tl.range(pid, num_tile, GRID, num_stages=ST):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.exp(x.to(tl.float32)) - 1.0
        tl.store(y_ptr + offsets, y.to(y_ptr.type.element_ty), mask=mask)


def expm1(A):
    logger.debug("GEMS_ENFLAME EXPM1")
    if A.is_floating_point() and A.is_contiguous() and A.numel() > 0:
        n = A.numel()
        block = _choose_block(n, A.element_size())
        if triton.cdiv(n, block) * block <= _INT32_MAX:
            out = torch.empty_like(A)
            grid = (min(_GRID_SIZE, triton.cdiv(n, block)), 1, 1)
            with torch_device_fn.device(A.device.index):
                _expm1_kernel[grid](
                    A.view(-1),
                    out.view(-1),
                    n,
                    BLOCK=block,
                    GRID=grid[0],
                    ST=1,
                    num_warps=2,
                )
            return out
    return _expm1_func(A)
