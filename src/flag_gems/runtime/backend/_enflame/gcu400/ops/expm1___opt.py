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

"""expm1_ candidate v1: reuse the expm1__opt jit object that already overrides config per operator.

Production `expm1_(A)` and `expm1(A)` share `expm1_func`, the only difference being
`out0=A` for in-place write-back. `expm1__opt.py` delivered in item 17 already changed
`max_grid_size=(12,1,1)->(1024,1,1)` and kept `INT_TO_FLOAT`
promotion (expm1 production-only 3 rounds PASS 3.220x), so this item imports and reuses it directly,
with no duplicate jit objects and zero Python branches.

`out0=A` guarantees the in-place contract; integral inputs are promoted to floating-point output,
matching the common implementation.
"""

import logging

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.expm1__opt import _expm1_func

logger = logging.getLogger(__name__)

__all__ = ["expm1_"]

_GRID_SIZE = 48
_INT32_MAX = 2**31 - 1


def _choose_block(n, itemsize):
    # In-place A/B calibration (round-robin do_bench median, job
    # j20261117030037_e92807ca; re-anchored on the 2026-11-18 toolchain):
    #   - big band n > 2^28: 256KB tile (2B -> 131072 / 4B -> 65536), nw=2
    #     (sibling special_expm1 round-2 ladder; 512M fp16/bf16/fp32 optimal).
    #   - mid band 2^18 < n <= 2^28: block ~ n/128 clamped to the ladder cap
    #     (fixes the single-CTA/oversized-tile hole at 2^19-2^21; 2^20 fp16
    #     6.40us vs 7.27us, 2^22 fp16 12.21 vs 12.64; 2^24 hits the cap and
    #     matches the sibling ladder exactly).
    #   - small band n <= 2^18: multi-CTA small tiles ~ n/4 (n=4096: 3.19us
    #     B1024/g4 vs 3.37us single-CTA; n=65536: 4.06-4.11us vs 5.81-6.43us).
    if n > (1 << 28):
        return 131072 if itemsize <= 2 else 65536
    if n > (1 << 18):
        cap = 65536 if itemsize <= 2 else 32768
        return max(4096, min(cap, triton.next_power_of_2(n) >> 7))
    return max(256, min(4096, triton.next_power_of_2((n + 3) // 4)))


@triton.jit
def _expm1_inplace_kernel(
    x_ptr,
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
        tl.store(x_ptr + offsets, y.to(x_ptr.type.element_ty), mask=mask)


def expm1_(A):
    logger.debug("GEMS_ENFLAME EXPM1_")
    if A.is_floating_point() and A.is_contiguous() and A.numel() > 0:
        n = A.numel()
        block = _choose_block(n, A.element_size())
        if triton.cdiv(n, block) * block <= _INT32_MAX:
            grid = (min(_GRID_SIZE, triton.cdiv(n, block)), 1, 1)
            with torch_device_fn.device(A.device.index):
                _expm1_inplace_kernel[grid](
                    A.view(-1),
                    n,
                    BLOCK=block,
                    GRID=grid[0],
                    ST=1,
                    num_warps=2,
                )
            return A
    return _expm1_func(A, out0=A)
