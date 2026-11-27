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

"""square_ candidate v2: in-place 1D hand-written kernel.

Production square_ is `_kernel_for(A.dtype)(A, out0=A)`: fp16/bf16 promote to fp32, multiply
then write back to the narrow dtype; fp32 multiplies directly. pointwise_dynamic's ENFLAME
max_grid_size=(12,1,1) leaves large shapes under-parallelized (measured ~6ms vs torch 0.76ms).
Three BLOCK tiers reuse the strategy validated by square/square_out.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def square_inplace_flat_kernel(X, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    # int32 safety domain: host guard keeps n <= 2^31 - 2^17 so that
    # block_id * BLOCK + arange (int32) cannot wrap negative on the tail
    # block (a wrapped-negative off would pass `off < N_total` and read
    # out of bounds).
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(X + off, mask=mask).to(tl.float32)
        y = x * x
        tl.store(X + off, y.to(X.dtype.element_ty), mask=mask)


def _choose_launch(n_total, itemsize):
    # BLOCK ladder unchanged (validated on this 1R1W flat kernel body); the
    # fp32 small-block bands want num_warps 8 (2026-11-16 interleaved 5-round
    # probe, do_bench median w1000/r100: n=65536 4.73->4.09us -13.5%,
    # n=262144 5.23->4.84us -7.5%, n=524288 5.83->5.73us -1.7%, never worse;
    # same choice as the abs family ladder for the <=524288 band). 16-bit
    # bands and >524288 keep nw2 (probe WITHIN-NOISE / optimal, and the fp32
    # big bucket is at the in-place memory floor).
    if n_total <= 65536:
        return 8192, 8 if itemsize == 4 else 2
    if n_total <= 524288:
        return 32768, 8 if itemsize == 4 else 2
    if n_total > (1 << 28) and itemsize <= 2:
        return 131072, 2
    return 65536, 2


def square_(A):
    logger.debug("GEMS_ENFLAME SQUARE_")
    n = A.numel()
    if (
        n < 65536
        or not A.is_contiguous()
        or A.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or n >= (1 << 31) - (1 << 17)
    ):
        # Small shapes are at the launch-overhead floor; reusing the same underlying call as production guarantees
        # exactly 1.000x
        # (2026-11-16 probe: tiny (64,64) fallback 3.11us beats all flat variants
        # 3.16-3.26us; the 8192<n<65536 band goes the same way, and the fallback scope is unchanged).
        from flag_gems.ops.square import _kernel_for

        _kernel_for(A.dtype)(A, out0=A)
        return A
    BLOCK, num_warps = _choose_launch(n, A.element_size())
    grid_size = min(triton.cdiv(n, BLOCK), NUM_SIPS * 2)
    # Cold-compile flake retry, same toolchain-family precedent as square.py v3.
    for attempt in range(3):
        try:
            with torch_device_fn.device(A.device):
                square_inplace_flat_kernel[(grid_size,)](
                    A,
                    n,
                    BLOCK=BLOCK,
                    num_warps=num_warps,
                )
            return A
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "square_inplace flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )
