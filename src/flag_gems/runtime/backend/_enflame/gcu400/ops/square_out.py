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

"""square_out candidate v2: on top of v1, small shapes fall back to the common implementation.

v1's 3 benchmark rounds: overall 4.15x / fp16 4.32x / fp16_large 4.34x,
but small shapes (64x64) showed slight 0.920~0.987 regressions, triggering the per-case p10>=0.95
gate and FAIL. v2 sends n<65536 straight to common pointwise_dynamic,
removing that regression; everything else keeps v1's large-BLOCK hand-written kernel.
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
def square_out_flat_kernel(X, Y, N_total, BLOCK: tl.constexpr):
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
        tl.store(Y + off, y.to(Y.dtype.element_ty), mask=mask)


def _choose_block(n_total, itemsize):
    # Same ladder as square.py v3 (validated on this 1R1W flat kernel body).
    if n_total <= 65536:
        return 8192
    if n_total <= 524288:
        return 32768
    if n_total > (1 << 28) and itemsize <= 2:
        return 131072
    return 65536


def _public(A, out):
    from flag_gems.ops.square import square_out as _impl

    return _impl(A, out=out)


def square_out(A, *, out=None):
    logger.debug("GEMS_ENFLAME SQUARE_OUT")
    if out is None:
        out = torch.empty_like(A)
    n = A.numel()
    if (
        n < 65536
        or not A.is_contiguous()
        or not out.is_contiguous()
        or A.shape != out.shape
        or A.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or out.dtype != A.dtype
        or n >= (1 << 31) - (1 << 17)
    ):
        return _public(A, out)
    BLOCK = _choose_block(n, A.element_size())
    grid_size = min(triton.cdiv(n, BLOCK), NUM_SIPS * 2)
    # Cold-compile flake retry, same toolchain-family precedent as square.py v3.
    for attempt in range(3):
        try:
            with torch_device_fn.device(A.device):
                square_out_flat_kernel[(grid_size,)](
                    A,
                    out,
                    n,
                    BLOCK=BLOCK,
                    num_warps=2,
                )
            return out
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "square_out flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )
