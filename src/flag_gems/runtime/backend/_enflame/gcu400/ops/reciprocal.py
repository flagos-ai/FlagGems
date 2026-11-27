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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# int32 offset safety range: even if the tail block is masked out, block_id * BLOCK + arange is still computed,
# up to num_blocks * BLOCK - 1; limiting n_total <= 2^31 - 2^17 (BLOCK <= 131072)
# ensures the int32 offset does not wrap around to a negative value in the tail block.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def reciprocal_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """1D flattened grid-stride reciprocal kernel.

    EVEN means N_total is a multiple of BLOCK, which drops the mask: on GCU400 a masked
    access degrades to per-lane predicates, measuring about 29% slower for large fp32 shapes.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    if EVEN:
        num_blocks = N_total // BLOCK
        for block_id in tl.range(pid, num_blocks, num_pids):
            off = block_id * BLOCK + arange
            x = tl.load(x_ptr + off)
            tl.store(out_ptr + off, 1.0 / x.to(tl.float32))
    else:
        num_blocks = (N_total + BLOCK - 1) // BLOCK
        for block_id in tl.range(pid, num_blocks, num_pids):
            off = block_id * BLOCK + arange
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
            tl.store(out_ptr + off, 1.0 / x.to(tl.float32), mask=mask)


# GCU400 tuning notes (2026-11-18, same-source do_bench A/B, compared against torch / pointwise_dynamic):
# - The no-mask fast path is the biggest win: fp32 512M 1.967ms -> 1.395ms (w4),
#   fp32 16.7M 0.093ms -> 0.053ms (w4).
# - Large 16-bit shapes prefer num_warps=2 + 256KB tiles; fp32 prefers num_warps=4.
# - For 512M 16-bit, raising BLOCK from 65536 to 131072 gives another 2.7% reduction.
# - For tiny (<=8192 elements), use a next_pow2 tile + w4; pure launch overhead dominates, 3x better than the original
# implementation.
def _choose_launch(n_total, itemsize):
    if n_total <= 8192:
        return max(triton.next_power_of_2(n_total), 64), 4
    if n_total <= 65536:
        return 8192, 2
    if itemsize >= 4:
        # For fp32 medium shapes (16.7M), a 128KB tile is slightly better than 256KB: 0.0531ms vs 0.0539ms;
        # 512M still uses 256KB tiles (0.0531 gains only 1.5%; large shapes are bandwidth-bound).
        return (32768, 4) if n_total <= (1 << 25) else (65536, 4)
    if n_total > (1 << 26):
        return 131072, 2
    return 65536, 2


def _launch_reciprocal(inp, out):
    n_total = inp.numel()
    if n_total == 0:
        return
    block, num_warps = _choose_launch(n_total, inp.element_size())
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    # The external llvm converter occasionally fails during cold compilation (same precedent in neg.py / abs.py);
    # retrying succeeds.
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                reciprocal_flat_kernel[(grid_size,)](
                    inp,
                    out,
                    n_total,
                    BLOCK=block,
                    EVEN=n_total % block == 0,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "reciprocal flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _flat_path_ok(A):
    return (
        A.is_contiguous() and A.dtype in _FLAT_DTYPES and A.numel() < _INT32_SAFE_LIMIT
    )


def reciprocal(A):
    logger.debug("GEMS_ENFLAME RECIPROCAL")
    if not _flat_path_ok(A):
        from flag_gems.ops.reciprocal import reciprocal as _public

        return _public(A)
    out = torch.empty_like(A)
    _launch_reciprocal(A, out)
    return out


def reciprocal_(A):
    logger.debug("GEMS_ENFLAME RECIPROCAL_")
    if not _flat_path_ok(A):
        from flag_gems.ops.reciprocal import reciprocal_ as _public

        return _public(A)
    _launch_reciprocal(A, A)
    return A
