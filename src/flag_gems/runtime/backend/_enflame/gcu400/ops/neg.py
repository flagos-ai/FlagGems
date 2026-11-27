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
from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24

_INT32_SAFE_LIMIT = 0x7FFFFFFF


# GCU400 retune (2026-11-16, topsprof kernel-only A/B, L600): the old
# grid<=NUM_SIPS(24) x num_warps=8 launch left the fp16/bf16 2^29 core
# shapes ~4.9% above the native kernel (706.1us vs 673.2us). Raising the
# CTA cap to the full 48 SPC slots and dropping to num_warps=2 with a
# 2-stage software pipeline closes the gap to native parity (675.8us,
# -4.3%); fp32 ties native at 1.37ms for every probed config and keeps
# BLOCK=32768 so the 2-stage pipeline stays within the 448KB DSM budget
# (48-way residency). Small blocks (<=16384) regress 20-30% on large
# shapes and the EVEN specialisation loses at 2^29, so the masked
# grid-stride loop stays.
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def neg_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    NS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=NS):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        tl.store(out_ptr + off, -x, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def neg_func(x):
    return -x


def _choose_block(n_total, itemsize):
    # do_bench(median) sweeps on the 1R1W sibling flat kernels (rad2deg /
    # sigmoid, plus this op's own topsprof A/B): tiny shapes want the
    # exact next_pow2 tile; mid shapes peak at 8192/32768; 16-bit
    # elements keep BLOCK=65536 at the big end while fp32 stays at
    # 32768 (DSM: BLOCK * itemsize * NS <= 448KB keeps 48-way).
    if n_total <= 8192:
        return triton.next_power_of_2(n_total)
    if n_total <= 65536:
        return 8192
    if n_total <= 524288:
        return 32768
    return 32768 if itemsize == 4 else 65536


def _run_neg(inp, out):
    n_total = inp.numel()
    if n_total == 0:
        return
    block = _choose_block(n_total, inp.element_size())
    if n_total > _INT32_SAFE_LIMIT - block:
        # int32 offset safe domain: fall back to the generic strided
        # kernel instead of silently corrupting flat offsets > 2^31.
        neg_func(inp, out0=out)
        return

    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds. 3 attempts
    # keep deterministic breakage loud while riding out the flake
    # (rad2deg.py / atan2_.py / sinh.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                neg_flat_kernel[(grid_size,)](
                    inp,
                    out,
                    n_total,
                    BLOCK=block,
                    NS=2,
                    num_warps=2,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "neg flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def neg(A):
    logger.debug("GEMS_ENFLAME NEG")
    if not A.is_contiguous():
        # Single strided kernel instead of a contiguous() copy plus a
        # second pass over the data.
        return neg_func(A)
    out = torch.empty_like(A)
    _run_neg(A, out)
    return out


def neg_(A):
    logger.debug("GEMS_ENFLAME NEG_")
    if not A.is_contiguous():
        # In-place through the strided kernel keeps the storage-aliasing
        # semantics without the contiguous copy / copy_() round trip.
        neg_func(A, out0=A)
        return A
    _run_neg(A, A)
    return A
