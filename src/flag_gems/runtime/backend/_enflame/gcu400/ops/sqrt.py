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

# Tuned on GCU400 for the streaming pointwise family (see log1p/topk_softplus_sqrt
# precedents): fixed grid capped at 2 CTAs per SIP, no software pipelining (the
# chain is short enough to hide behind the loads), BLOCK ladder below.
_GRID_CAP = NUM_SIPS * 2
_SMALL_MAX = 65536
_MID_MAX = 1 << 26  # 2**26: above this, 16-bit dtypes prefer the 128KiB tile
_BLOCK_LARGE = (
    1 << 17
)  # 131072 elements: best for 16-bit >= 2**26 (within ~2% of 65536)
_BLOCK_MID = 1 << 16  # 65536 elements


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def sqrt_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    USE_RSQRT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        if USE_RSQRT:
            # GCU400 lowering: tl.sqrt is a ~0.49ns/elem software chain that
            # cannot overlap the streaming loads, while tl.rsqrt is a hardware
            # primitive (~0.03ns/elem) -- decisive for 16-bit dtypes where the
            # chain is a visible fraction of the 1R1W floor (fp32 is already
            # memory-bound and keeps tl.sqrt for exact subnormal behaviour).
            # sqrt(x) == x * rsqrt(x) for every finite positive normal x; the
            # raw product only misbehaves on x in {0, -0, 1, +inf} where
            # 0*inf / inf*0 = NaN, and for exactly that set x*x == x holds while
            # sqrt(x) == x -- so one select on x*x == x restores IEEE semantics
            # (negatives and NaN keep propagating through rsqrt(NaN) naturally).
            f = tl.where(x * x == x, 1.0, tl.rsqrt(x))
            out = x * f
        else:
            out = tl.sqrt(x)
        tl.store(out_ptr + off, out, mask=mask)


def _choose_params(N_total, use_rsqrt):
    if N_total <= _SMALL_MAX:
        # Amortize launch latency with several small CTAs (log1p precedent).
        block = max(512, triton.next_power_of_2(max(N_total, 1) // 2))
        return block, 2
    if use_rsqrt and N_total >= _MID_MAX:
        return _BLOCK_LARGE, 2
    return _BLOCK_MID, 2


def _launch_sqrt(inp, out, N_total):
    use_rsqrt = inp.dtype in (torch.float16, torch.bfloat16)
    BLOCK, nw = _choose_params(N_total, use_rsqrt)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, _GRID_CAP)

    with torch_device_fn.device(inp.device):
        sqrt_flat_kernel[(grid_size,)](
            inp,
            out,
            N_total,
            BLOCK=BLOCK,
            USE_RSQRT=use_rsqrt,
            num_warps=nw,
        )


def sqrt(A):
    logger.debug("GEMS_ENFLAME SQRT")
    inp = A.contiguous()
    N_total = inp.numel()
    out = torch.empty_like(inp)
    _launch_sqrt(inp, out, N_total)
    return out


def sqrt_(A):
    logger.debug("GEMS_ENFLAME SQRT_")
    inp = A.contiguous()
    N_total = inp.numel()
    _launch_sqrt(inp, A, N_total)
    return A
