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
MAX_GRID = 48

# Odd minimax polynomial for atan(t) on t in [0, 1]:
# atan(t) ~= t * (P0 + P1*t^2 + P2*t^4 + P3*t^6 + P4*t^8), max abs error
# 2.3e-5 (4x inside the fp32 gate atol 1e-4; invisible to 16-bit outputs).
# Coefficients are shared verbatim with the atan2 op in this directory.
_P0 = tl.constexpr(0.9999693036)
_P1 = tl.constexpr(-0.3316778839)
_P2 = tl.constexpr(0.1851029694)
_P3 = tl.constexpr(-0.09174843878)
_P4 = tl.constexpr(0.02377510071)
_PIO2 = tl.constexpr(1.5707963267948966)


@libentry()
@triton.jit(do_not_specialize=["N"])
def atan_kernel(x_ptr, out_ptr, N, SEL16: tl.constexpr, BLOCK: tl.constexpr):
    # atan(x) = sign(x) * (|x|<=1 ? p(|x|) : pi/2 - p(1/|x|)), with the
    # range-reduced t = min(|x|, 1/|x|) in [0, 1]. The libdevice extern call
    # (software path, measured ~343us/16.7M fp32 vs ~40us for this body) is
    # replaced by the same minimax polynomial already gated by the atan2 tests.
    # SEL16: for 16-bit outputs the sign select runs in the 16-bit domain
    # (magnitude is cast first), keeping two f32 live vectors off the store
    # stage (arcsin op, -31% on the select bucket). fp32 keeps everything in
    # the f32 domain.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        ax = tl.abs(x)
        t = tl.minimum(ax, 1.0 / ax)
        s = t * t
        s2 = s * s
        e0 = _P0 + s * _P1
        e1 = _P2 + s * _P3
        p = e0 + s2 * (e1 + s2 * _P4)
        tp = t * p
        a = tl.where(ax > 1.0, _PIO2 - tp, tp)
        neg = x.to(tl.int32, bitcast=True) < 0
        if SEL16:
            a16 = a.to(out_ptr.dtype.element_ty)
            y = tl.where(neg, -a16, a16)
        else:
            y = tl.where(neg, -a, a).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + off, y, mask=mask)


def _choose_launch(N, itemsize):
    # Returns (BLOCK, num_warps). Bands measured on GCU400 (512M elems):
    #   16-bit: 16384/nw4 (<=2^22) -> 65536/nw2 (<=2^24) -> 131072/nw2
    #           (2^24: 94.2us vs 104us for 16384/nw4; 512M: 2334us vs
    #            5096us for the old extern body at 8192/nw8)
    #   fp32:   32768/nw2 (<=2^24) -> 65536/nw2 (512M: 2564 vs 4718 extern)
    # Tiny shapes use several small CTAs to hide the launch latency
    # (4096 elems: 512/nw4 3.7us vs single-CTA 4.1-4.2us).
    if N <= 8192:
        return max(triton.next_power_of_2(N // 8), 512), 4
    if itemsize <= 2:
        if N <= (1 << 22):
            return 16384, 4
        if N <= (1 << 24):
            return 65536, 2
        return 131072, 2
    if N <= (1 << 24):
        return 32768, 2
    return 65536, 2


def _launch(inp, out):
    N = inp.numel()
    if N == 0:
        return out
    BLOCK, warps = _choose_launch(N, inp.element_size())
    grid_size = min(triton.cdiv(N, BLOCK), MAX_GRID)
    with torch_device_fn.device(inp.device):
        atan_kernel[(grid_size,)](
            inp,
            out,
            N,
            SEL16=inp.element_size() <= 2,
            BLOCK=BLOCK,
            num_warps=warps,
        )
    return out


def atan(A):
    logger.debug("GEMS_ENFLAME ATAN")
    inp = A if A.is_contiguous() else A.contiguous()
    out = torch.empty_like(inp)
    return _launch(inp, out)


def atan_(A):
    logger.debug("GEMS_ENFLAME ATAN_")
    inp = A if A.is_contiguous() else A.contiguous()
    return _launch(inp, A)
