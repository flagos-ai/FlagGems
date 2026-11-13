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
import time

import torch
import triton
import triton.language as tl

from flag_gems.ops.arccos import arccos_kernel as _strided_arccos_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_PROGRAMS = 48

# arccos(x) = 2*atan(sqrt((1-|x|)/(1+|x|))) = t*Q(t^2), t = sqrt((1-|x|)/(1+|x|))
# in [0, 1]. The libdevice acos lowers to a per-vector extern call
# (tops_ef_vacosf_v1/v2_fp32) on this backend; the angle is evaluated with
# plain arithmetic instead: one native rsqrt for t, a deg-4 minimax odd
# polynomial Q (Lawson/IRLS fit on [0,1], fp32-rounded coefficients), and a
# sign-bit quadrant fixup. Measured max abs error 2.58e-5 against the accuracy
# gate atol=1e-4 (3.9x margin, deg-6 gives 9.0e-7 at ~9% cost).
# Out-of-domain and NaN inputs propagate naturally: |x|>1 makes u=1-|x|
# negative and rsqrt(negative)=NaN flows through the whole chain; |x|==1 hits
# 0*inf=NaN and is the only value that needs an explicit guard (exact 0/pi).
_Q0 = tl.constexpr(1.99977445602417)
_Q1 = tl.constexpr(-0.6610356569290161)
_Q2 = tl.constexpr(0.3615966737270355)
_Q3 = tl.constexpr(-0.1717950850725174)
_Q4 = tl.constexpr(0.04228028282523155)
_PI = tl.constexpr(3.141592653589793)

# Failed compiles are not cached, so re-attempting the launch recompiles
# and usually succeeds; once a kernel is compiled the retry is a no-op.
_COMPILE_RETRY_ATTEMPTS = 6
_COMPILE_RETRY_DELAY_S = 1.0


def _launch_retry(launch):
    for attempt in range(_COMPILE_RETRY_ATTEMPTS):
        try:
            return launch()
        except Exception:
            if attempt == _COMPILE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_COMPILE_RETRY_DELAY_S)


@libentry()
@triton.jit
def arccos_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        a = tl.abs(x)
        u = 1.0 - a
        v = 1.0 + a
        t = u * tl.math.rsqrt(u * v)
        t = tl.where(u == 0.0, 0.0, t)
        q = t * t
        p = _Q4
        p = p * q + _Q3
        p = p * q + _Q2
        p = p * q + _Q1
        p = p * q + _Q0
        th = t * p
        xi = x.to(tl.int32, bitcast=True)
        th = tl.where(xi < 0, _PI - th, th)
        tl.store(out_ptr + off, th.to(out_ptr.dtype.element_ty), mask=mask)


def _choose_block(N, is_fp32):
    if is_fp32:
        # fp32 loads stay on v1 vectors: large tiles at num_warps=2 amortize
        # the per-iteration cost best (scan: B65536/w2 at 16.7M-512M). Reverse-
        # derive the tile from grid occupancy so mid sizes keep >=2 programs
        # instead of collapsing onto one fat CTA.
        if N <= 128:
            return 128
        want = triton.next_power_of_2((N + NUM_PROGRAMS - 1) // NUM_PROGRAMS)
        want = max(2048, min(65536, want))
        return min(triton.next_power_of_2(max(N, 1)), want)
    # 16-bit loads convert to v2 (256-lane) compute vectors: the scan puts the
    # sweet spot at BLOCK=4096 with num_warps=4 (B16384+ degrades).
    return max(128, min(4096, triton.next_power_of_2(max(N, 1))))


def _launch(inp, out):
    N = inp.numel()
    if N == 0:
        return out
    is_fp32 = inp.dtype == torch.float32
    BLOCK = _choose_block(N, is_fp32)
    num_warps = 2 if is_fp32 else 4
    grid = min(triton.cdiv(N, BLOCK), NUM_PROGRAMS)
    with torch_device_fn.device(inp.device):
        _launch_retry(
            lambda: arccos_kernel[(grid,)](
                inp,
                out,
                N,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        )
    return out


def arccos(A):
    logger.debug("GEMS_ENFLAME ARCCOS")
    inp = A if A.is_contiguous() else A.contiguous()
    out = torch.empty_like(inp)
    _launch(inp, out)
    return out


def arccos_(x):
    logger.debug("GEMS_ENFLAME ARCCOS_")
    if x.numel() == 0:
        return x
    if not x.is_contiguous():
        # Writing through a contiguous copy would drop the in-place contract
        # (the original x would be left untouched); use the strided kernel.
        _launch_retry(lambda: _strided_arccos_kernel(x, out0=x))
        return x
    _launch(x, x)
    return x
