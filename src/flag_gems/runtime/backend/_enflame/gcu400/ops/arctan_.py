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
FLAT_GRID = NUM_SIPS * 2

# Odd minimax polynomial for atan(t) on t in [0, 1]:
# atan(t) ~= t * (P0 + P1*t^2 + P2*t^4 + P3*t^6 + P4*t^8), max abs error
# 2.3e-5 (4x inside the fp32 gate atol 1e-4; invisible to 16-bit outputs).
# Coefficients are shared verbatim with the atan2/atan ops in this directory.
_P0 = tl.constexpr(0.9999693036)
_P1 = tl.constexpr(-0.3316778839)
_P2 = tl.constexpr(0.1851029694)
_P3 = tl.constexpr(-0.09174843878)
_P4 = tl.constexpr(0.02377510071)
_PIO2 = tl.constexpr(1.5707963267948966)


@libentry()
@triton.jit(do_not_specialize=["N"])
def arctan_kernel(
    x_ptr, out_ptr, N, SEL16: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    # atan(x) = sign(x) * (|x|<=1 ? t*p(t^2) : pi/2 - t*p(t^2)) with the
    # range-reduced t = min(|x|, 1/|x|) in [0, 1]. The libdevice extern call
    # (tl_extra_shim.atan, per-128-lane software path; family-measured on the
    # sibling extern atanh: 11674 vs 2419 us at 512Mi fp32) is replaced by the
    # same minimax polynomial already gated by the atan2/atan tests.
    # Estrin's scheme is used for ALL dtypes (family precedent atan: 16-bit
    # Estrin beats Horner by 10%, unlike atan2's 16-bit Horner result --
    # evaluation order does not transfer across sibling ops).
    # The reduced argument is built with tl.where(big, r2, ax) instead of
    # tl.minimum: GCU min/max drop NaN (family precedent), which would turn a
    # NaN input into the spurious finite ratio; where() keeps the NaN alive
    # through the polynomial, matching aten NaN semantics with no dedicated
    # fixup select. The 1/|x| on the big arm uses the hardware rsqrt identity
    # rsqrt(ax)^2 (family precedent arctanh_: rsqrt is a ~free vector
    # primitive) instead of a division: in this exact 1L1S EVEN form the real
    # division costs 862 us at 512Mi fp32 / ~700 us on 16-bit (A/B measured),
    # far above the 241 us the arctanh_ note books inside its log formula.
    # No clamp before rsqrt is needed: on the small arm the where() discards
    # the rsqrt result (rsqrt(+0) = inf, silently dropped), so NaN/inf
    # semantics stay intact without a max() op.
    # SEL16: for 16-bit outputs the sign select runs in the 16-bit domain
    # (magnitude is cast first), keeping two f32 live vectors off the store
    # stage (family precedent atan/arcsin: -31% on the select bucket).
    # EVEN is set when N % BLOCK == 0: every tile is full, so the `off < N`
    # compare and the predicated load/store are dropped (family precedent
    # arctan2: bit-exact vs the masked path, ~1-2% faster at 512Mi).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + off).to(tl.float32)
        else:
            mask = off < N
            x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        ax = tl.abs(x)
        big = ax > 1.0
        r = tl.rsqrt(ax)
        r2 = r * r
        t = tl.where(big, r2, ax)
        s = t * t
        s2 = s * s
        e0 = _P0 + s * _P1
        e1 = _P2 + s * _P3
        p = e0 + s2 * (e1 + s2 * _P4)
        tp = t * p
        a = tl.where(big, _PIO2 - tp, tp)
        neg = x.to(tl.int32, bitcast=True) < 0
        if SEL16:
            a16 = a.to(out_ptr.dtype.element_ty)
            y = tl.where(neg, -a16, a16)
        else:
            y = tl.where(neg, -a, a).to(out_ptr.dtype.element_ty)
        if EVEN:
            tl.store(out_ptr + off, y)
        else:
            tl.store(out_ptr + off, y, mask=off < N)


def _choose_block(N, itemsize):
    # Size bands measured on GCU400 (official bench kernel mode, do_bench
    # w1000/r100 median; each config measured in its own single-config
    # process -- the multi-config in-process sweep reads 20-30% high on this
    # backend and must not be used for arbitration):
    #   tiny (N <= 32768): next_pow2(N/8) padded to 512, nw2
    #     (4096 elems: 3.6 us; beats 512/nw4 3.8 and 1024/nw4 4.1)
    #   16-bit mid (N <= 2^24): 32768/nw2 (16.7M: 108.9 us, beats 16384/nw4
    #     116.2 and 65536/nw2 113.5)
    #   everything larger: 65536/nw2 (2^25..2^29 monotone best for both
    #     16-bit and fp32: 512Mi fp16 3019 / fp32 2989 us; B131072/nw2 is
    #     within 1.2% at 512Mi but 8% worse at 2^25)
    if N <= 32768:
        return max(triton.next_power_of_2(max(N // 8, 1)), 512), 2
    if itemsize <= 2 and N <= (1 << 24):
        return 32768, 2
    return 65536, 2


def _launch(inp, out):
    N = inp.numel()
    if N == 0:
        return out
    BLOCK, num_warps = _choose_block(N, inp.element_size())
    grid_size = min(triton.cdiv(N, BLOCK), FLAT_GRID)
    with torch_device_fn.device(inp.device):
        arctan_kernel[(grid_size,)](
            inp,
            out,
            N,
            SEL16=inp.element_size() <= 2,
            BLOCK=BLOCK,
            EVEN=N % BLOCK == 0,
            num_warps=num_warps,
        )
    return out


def _arctan(A):
    inp = A if A.is_contiguous() else A.contiguous()
    return _launch(inp, torch.empty_like(inp))


def arctan(A):
    logger.debug("GEMS_ENFLAME ARCTAN")
    return _arctan(A)


def arctan_(A):
    logger.debug("GEMS_ENFLAME ARCTAN_")
    if not A.is_contiguous():
        A.copy_(_arctan(A))
        return A
    return _launch(A, A)
