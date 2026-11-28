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

# Odd minimax polynomial for atan(t) on t in [0, 1]:
# atan(t) ~= t * (P0 + P1*t^2 + P2*t^4 + P3*t^6 + P4*t^8), max abs error
# 2.3e-5 — 4x inside the fp32 gate (atol 1e-4) and invisible to 16-bit
# outputs (fp16 rtol 1e-3 / bf16 rtol 0.016).
_P0 = tl.constexpr(0.9999693036)
_P1 = tl.constexpr(-0.3316778839)
_P2 = tl.constexpr(0.1851029694)
_P3 = tl.constexpr(-0.09174843878)
_P4 = tl.constexpr(0.02377510071)
_PI = tl.constexpr(3.141592653589793)
_PIO2 = tl.constexpr(1.5707963267948966)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _atan2_kernel(X, Y, Out, N, HALF: tl.constexpr, BLOCK: tl.constexpr):
    # ATen: atan2(input, other) = math atan2(y=input, x=other).
    # Quadrant reduction to [0, pi/2] via min/max, one division, odd minimax
    # polynomial, then sign/quadrant fixes with bit-level sign tests so
    # -0.0 is handled like the ATen op.
    # HALF picks the polynomial evaluation order for 16-bit streams:
    # both dtypes are compute-bound at 1G elements (fp16 ran at 658 GB/s
    # against a 2.9 TB/s copy floor), so the degree-6 polynomial was cut to
    # degree 4. Measured per dtype on GCU400 (1G elements):
    #   fp16/bf16 @ (BLOCK=65536, num_warps=1): Horner 6.14ms vs Estrin 6.26ms
    #   fp32     @ (BLOCK=65536, num_warps=4): Estrin 5.72ms vs Horner 7.50ms
    # (16-bit big tiles: 9.79ms -> 6.14ms, -37%; fp32: 6.29ms -> 5.72ms, -9%.)
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        ym = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        xm = tl.load(Y + off, mask=mask, other=0.0).to(tl.float32)
        aym = tl.abs(ym)
        axm = tl.abs(xm)
        mn = tl.minimum(aym, axm)
        mx = tl.maximum(aym, axm)
        t = mn / tl.where(axm == 0.0, 1.0, mx)
        if HALF:
            s = t * t
            p = _P4
            p = p * s + _P3
            p = p * s + _P2
            p = p * s + _P1
            p = p * s + _P0
        else:
            s = t * t
            s2 = s * s
            e0 = _P0 + s * _P1
            e1 = _P2 + s * _P3
            p = e0 + s2 * (e1 + s2 * _P4)
        a = t * p
        a = tl.where(aym > axm, _PIO2 - a, a)
        xneg = xm.to(tl.int32, bitcast=True) < 0
        a = tl.where(xneg, _PI - a, a)
        yneg = ym.to(tl.int32, bitcast=True) < 0
        a = tl.where(yneg, -a, a)
        tl.store(Out + off, a.to(Out.dtype.element_ty), mask=mask)


def _choose_block(N, itemsize):
    # Tier by dtype: half-precision large shapes use a 65536-wide tile + single warp (GCU400 measured
    # 1G fp16 9.79ms→6.16ms; a compute-chain-hiding case, mirroring gelu 16-bit's 65536/nw1
    # precedent); fp32 keeps 65536/nw4; small shapes with N≤16384 keep the next_pow2 width.
    # Return (BLOCK, num_warps).
    if N <= 16384:
        return triton.next_power_of_2(max(N, 1)), 4
    if itemsize <= 2:
        return 65536, 1
    return 65536, 4


def _launch_atan2(X, Y, out):
    N = X.numel()
    if N == 0:
        return out
    BLOCK, warps = _choose_block(N, X.element_size())
    grid = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(X.device):
        _atan2_kernel[(grid,)](
            X, Y, out, N, HALF=X.element_size() <= 2, BLOCK=BLOCK, num_warps=warps
        )
    return out


def _prepare(A, B):
    if A.shape != B.shape or A.dtype != B.dtype:
        A, B = torch.broadcast_tensors(A, B)
    A = A if A.is_contiguous() else A.contiguous()
    B = B if B.is_contiguous() else B.contiguous()
    out_dtype = torch.promote_types(A.dtype, B.dtype)
    if not out_dtype.is_floating_point:
        out_dtype = torch.float32
    return A, B, out_dtype


def atan2(input, other):
    logger.debug("GEMS_ENFLAME ATAN2")
    A, B, out_dtype = _prepare(input, other)
    out = torch.empty(A.shape, dtype=out_dtype, device=A.device)
    return _launch_atan2(A, B, out)


def atan2_out(input, other, out):
    logger.debug("GEMS_ENFLAME ATAN2_OUT")
    # ATen rejects an integer or bool out buffer (the float result cannot be
    # cast into it); mirror that instead of silently truncating.
    if not out.dtype.is_floating_point:
        raise RuntimeError(
            "result type Float can't be cast to the desired output type " f"{out.dtype}"
        )
    A, B, _ = _prepare(input, other)
    return _launch_atan2(A, B, out)
