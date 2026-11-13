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

# Odd minimax polynomial for atan(t) on t in [0, 1], fitted by least squares:
# atan(t) ~= t * (C0 + C1*t^2 + ... + C6*t^12), max abs error 1.9e-6 (fp32).
_C0 = tl.constexpr(9.999994932e-01)
_C1 = tl.constexpr(-3.332772185e-01)
_C2 = tl.constexpr(1.989734582e-01)
_C3 = tl.constexpr(-1.356217090e-01)
_C4 = tl.constexpr(8.545280521e-02)
_C5 = tl.constexpr(-3.853540616e-02)
_C6 = tl.constexpr(8.408503156e-03)
_PI = tl.constexpr(3.141592653589793)
_PIO2 = tl.constexpr(1.5707963267948966)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _atan2_kernel(X, Y, Out, N, BLOCK: tl.constexpr):
    # ATen: atan2(input, other) = math atan2(y=input, x=other).
    # Quadrant reduction to [0, pi/2] via min/max, one division, odd polynomial
    # in Estrin form (shorter dependency chain than Horner), then sign/quadrant
    # fixes with bit-level sign tests so -0.0 is handled like the ATen op.
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
        t2 = t * t
        t4 = t2 * t2
        t8 = t4 * t4
        q0 = _C0 + t2 * _C1
        q1 = _C2 + t2 * _C3
        q2 = _C4 + t2 * _C5
        r0 = q0 + t4 * q1
        r1 = q2 + t4 * _C6
        p = r0 + t8 * r1
        a = t * p
        a = tl.where(aym > axm, _PIO2 - a, a)
        xneg = xm.to(tl.int32, bitcast=True) < 0
        a = tl.where(xneg, _PI - a, a)
        yneg = ym.to(tl.int32, bitcast=True) < 0
        a = tl.where(yneg, -a, a)
        tl.store(Out + off, a.to(Out.dtype.element_ty), mask=mask)


def _choose_block(N, itemsize):
    # 按 dtype 分档：半精度保持 8192、fp32 放大到 65536，
    # 宽 tile 只缩短 fp32 的 grid-stride 循环，不影响半精度既有最优宽度。
    if N <= 16384:
        return triton.next_power_of_2(max(N, 1))
    return 8192 if itemsize <= 2 else 65536


def _launch_atan2(X, Y, out):
    N = X.numel()
    if N == 0:
        return out
    BLOCK = _choose_block(N, X.element_size())
    grid = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(X.device):
        _atan2_kernel[(grid,)](X, Y, out, N, BLOCK=BLOCK, num_warps=4)
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
    A, B, _ = _prepare(input, other)
    return _launch_atan2(A, B, out)
