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

NUM_PROGRAMS = 48

# atan(t) ~= t * P(t^2) minimax on [0, 1]; max abs error 2.3e-5 (test gate atol 1e-4).
_P0 = tl.constexpr(0.9999693036)
_P1 = tl.constexpr(-0.3316778839)
_P2 = tl.constexpr(0.1851029694)
_P3 = tl.constexpr(-0.09174843878)
_P4 = tl.constexpr(0.02377510071)
_PI_OVER_2 = tl.constexpr(1.5707963267948966)
_PI = tl.constexpr(3.141592653589793)
_NAN = tl.constexpr(float("nan"))


@triton.jit
def _atan2_vec(x, y, HALF: tl.constexpr):
    # x is the horizontal coordinate, y the vertical one; returns atan2(y, x).
    # The libdevice atan2 lowers to a per-128-lane extern call (tops_ef_vatan2f)
    # on this backend, so the angle is evaluated with plain arithmetic instead:
    # min/max ratio reduction to [0, 1], one odd minimax polynomial
    # (max abs error 2.3e-5, inside the fp32 gate atol 1e-4), quadrant fixups.
    # HALF picks the evaluation order for the degree-4 odd polynomial:
    # 16-bit streams are ALU-bound and favor the short Horner chain, fp32
    # favors Estrin's scheme, which cuts the dependency chain from 5 to 3.
    # Measured per dtype on GCU400 (1G elements):
    #   fp16/bf16 @ (BLOCK=65536, num_warps=1): Horner 6.14ms vs Estrin 6.26ms
    #   fp32     @ (BLOCK=65536, num_warps=4): Estrin 5.72ms vs Horner 7.50ms
    ax = tl.abs(x)
    ay = tl.abs(y)
    mn = tl.minimum(ay, ax)
    mx = tl.maximum(ay, ax)
    # x == 0 (covers the 0/0 pair) divides by 1, keeping t = 0. inf/inf
    # yields a NaN ratio; the trailing minimum maps it back to the true
    # ratio 1 (GCU min drops NaN). For finite pairs the correctly-rounded
    # ratio never exceeds 1, so the minimum is a no-op there.
    t = tl.minimum(mn / tl.where(ax == 0.0, 1.0, mx), 1.0)
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
    a = tl.where(ay > ax, _PI_OVER_2 - a, a)
    xi = x.to(tl.int32, bitcast=True)
    yi = y.to(tl.int32, bitcast=True)
    a = tl.where(xi < 0, _PI - a, a)
    # GCU min/max drop NaN instead of propagating it, so a NaN input
    # would come out as a finite ratio; restore the aten NaN result.
    a = tl.where((x != x) | (y != y), _NAN, a)
    return tl.where(yi < 0, -a, a)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _arctan2_kernel(
    In0, In1, Out, N, HALF: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    # EVEN is set when N % BLOCK == 0: every tile is full, so the `off < N`
    # compare and the predicated load/store are dropped (bit-exact vs the
    # masked path, ~1-2% faster at 1G elements on GCU400).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            y = tl.load(In0 + off).to(tl.float32)
            x = tl.load(In1 + off).to(tl.float32)
        else:
            mask = off < N
            y = tl.load(In0 + off, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(In1 + off, mask=mask, other=0.0).to(tl.float32)
        res = _atan2_vec(x, y, HALF)
        if EVEN:
            tl.store(Out + off, res.to(Out.dtype.element_ty))
        else:
            tl.store(Out + off, res.to(Out.dtype.element_ty), mask=off < N)


def _choose_block(N, itemsize):
    # Per-dtype dispatch, measured on GCU400 at 1G elements (same math
    # chain as the sibling atan2 kernel): 16-bit streams are ALU-bound
    # (BLOCK=2048/nw4: 11.4 ms) and hide the polynomial best behind the
    # widest tile with a single warp (65536/nw1: 6.1 ms); fp32 keeps 4
    # warps. Tiny N uses a padded power-of-two tile: width-1 vectors
    # are pathological on this backend.
    if N <= 16384:
        return max(128, triton.next_power_of_2(max(N, 1))), 4
    if itemsize <= 2:
        return 65536, 1
    return 65536, 4


def _launch(in0, in1, out):
    N = out.numel()
    if N == 0:
        return out
    BLOCK, warps = _choose_block(N, in0.element_size())
    grid = min(triton.cdiv(N, BLOCK), NUM_PROGRAMS)
    with torch_device_fn.device(in0.device):
        _arctan2_kernel[(grid,)](
            in0,
            in1,
            out,
            N,
            HALF=in0.element_size() <= 2,
            BLOCK=BLOCK,
            EVEN=N % BLOCK == 0,
            num_warps=warps,
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


def arctan2(input, other):
    logger.debug("GEMS_ENFLAME ARCTAN2")
    A, B, out_dtype = _prepare(input, other)
    out = torch.empty(A.shape, dtype=out_dtype, device=A.device)
    return _launch(A, B, out)


def arctan2_(input, other):
    """Compute atan2(input, other) in place, writing the result back into input itself per aten semantics.

    Contract: the return value must be the input object; other only supplies the denominator and may broadcast to
    input.shape, but must not change the output shape. The main kernel uses flat indices, so other must
    be materialized as a contiguous tensor of the output shape; non-contiguous input is computed on a contiguous
    copy and written back with copy_, preserving the in-place alias without falling back to common codegen's multi-dim
    index decomposition.
    """
    logger.debug("GEMS_ENFLAME ARCTAN2_")
    # ATen rejects in-place arctan2 on integer and bool inputs (the float result
    # cannot be cast back into the output buffer); mirror that instead of
    # silently truncating.
    if not input.dtype.is_floating_point:
        raise RuntimeError(
            "result type Float can't be cast to the desired output type "
            f"{input.dtype}"
        )
    if input.numel() == 0:
        return input
    # aten in-place semantics allow other to broadcast to input's shape; materialize it first as a contiguous
    # tensor of the output shape so the flat-index kernel can fetch the matching numerator/denominator in output element
    # order.
    broadcast_other = torch.broadcast_to(other, input.shape)
    if not broadcast_other.is_contiguous():
        broadcast_other = broadcast_other.contiguous()
    if not input.is_contiguous():
        # Non-contiguous output view: compute on a contiguous copy and write back, avoiding common codegen's multi-dim
        # index decomposition
        contiguous_input = input.contiguous()
        _launch(contiguous_input, broadcast_other, contiguous_input)
        input.copy_(contiguous_input)
        return input
    return _launch(input, broadcast_other, input)
