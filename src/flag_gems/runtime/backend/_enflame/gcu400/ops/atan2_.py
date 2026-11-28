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

"""GCU400 atan2_ (aten::atan2_, in-place) flat fast path.

This shell used to call the stock ``atan2_kernel(A, B, out0=A)`` directly -- generic
pointwise_dynamic codegen lowers atan2 to the ``__ef_atan2f`` extern SIP call on this backend
(~30x the memory floor under a 48-CTA clamped grid: 2^30 fp16 measured 419ms vs native 12.9ms;
after rebuilding the local kernel in 831cb12c8 on 2026-11-13 the core 15-case geomean is 1.133x).
This version recasts the kernel body following the same-family second round of 2026-11-16
(arctan2.py / atan2.py) and re-sweeps the hyperparameters on the new stack (Triton GCU 4.1.6, TopsRider 20261116):

- a K=4 minimax odd polynomial (maxerr 2.3e-5, 1/4 of the fp32 atol 1e-4 gate) replaces K=6,
  evaluated in a dtype-dispatched order (HALF: 16-bit Horner / fp32 Estrin);
- the inf/inf corner guard folds into a single leading ``minimum(t, 1.0)`` (exploiting GCU min
  dropping NaN: the inf/inf ratio NaN collapses to the true ratio 1 -> angle pi/4; finite ratios
  are always <= 1, so the normal path is a no-op) -- an inline select would serialize the software
  pipeline (4096^2 fp32 probe: 0.68x vs 1.11x);
- the negative-y flip becomes a bit-level XOR: ``where(yneg, -a, a)`` loses the -0.0 sign on this
  backend (atan2(-0, x>0)/atan2(y<0, +inf) give +0 instead of aten's -0.0, proven by 169 corner
  probes on 2026-11-18; the old version had the same defect), while XOR keeps the sign bit and drops a cmp/select;
- EVEN predicate-free arm: with N % BLOCK == 0 the ``off < N`` masked load/store is dropped;
- hyperparameter re-sweep (new stack, do_bench median w1000/r100): 16-bit 1G 65536/nw1 7.13ms
  (8192/nw4 11.86ms), 16.7M 32768/nw1 132us (old 192us); fp32 32768/nw2 (1G 6.3-6.6ms, 16.7M
  113us, both better than the old stack's 65536/nw4 sweet spot -- the new stack flips the shape);
  131072 fp32 exceeds the DSM 917504B hard limit and is rejected at compile time.

Guards and fallbacks: a scalar other is wrapped by an fp32 0-dim factory and takes the fast-path
broadcast; inputs beyond the int32 safety domain (numel >= 2^31) or of a non fp16/bf16/fp32 dtype
go to the stock wrapper (semantics unchanged); a non-contiguous input is computed on a contiguous
copy and written back with copy_; other is materialized per the broadcast rules into a contiguous
tensor shaped like input. The in-place aliasing launch is safe (load then store at the same offset
per element). Launches retry 3 times for P02 toolchain cold-compile flakes (sinh/rad2deg precedent).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.atan2 import atan2_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_PROGRAMS = 48

# NOTE: module-level constants are referenced from the `_atan2_vec` helper
# only. This Triton fork's global-drift check deep-copies captured globals
# and compares them with `!=` on every `JITFunction.run`, so a NaN-valued
# constexpr referenced directly in a launched kernel body raises
# "Global variable _NAN has changed" (constexpr[NaN] != constexpr[NaN]) on
# the very first launch. The helper is inlined at compile time and never
# launched on its own, so its captured globals stay out of the check.
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
    # ATen: atan2(input, other) = math atan2(y=input, x=other).
    # Quadrant reduction to [0, pi/2] via min/max, one division, degree-4
    # odd minimax polynomial (max abs error 2.3e-5), quadrant fixups with
    # bit-level sign tests so -0.0 is handled like the ATen op.
    # HALF picks the evaluation order: 16-bit streams are ALU-bound and favor
    # the short Horner chain, fp32 favors Estrin (dependency chain 5 -> 3).
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
    # Sign of y via bit-level XOR: `where(yneg, -a, a)` loses the sign of
    # -0.0 on this backend (-(+0) folds to +0); the XOR keeps atan2(-0, x>0)
    # and atan2(y<0, +inf) at -0.0 exactly like aten.
    a = (a.to(tl.int32, bitcast=True) ^ (yi & -2147483648)).to(tl.float32, bitcast=True)
    # GCU min/max drop NaN instead of propagating it, so a NaN input
    # would come out as a finite ratio; restore the aten NaN result.
    a = tl.where((x != x) | (y != y), _NAN, a)
    return a


@libentry()
@triton.jit(do_not_specialize=["N"])
def _atan2_inplace_kernel(
    X, Y, Out, N, HALF: tl.constexpr, EVEN: tl.constexpr, BLOCK: tl.constexpr
):
    # EVEN is set when N % BLOCK == 0: every tile is full, so the `off < N`
    # compare and the predicated load/store are dropped (~1-2% at 1G on
    # GCU400). Out aliases X (in-place): load and store share offsets.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            y = tl.load(X + off).to(tl.float32)
            x = tl.load(Y + off).to(tl.float32)
        else:
            mask = off < N
            y = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(Y + off, mask=mask, other=0.0).to(tl.float32)
        res = _atan2_vec(x, y, HALF)
        if EVEN:
            tl.store(Out + off, res.to(Out.dtype.element_ty))
        else:
            tl.store(Out + off, res.to(Out.dtype.element_ty), mask=off < N)


def _choose_block(N, itemsize):
    # Per-dtype dispatch re-measured on GCU400 with Triton GCU 4.1.6
    # (do_bench median, w1000/r100):
    #   16-bit  1G:    65536/nw1 7.13ms vs 8192/nw4 11.86ms
    #   16-bit  16.7M: 32768/nw1 132µs vs 8192/nw4 192µs (65536/nw1 135µs)
    #   fp32    1G:    32768/nw2 6.3-6.6ms vs 65536/nw4 6.6-6.8ms
    #   fp32    16.7M: 32768/nw2 113µs vs 65536/nw4 131µs
    # nw2 is pathological on wide 16-bit tiles (1G fp16 10.8ms, 2x worse)
    # and 131072 fp32 exceeds the 917504B DSM limit (compile-time reject).
    # Tiny N uses a padded power-of-two tile: width-1 vectors are
    # pathological on this backend.
    if N <= 16384:
        return max(128, triton.next_power_of_2(max(N, 1))), 4
    if itemsize <= 2:
        return (32768, 1) if N <= 2**24 else (65536, 1)
    return 32768, 2


def _launch_inplace(X, Y, out):
    N = X.numel()
    if N == 0:
        return out
    BLOCK, warps = _choose_block(N, X.element_size())
    grid = min(triton.cdiv(N, BLOCK), NUM_PROGRAMS)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic; observed as converterResult=-2 during host-level
    # userland flake windows); retrying the launch recompiles and succeeds.
    # 3 attempts keep deterministic breakage loud while riding out the flake
    # (sinh.py / rad2deg / atan2_.py precedent).
    for attempt in range(3):
        try:
            with torch_device_fn.device(X.device):
                _atan2_inplace_kernel[(grid,)](
                    X,
                    Y,
                    out,
                    N,
                    HALF=X.element_size() <= 2,
                    BLOCK=BLOCK,
                    EVEN=N % BLOCK == 0,
                    num_warps=warps,
                )
            return out
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "atan2_ flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Flat 1D indexing computes offsets up to ceil(N / BLOCK) * BLOCK - 1; capping
# N at 2**31 - 1 - 65536 keeps the maximum in-tile offset inside int32 for the
# largest BLOCK (65536) used by this kernel (copysign_/relu6_ guard).
_FLAT_MAX_NUMEL = 2**31 - 1 - 65536


def atan2_(input, other):
    logger.debug("GEMS_ENFLAME ATAN2_")
    if not isinstance(other, torch.Tensor):
        # Common pointwise_dynamic has no scalar tier (direct call raises ValueError); wrap the weakly typed
        # scalar in an fp32 0-dim factory per aten semantics and take the fast-path broadcast.
        other = torch.tensor(other, dtype=torch.float32, device=input.device)
    numel = input.numel()
    if numel == 0:
        return input
    if input.dtype not in _FLAT_DTYPES or numel > _FLAT_MAX_NUMEL:
        return atan2_kernel(input, other, out0=input)
    # aten in-place semantics allow other to broadcast to input's shape; materialize it first as a
    # contiguous tensor of the output shape so the flat-index kernel can fetch the matching numerator/denominator in
    # output element order.
    # Same-shape contiguous inputs take the view-free fast path (host overhead matters for launch-bound small shapes).
    if other.shape == input.shape and other.is_contiguous():
        broadcast_other = other
    else:
        broadcast_other = torch.broadcast_to(other, input.shape)
        if not broadcast_other.is_contiguous():
            broadcast_other = broadcast_other.contiguous()
    if not input.is_contiguous():
        # Non-contiguous output view: compute on a contiguous copy and write back, avoiding flat kernel linear
        # writes corrupting the view layout (same contract already delivered for arctan2_).
        contiguous_input = input.contiguous()
        _launch_inplace(contiguous_input, broadcast_other, contiguous_input)
        input.copy_(contiguous_input)
        return input
    return _launch_inplace(input, broadcast_other, input)
