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

# int32 offset safe domain: the masked tail block evaluates
# block_id * BLOCK + arange for masked lanes as well, reaching up to
# num_blocks * BLOCK - 1; keep n <= 2^31 - 2^17 (BLOCK <= 131072)
# so flat int32 offsets cannot wrap negative on the tail block.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


# GCU400 retune (2026-11-16, interleaved do_bench(median) A/B, w1000/r100):
# the old f32-promote body (x.to(f32) + floor/ceil/where + implicit cvt back)
# with the old ladder (16-bit fixed B4096/nw4, fp32 B65536/nw4) measured
# 0.745x (16-bit 512Mi), 0.83x (16-bit 16.7M) and 1.00x (fp32) vs native.
# The new body reuses the trunc_ family's int-domain IEEE-exact trunc (native
# fp16/bf16 bit pattern in int16, fp32 in int32; no promote round-trip, no cvt
# pair, no compare/select chain) plus one native-domain fsub x - t, which is
# exact by Sterbenz for every representable input (the fraction has fewer
# significant bits than x), and keeps aten trunc/frac edge semantics: ±0.0 ->
# +0.0, |x| < 1 keeps the value, inf/NaN propagate through the fsub.
# Probe numbers (same-session interleaved, jobs/j20261116221938_312d760e):
#   16-bit 512Mi 673.8-699.3us (copy floor 695.3, old 3324.5);
#   16-bit 16.7M 27.0us (floor 25.8, old 109.6);
#   fp32 512Mi 1370.1-1375.0us (floor 1390.2, old 2187.0);
#   fp32 16.7M 49.1-49.3us (floor 48.2, old 83.6);
#   4K tiny 3.2us (old 4.0).
# Two Triton typing traps are inherited from the trunc_ body: min/max or
# where with a Python scalar on an int16 tensor silently promotes the result
# to int32 (which would make the bitcast back to fp16 illegal), so the clamp
# bounds are tensor-ized (zero = u ^ u).
@triton.jit
def _frac_trunc_scalar(x0, IS_FP16: tl.constexpr, IS_BF16: tl.constexpr):
    if IS_FP16 or IS_BF16:
        u = x0.to(tl.int16, bitcast=True)
        zero = u ^ u
        if IS_BF16:
            # bf16: 8 exponent bits at offset 7, 7 mantissa bits.  Bits to
            # drop: max(127 + 7 - e, 0); |x| < 1 (s >= 8) clears all but the
            # sign bit; e >= 134 (|x| >= 2^7) and inf/NaN keep the input.
            e = (u >> 7) & 0xFF
            s = tl.maximum(134 - e, zero)
            s = tl.where(s >= zero + 8, zero + 15, s)
        else:
            # fp16: 5 exponent bits at offset 10, 10 mantissa bits.  Bits to
            # drop: max(15 + 10 - e, 0); |x| < 1 (s >= 11) clears all but the
            # sign bit; e >= 25 (|x| >= 2^10) and inf/NaN keep the input.
            e = (u >> 10) & 0x1F
            s = tl.maximum(25 - e, zero)
            s = tl.where(s >= zero + 11, zero + 15, s)
        return ((u >> s) << s).to(x0.dtype, bitcast=True)
    else:
        u = x0.to(tl.int32, bitcast=True)
        e = (u >> 23) & 0xFF
        # mantissa bits below 2^0 for 1 <= |x| < 2^23; all but the sign bit
        # for |x| < 1 (s >= 24); nothing for |x| >= 2^23 / inf / nan.
        s = tl.maximum(150 - e, 0)
        s = tl.where(s >= 24, 31, s)
        keep = -1 << s
        return (u & keep).to(tl.float32, bitcast=True)


@libentry()
@triton.jit(do_not_specialize=["N"])
def frac_kernel(
    X,
    Out,
    N,
    BLOCK: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x0 = tl.load(X + off)
            t = _frac_trunc_scalar(x0, IS_FP16, IS_BF16)
            tl.store(Out + off, x0 - t)
        else:
            m = off < N
            x0 = tl.load(X + off, mask=m)
            t = _frac_trunc_scalar(x0, IS_FP16, IS_BF16)
            tl.store(Out + off, x0 - t, mask=m)


def _choose_launch(n, dtype):
    # Interleaved A/B probe on the 1R1W flat band ladder (2026-11-16):
    #   tiny wants the exact next_pow2 tile (3.2us @4K);
    #   16-bit mid (2^24) peaks at 131072/nw4 (27.0 vs 28.6-30.2);
    #   16-bit big (2^29) ties 131072/nw4 with 65536/nw2 within window noise
    #     (676-704 vs 683-706) and matches the trunc_ sibling table;
    #   fp32 sits at the DRAM floor with 32768/nw2 in both mid (49.2) and
    #     big (1374) bands.
    if n <= 8192:
        return max(triton.next_power_of_2(n), 256), 2
    if n <= 65536:
        return 8192, 2
    if n <= (1 << 21) or dtype == torch.float32:
        return 32768, 2
    return 131072, 4


def _launch(inp, out):
    n = inp.numel()
    if n == 0:
        return out
    block, num_warps = _choose_launch(n, inp.dtype)
    grid_size = min(triton.cdiv(n, block), NUM_SIPS * 2)
    is_fp16 = inp.dtype == torch.float16
    is_bf16 = inp.dtype == torch.bfloat16
    even = n % block == 0
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds (abs / trunc_ /
    # neg precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                frac_kernel[(grid_size,)](
                    inp,
                    out,
                    n,
                    BLOCK=block,
                    IS_FP16=is_fp16,
                    IS_BF16=is_bf16,
                    EVEN=even,
                    num_warps=num_warps,
                )
            return out
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "frac flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _flat_ok(A):
    return (
        A.is_contiguous() and A.dtype in _FLAT_DTYPES and A.numel() < _INT32_SAFE_LIMIT
    )


def frac(A):
    logger.debug("GEMS_ENFLAME FRAC")
    if not _flat_ok(A):
        # Single strided kernel instead of a contiguous() copy plus a
        # second pass over the data (abs / neg precedents); also the only
        # correct route for dtypes outside the flat domain (e.g. fp64).
        from flag_gems.ops.frac_ import frac as _generic_frac

        return _generic_frac(A)
    out = torch.empty_like(A)
    return _launch(A, out)


def frac_(A):
    logger.debug("GEMS_ENFLAME FRAC_")
    if not _flat_ok(A):
        from flag_gems.ops.frac_ import frac_ as _generic_frac_

        return _generic_frac_(A)
    return _launch(A, A)
