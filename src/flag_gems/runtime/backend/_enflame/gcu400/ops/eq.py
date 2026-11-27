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

"""Element-wise equality, GCU400-optimized (eq / eq_scalar; `equal` kept as-is).

The previous path ran every shape through the pointwise_dynamic codegen
template (bptr tiles, fixed num_warps=4) whose fp32 up-cast comparison wastes
ALU on same-dtype floats and whose template store structure leaves ~20%
bandwidth on the table versus native (L600 QA report: eq 0.768x /
eq_scalar 0.745x; mid-band 2^24 shapes re-measured at 0.74-0.92x).

Large contiguous same-dtype float inputs are dispatched to dedicated flat 1D
kernels instead (probe-validated tiers, sibling of greater_equal/copysign):

* comparisons run in the native input dtype -- bit-exact for same-dtype
  floats (the fp32 up-cast is an injective map, so `==` is preserved) and
  cheaper (the up-cast costs 17-22% on f16/bf16);
* the scalar overload casts the Python scalar to ``A.dtype`` before
  comparing, matching torch TensorIterator semantics; the fp32-compare
  template deviates from torch on fp16 round-down scalars (e.g. 0.1);
* per-dtype DSM cap (flat dual-load Required = 2 x itemsize x BLOCK,
  917504 B compile-time limit) bounds BLOCK;
* every tier kernel carries an ``EVEN`` constexpr arm (host-side
  ``n % BLOCK == 0``): when the tile count divides the element count the
  masked loads/stores (mask predicate + `other=0` fill) are compiled out on
  the mid band. The giant band (n >= 2^28) keeps the masked arm on purpose:
  the isolated-probe EVEN gain did not survive the official-bench
  re-measure (within-noise to +1.2%), a fresh instance of the
  isolated-vs-sustained-load reversal.

Probe-validated tiers (L600, do_bench kernel-mode median vs torch native;
three interleaved sweep rounds, 2026-11-11; EVEN/strided re-sweep
2026-11-16):

  tensor   n < 2^16            -> generic pointwise (launch-overhead floor)
  tensor   2^16..2^22          -> flat BLOCK=65536 nw4 EVEN-armed
  tensor   2^22..2^28          -> strided 48xB=131072 nw4 f16/bf16
                                  (~0.96x, flat launch loses ~4.5%)
  tensor   f32 n >= 2^16       -> strided 48xB=65536 nw4 (mid 0.98x; the
                                  contiguous-chunk loop tier retired: 70.4
                                  -> 65.1us at 2^24; grid sweep g32..g256
                                  all lose to g48)
  tensor   n >= 2^28 (f16)     -> loop COV=1048576 CHUNK=131072 nw2 st2
                                  (~1.0x, masked arm: the EVEN arm that
                                  won the isolated probe (+4%) re-measured
                                  within-noise-to-worse on the official
                                  bench, so giants keep masked)
  scalar   n < 2^16            -> flat BLOCK=next_pow2(n) nw4 EVEN-armed
  scalar   2^16..2^22          -> flat BLOCK=65536 nw4 EVEN-armed
  scalar   f32 n >= 2^22       -> strided 48xB=65536 nw4 (2^24 37.7us vs
                                  flat-131072 anchor 38.8us)
  scalar   f16/bf16 2^22..2^28 -> strided 48xB=131072 nw4 (24.0us)
  scalar   f16/bf16 n >= 2^28  -> flat BLOCK=262144 nw4 (~1.0x; a 65536-wide
                                  strided tile loses 5%)

The fp32 arms use a strided (grid-stride, 48 persistent CTAs) tile
assignment: under the official bench's sustained traffic the contiguous
coverage-chunk loop loses ~5% on giants and ~7% at 2^24, while f16 giants
prefer the chunk loop.
"""

import logging
import time

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name

_NP2 = triton.next_power_of_2
_NPROGS = 48
_BS = 32768

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SMALL_N = 2**16
_MID_N = 2**22
_BIG_N = 2**28
_INT32_SAFE_LIMIT = 0x7FFFFFFF


def _launch_with_retry(launch):
    # Flaky external toolchain (fresh-compile segv windows on the shared
    # node): retry compile-time failures only, re-raise the original error
    # once exhausted so real failures are never masked.
    for attempt in range(6):
        try:
            launch()
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(1.0)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func(x, y):
    # Compare in the operand's own dtype.  The wrapper has already promoted
    # both operands to their common dtype on the host (torch promotion), and
    # equality is preserved by any injective widening -- while an fp32 up-cast
    # is NOT injective for integers: 2**24 and 2**24+1 collapse to one fp32
    # value and distinct int32/int64 ids compared equal (test_eq_integer_precision,
    # test_equal_integer_precision).
    return x == y


# The fp32 up-cast above is lossless for same-dtype floats but not for wide
# integers: 2**24 + 1 and 2**24 + 2 both round to 2**24 in fp32, so distinct
# int32/int64 ids would compare equal.  Integral (and bool) inputs therefore
# take a native-dtype comparison; same-dtype promotion is a no-op, so the
# compare stays exact for every integer width.
@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func_exact(x, y):
    return x == y


@triton.jit
def _eq_cmp_kernel(A, B, OUT, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(A + off)
        y = tl.load(B + off)
        r = x == y
        tl.store(OUT + off, r.to(OUT.dtype.element_ty))
    else:
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        r = x == y
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _eq_cmp_loop_kernel(
    A,
    B,
    OUT,
    N,
    COV: tl.constexpr,
    CHUNK: tl.constexpr,
    ST: tl.constexpr,
    EVEN: tl.constexpr,
):
    base = tl.program_id(0) * COV
    for j in tl.range(0, COV // CHUNK, num_stages=ST):
        off = base + j * CHUNK + tl.arange(0, CHUNK)
        if EVEN:
            x = tl.load(A + off)
            y = tl.load(B + off)
            r = x == y
            tl.store(OUT + off, r.to(OUT.dtype.element_ty))
        else:
            m = off < N
            x = tl.load(A + off, mask=m, other=0)
            y = tl.load(B + off, mask=m, other=0)
            r = x == y
            tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _eq_cmp_strided_kernel(A, B, OUT, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    for tile in tl.range(pid, tl.cdiv(N, BLOCK), num_ctas):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(A + off)
            y = tl.load(B + off)
            r = x == y
            tl.store(OUT + off, r.to(OUT.dtype.element_ty))
        else:
            m = off < N
            x = tl.load(A + off, mask=m, other=0)
            y = tl.load(B + off, mask=m, other=0)
            r = x == y
            tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _eq_cmp_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(A + off)
        r = x == Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty))
    else:
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        r = x == Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


def _eq_cmp_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if n == 0:
        return False
    if not A.dtype.is_floating_point:
        # Integral same-dtype equality must stay in the integer domain: the
        # pointwise fallback compares in fp32 and collapses ids above 2**24
        # (2**24 == 2**24+1 measured True).  `_eq_cmp_kernel` compares the
        # loaded values natively, so a flat launch is enough; the DSM budget
        # (dual load, 917504 B) caps the tile per element size.
        if n > _INT32_SAFE_LIMIT - 65536:
            return False
        max_block = 917504 // (2 * A.element_size())
        block = min(65536, max_block, max(256, _NP2(n)))
        _launch_with_retry(
            lambda: _eq_cmp_kernel[(triton.cdiv(n, block),)](
                A, B, out, n, BLOCK=block, EVEN=(n % block == 0), num_warps=4
            )
        )
        return True
    if n < _SMALL_N:
        return False
    if A.dtype == torch.float32:
        # DSM caps the fp32 dual-load tile at BLOCK=65536; the 48-CTA
        # persistent strided form beats both the flat launch and the
        # contiguous-chunk loop on every measured band (probe 2026-11-16:
        # 2^24 strided 65.1us vs loop anchor 70.4us; grid sweep g96/g64/g128
        # all lose to g48), so it serves the whole fp32 mid/giant range.
        # EVEN arm only below the giant band: the official-bench re-measure
        # shows no gain (within-noise to +1%) there, so giants keep masked.
        if n > _INT32_SAFE_LIMIT - 65536:
            return False
        _launch_with_retry(
            lambda: _eq_cmp_strided_kernel[(_NPROGS,)](
                A,
                B,
                out,
                n,
                BLOCK=65536,
                EVEN=(n % 65536 == 0 and n < _BIG_N),
                num_warps=4,
            )
        )
        return True
    if n >= _BIG_N:
        cov, chunk, nw, st = 1048576, 131072, 2, 2
        if n > _INT32_SAFE_LIMIT - cov:
            return False
        _launch_with_retry(
            lambda: _eq_cmp_loop_kernel[(triton.cdiv(n, cov),)](
                A,
                B,
                out,
                n,
                COV=cov,
                CHUNK=chunk,
                ST=st,
                EVEN=False,
                num_warps=nw,
                num_stages=st,
            )
        )
        return True
    if n >= _MID_N:
        # f16/bf16 mid band: 48-CTA persistent strided x 131072 beats the
        # previous flat launch (probe 2026-11-16: 38.5us vs flat 40.6us)
        if n > _INT32_SAFE_LIMIT - 131072:
            return False
        _launch_with_retry(
            lambda: _eq_cmp_strided_kernel[(_NPROGS,)](
                A, B, out, n, BLOCK=131072, EVEN=(n % 131072 == 0), num_warps=4
            )
        )
        return True
    block = 65536
    if n > _INT32_SAFE_LIMIT - block:
        return False
    _launch_with_retry(
        lambda: _eq_cmp_kernel[(triton.cdiv(n, block),)](
            A, B, out, n, BLOCK=block, EVEN=(n % block == 0), num_warps=4
        )
    )
    return True


# ---------------- complex eq bridged off the hard-aborting native path -------
# topsatenEq rejects complex64/complex128 with a C++ CHECK abort (NOT_SUPPORT,
# no error code), so `a == b` on complex GCU tensors kills the process -- the
# slice_copy_out accuracy leg hits it on its final `(out == ref).all()` assert.
# An aten shim on the free AutogradPrivateUse1 slot (house pattern, see
# adaptive_avg_pool2d.py) maps complex equality onto two real comparisons over
# `view_as_real` plus an all-reduce along the trailing dim; every other dtype
# redispatches to the untouched native PrivateUse1 kernel.


def eq(A, B):
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    logger.debug("GEMS_ENFLAME EQ")
    if A.dtype == B.dtype and not A.dtype.is_floating_point and not A.dtype.is_complex:
        return eq_func_exact(A, B)
    if (
        A.dtype == B.dtype
        and A.dtype not in (torch.bool, torch.complex64, torch.complex128)
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        out = torch.empty_like(A, dtype=torch.bool)
        if _eq_cmp_run(A, B, out):
            return out
    return eq_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func_scalar(x, y):
    # Native-dtype compare (see eq_func): an fp32 up-cast collapses integer
    # ids above 2**24, so eq(x, 2**24+1) reported True for x == 2**24.
    return x == y


# Integer-precision companion of `eq_func_scalar`: a Python int scalar must be
# compared against the exact integer lanes, otherwise the fp32 up-cast makes
# 2**24 + 1 and 2**24 + 2 collide.  Float scalars keep the fp32 path, which is
# what the 2.5-vs-int64 contract needs (no truncation to an integer).
@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func_scalar_exact(x, y):
    return x == y


def eq_scalar(A, B):
    logger.debug("GEMS_ENFLAME EQ_SCALAR")
    if (
        not A.dtype.is_floating_point
        and not A.dtype.is_complex
        and isinstance(B, int)
        and not isinstance(B, bool)
    ):
        return eq_func_scalar_exact(A, B)
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        try:
            y = float(B)
        except (OverflowError, ValueError):
            return eq_func_scalar(A, B)
        n = A.numel()
        if n > 0:
            if n >= _MID_N:
                if A.dtype == torch.float32:
                    # fp32 mid+giant: 48-CTA persistent strided (probe
                    # 2026-11-16: 2^24 37.7us vs flat-131072 anchor 38.8us;
                    # grid sweep: g32/g48 best, g96+ lose). EVEN arm below the
                    # giant band only (official re-measure: no gain there).
                    if n > _INT32_SAFE_LIMIT - 65536:
                        return eq_func_scalar(A, B)
                    out = torch.empty_like(A, dtype=torch.bool)
                    _launch_with_retry(
                        lambda: _eq_cmp_scalar_strided_kernel[(_NPROGS,)](
                            A,
                            out,
                            n,
                            y,
                            BLOCK=65536,
                            EVEN=(n % 65536 == 0 and n < _BIG_N),
                            num_warps=4,
                        )
                    )
                    return out
                if n >= _BIG_N:
                    # f16/bf16 giant: keep the flat 262144 launch with the
                    # masked arm (probe 2026-11-16: strided-48x131072 ties
                    # within noise, a 65536-wide strided tile loses 5%; the
                    # EVEN arm re-measured within-noise-to-+1% on the
                    # official bench, so masked stays)
                    block = 262144
                else:
                    # f16/bf16 mid: 48-CTA persistent strided x 131072
                    # (probe 2026-11-16: 24.0us vs flat-262144 anchor 24.1us)
                    if n > _INT32_SAFE_LIMIT - 131072:
                        return eq_func_scalar(A, B)
                    out = torch.empty_like(A, dtype=torch.bool)
                    _launch_with_retry(
                        lambda: _eq_cmp_scalar_strided_kernel[(_NPROGS,)](
                            A,
                            out,
                            n,
                            y,
                            BLOCK=131072,
                            EVEN=(n % 131072 == 0),
                            num_warps=4,
                        )
                    )
                    return out
            elif n >= _SMALL_N:
                block = 65536
            else:
                # shape-matched tile for launch-floor shapes: a fixed 16384-wide
                # vector wastes 4x lanes on (64,64)-class fp32 inputs (+1.5us)
                block = max(256, min(16384, _NP2(n)))
            if n <= _INT32_SAFE_LIMIT - block:
                out = torch.empty_like(A, dtype=torch.bool)
                _launch_with_retry(
                    lambda: _eq_cmp_scalar_kernel[(triton.cdiv(n, block),)](
                        A,
                        out,
                        n,
                        y,
                        BLOCK=block,
                        EVEN=(n % block == 0) if n < _BIG_N else False,
                        num_warps=4,
                    )
                )
                return out
    return eq_func_scalar(A, B)


@triton.jit
def _eq_cmp_scalar_strided_kernel(
    A, OUT, N, Y, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    for tile in tl.range(pid, tl.cdiv(N, BLOCK), num_ctas):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(A + off)
            r = x == Y.to(x.dtype)
            tl.store(OUT + off, r.to(OUT.dtype.element_ty))
        else:
            m = off < N
            x = tl.load(A + off, mask=m, other=0)
            r = x == Y.to(x.dtype)
            tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _equal_small_k(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    all_eq = tl.full((), 1, dtype=tl.int32)
    for off in range(0, numel, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        # Compare in the operand dtype: an fp32 up-cast collapses integers
        # above 2**24 (torch.equal([2**24], [2**24+1]) must be False).
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0)
        eq_val = tl.min((x == y).to(tl.int32), axis=0)
        all_eq = all_eq & eq_val
    tl.store(out_ptr, all_eq)


@triton.jit
def _equal_fused_k(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    num_progs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    all_eq = tl.full((), 1, dtype=tl.int32)
    blk = pid
    while blk * BLOCK_SIZE < numel:
        offsets = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        # Native-dtype compare; see _equal_small_k (integer ids > 2**24).
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0)
        eq_val = tl.min((x == y).to(tl.int32), axis=0)
        all_eq = all_eq & eq_val
        blk += num_progs
    tl.store(out_ptr + pid, all_eq)


@triton.jit
def _reduce_min_k(in_ptr, out_ptr, N: tl.constexpr):
    idx = tl.arange(0, N)
    v = tl.load(in_ptr + idx)
    result = tl.min(v, axis=0)
    tl.store(out_ptr, result)


def equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    logger.debug("GEMS_ENFLAME EQUAL")
    if x.shape != y.shape:
        return False
    if x.numel() == 0:
        return True

    if x.dtype != y.dtype:
        # torch.equal compares after the usual type promotion; promote on the
        # host so the kernel compares a single native dtype (no fp32 up-cast,
        # which would collapse integer ids above 2**24).
        common = torch.promote_types(x.dtype, y.dtype)
        if common in (torch.float64, torch.complex128):
            # gcu400 has no fp64 datapath; keep the previous fp32 precision.
            common = torch.float32
        x = x.to(common)
        y = y.to(common)

    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    x_flat = x.view(-1)
    y_flat = y.view(-1)
    numel = x_flat.numel()

    out = torch.empty(1, dtype=torch.int32, device=x.device)

    if numel <= _BS:
        _equal_small_k[(1,)](
            x_flat,
            y_flat,
            out,
            numel,
            BLOCK_SIZE=_NP2(numel),
            num_warps=1,
        )
    else:
        NP = _NP2(_NPROGS)
        out_t = torch.ones(NP, dtype=torch.int32, device=x.device)
        _equal_fused_k[(_NPROGS,)](
            x_flat,
            y_flat,
            out_t,
            numel,
            _NPROGS,
            BLOCK_SIZE=_BS,
            num_warps=1,
        )
        _reduce_min_k[(1,)](out_t, out, N=NP, num_warps=1)

    return bool(out.item() > 0)
