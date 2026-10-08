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
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

MIN_BLOCK = 2048
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False

# Magic round-to-nearest-int constants for the fully-vectorizable corner logic.
# C = 1.5 * 2**23 forces round-to-nearest via (v + C) - C (exact for |v| < 2**22);
# BIG = 2**23 is the threshold at/above which every fp32 is an even integer.
_MAGIC_C = tl.constexpr(12582912.0)
_MAGIC_BIG = tl.constexpr(8388608.0)


@triton.jit
def _float_power_corner(x, e):
    # x ** e with ATen/C++ pow corner semantics, using ONLY ops that have an
    # XPU vector form (abs/log/mul/exp/cmpf/addf/subf/divf/select) -- NO exp2/
    # log2, NO int bitcast, NO cmpi, NO i1 AND/OR. This lets the whole store
    # cone vectorize (16-lane SIMD) instead of reverting to a scalar loop.
    ax = tl.abs(x)
    r = tl.exp(tl.log(ax) * e)
    # (|x| == 1) | (e == 0) -> 1  (incl. 1^(+-inf/NaN), 0^0, NaN^0), nested where.
    r = tl.where(ax == 1.0, 1.0, tl.where(e == 0.0, 1.0, r))
    # integer / even-integer detection via magic round (fadd/fsub/cmpf only).
    e_int = ((e + _MAGIC_C) - _MAGIC_C) == e
    h = 0.5 * e
    e_even = ((h + _MAGIC_C) - _MAGIC_C) == h
    big = tl.abs(e) >= _MAGIC_BIG  # |e| huge -> always an even integer -> no flip
    # sign cascade for a negative-signed base: odd int -> -r, else +r. Used for
    # -inf / -0.0 / negative-integer-exponent bases.
    signed = tl.where(big, r, tl.where(e_int, tl.where(e_even, r, -r), r))
    # FINITE negative base ^ non-integer exponent -> NaN. Produce the NaN
    # NATURALLY via log(SIGNED x): log of a negative operand yields NaN. This
    # MUST NOT be done with a `tl.where(mask, float("nan"), ...)` literal select
    # -- that de-vectorizes the whole store cone on xpu3 (~10x slower,
    # 0.23x->0.93x at 2048^2; verified probe_fp_bisect) AND miscompiles under a
    # bf16->f32 `.to()` widen. `rn` is a real fp value in every lane so the
    # cascade stays SIMD. `rn` is ONLY consumed in the finite-neg-base +
    # non-integer-e lane where the answer is NaN, so a bare `log(x)` (ONE
    # transcendental) suffices -- no need for the full `exp(log(x)*e)` (TWO):
    # both give NaN there, but log alone is ~15% faster (0.935x->1.09x @ 2048^2,
    # verified probe_fp_cheapnan, 0 mismatches over 225 corner combos).
    rn = tl.log(x)
    is_inf = ax == float("inf")
    # x<0 branch: -inf / |e| huge / integer e -> signed ; else (finite neg,
    # non-integer e) -> rn (== NaN).
    neg = tl.where(is_inf, signed, tl.where(big, signed, tl.where(e_int, signed, rn)))
    # dispatch: x<0 (finite -or- -inf) -> neg ; -0.0 (1/x<0) -> signed ; else r.
    out = tl.where(x < 0.0, neg, tl.where(1.0 / x < 0.0, signed, r))
    # Exact integer-power fixup: x**1 == x exactly (sign/inf/nan/-0 preserved).
    return tl.where(e == 1.0, x, out)


# ---------------------------------------------------------------------------
# Fast 1D big-tile kernels (contiguous, equal-shape inputs).
# ---------------------------------------------------------------------------


@triton.jit
def float_power_tt_fast_kernel(x_ptr, e_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offset).to(tl.float32)
    e = tl.load(e_ptr + offset).to(tl.float32)
    r = _float_power_corner(x, e)
    tl.store(out_ptr + offset, r)


@triton.jit
def float_power_tt_fast_kernel_masked(
    x_ptr, e_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=1.0).to(tl.float32)
    e = tl.load(e_ptr + offset, mask=mask, other=1.0).to(tl.float32)
    r = _float_power_corner(x, e)
    tl.store(out_ptr + offset, r, mask=mask)


# bf16 ** bf16 needs NO special path anymore. The only obstacle was that a NaN
# value nested inside the corner's where-cascade miscompiled on xpu3 when the
# inputs came from a bf16->f32 `.to()` widen in the same kernel (inf even in
# valid positive lanes). Moving the NaN to a single tail select in
# `_float_power_corner` (see there) fixed the miscompile, so bf16 now flows
# through the normal `float_power_tt_fast_kernel` (dual `.to(fp32)`) exactly like
# fp16/fp32 -- one vectorized kernel, no fp32 scratch, no extra widen pass.


# Tensor ** scalar-exponent (scalar `e` passed as a runtime kernel arg).
@triton.jit
def float_power_ts_fast_kernel(x_ptr, e, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offset).to(tl.float32)
    r = _float_power_corner(x, e)
    tl.store(out_ptr + offset, r)


@triton.jit
def float_power_ts_fast_kernel_masked(
    x_ptr, e, out_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=1.0).to(tl.float32)
    r = _float_power_corner(x, e)
    tl.store(out_ptr + offset, r, mask=mask)


@triton.jit
def _float_power_ts_frac(x, e, ninf_val):
    # Tensor base ** NON-INTEGER scalar exponent (e known finite non-integer on
    # host). Since e is non-integer, EVERY finite negative base -> NaN, so the
    # whole result collapses to a SINGLE `exp(log(x)*e)`: log(x<0)=NaN which exp
    # propagates (the natural NaN, stays SIMD), log(x=+inf)=+inf -> +-inf/0,
    # log(0)=-inf -> 0/+inf, all matching ATen. Only x=-inf needs a fixup
    # (log(-inf)=NaN but ATen gives +inf for e>0, 0 for e<0): supply it as
    # `ninf_val` -- a RUNTIME scalar (float("inf") or 0.0 chosen on host by
    # sign(e)), NOT a compile-time literal in the select, so the store cone stays
    # vectorized. This is HALF the transcendentals of the full corner (2 vs 3) and
    # skips the entire integer-parity/sign cascade -> ~2x at 2048^2 (verified
    # probe_fp_tsfrac, 0 mismatches over the base grid for e in {+-0.5,+-1.5,2.5,1.234}).
    r = tl.exp(tl.log(x) * e)
    ninf = x == float("-inf")
    r = tl.where(ninf, ninf_val, r)
    return tl.where(x == 1.0, 1.0, r)


@triton.jit
def float_power_ts_frac_kernel(x_ptr, e, ninf_val, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offset).to(tl.float32)
    tl.store(out_ptr + offset, _float_power_ts_frac(x, e, ninf_val))


@triton.jit
def float_power_ts_frac_kernel_masked(
    x_ptr, e, ninf_val, out_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=1.0).to(tl.float32)
    tl.store(out_ptr + offset, _float_power_ts_frac(x, e, ninf_val), mask=mask)


# scalar-base ** Tensor (scalar `x` passed as a runtime kernel arg).
@triton.jit
def float_power_st_fast_kernel(x, e_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(e_ptr + offset).to(tl.float32)
    r = _float_power_corner(x, e)
    tl.store(out_ptr + offset, r)


@triton.jit
def float_power_st_fast_kernel_masked(
    x, e_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n_elements
    e = tl.load(e_ptr + offset, mask=mask, other=1.0).to(tl.float32)
    r = _float_power_corner(x, e)
    tl.store(out_ptr + offset, r, mask=mask)


@triton.jit
def _float_power_st_pos(x, e):
    # Scalar base x KNOWN >= 0 (incl 0, 1, +inf): no sign flip, no negative-base
    # NaN, so a SINGLE transcendental exp(log(x)*e) suffices (matches aten's fast
    # scalar^tensor path). Skips the second `exp(log(SIGNED x)*e)` NaN pass in the
    # full corner, ~halving the transcendental cost at large n (st 2048^2
    # 0.47x->~0.85x). log(0)=-inf/log(inf)=inf propagate the 0^e/ inf^e edges
    # correctly; the |x|==1 / e==0 / e==1 fixups match _float_power_corner.
    r = tl.exp(tl.log(x) * e)
    r = tl.where(x == 1.0, 1.0, tl.where(e == 0.0, 1.0, r))
    return tl.where(e == 1.0, x, r)


@triton.jit
def float_power_st_pos_kernel(x, e_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(e_ptr + offset).to(tl.float32)
    tl.store(out_ptr + offset, _float_power_st_pos(x, e))


@triton.jit
def float_power_st_pos_kernel_masked(
    x, e_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n_elements
    e = tl.load(e_ptr + offset, mask=mask, other=1.0).to(tl.float32)
    tl.store(out_ptr + offset, _float_power_st_pos(x, e), mask=mask)


def _pick_fp_block(n_elements):
    # 32768/8 measured best for large divisible sizes (2026-09-04 A/B);
    # small sizes use a masked 2048 block.
    if n_elements >= 262_144 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 16384 and n_elements % 16384 == 0:
        return 16384, 8, False
    if n_elements <= 65536:
        return MIN_BLOCK, 4, True
    return 16384, 8, True


def _launch_float_power_tt_fast(x, e, out):
    # All dtypes (fp16/fp32/bf16, and mixed) flow through the one vectorized
    # corner kernel, which loads both operands with `.to(tl.float32)`. bf16**bf16
    # needs NO special path: the old bf16 miscompile was the NaN-in-nested-cascade
    # bug, now fixed by the tail-nan select in `_float_power_corner`.
    n_elements = x.numel()
    if n_elements == 0:
        return
    block_size, num_warps, masked = _pick_fp_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        float_power_tt_fast_kernel_masked[grid](
            x,
            e,
            out,
            n_elements,
            BLOCK=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        float_power_tt_fast_kernel[grid](
            x,
            e,
            out,
            BLOCK=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def _fast_tt_ok(A, exponent, shape):
    # All fp16/fp32/bf16 combinations (incl. bf16**bf16) go through the single
    # vectorized float_power_tt_fast_kernel (dual `.to(fp32)` load + tail-nan
    # corner). No dtype is excluded now that the nested-nan miscompile is fixed.
    return (
        A.shape == shape
        and exponent.shape == shape
        and A.is_contiguous()
        and exponent.is_contiguous()
        and not A.is_complex()
        and not exponent.is_complex()
    )


def _launch_float_power_ts_fast(x, e, out):
    # Tensor base `x` ** scalar exponent `e`; stores fp32 directly (no fp64 cast).
    n_elements = x.numel()
    if n_elements == 0:
        return
    e = float(e)
    block_size, num_warps, masked = _pick_fp_block(n_elements)
    # NON-INTEGER finite exponent -> lean 2-transcendental frac path (every finite
    # negative base is NaN, so exp(log(x)*e) alone is exact; only x=-inf needs a
    # runtime `ninf_val` fixup). ~2x the full corner (verified probe_fp_tsfrac).
    # Integer / inf / NaN exponents keep the full sign/parity corner.
    frac = math.isfinite(e) and e != math.floor(e)
    if frac:
        ninf_val = float("inf") if e > 0.0 else 0.0
        if masked:
            grid = (triton.cdiv(n_elements, block_size),)
            float_power_ts_frac_kernel_masked[grid](
                x,
                e,
                ninf_val,
                out,
                n_elements,
                BLOCK=block_size,
                num_warps=num_warps,
                unroll_num=UNROLL_NUM,
                buffer_size_limit=BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            )
        else:
            grid = (n_elements // block_size,)
            float_power_ts_frac_kernel[grid](
                x,
                e,
                ninf_val,
                out,
                BLOCK=block_size,
                num_warps=num_warps,
                unroll_num=UNROLL_NUM,
                buffer_size_limit=BUFFER_SIZE_LIMIT,
                isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            )
        return
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        float_power_ts_fast_kernel_masked[grid](
            x,
            e,
            out,
            n_elements,
            BLOCK=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        float_power_ts_fast_kernel[grid](
            x,
            e,
            out,
            BLOCK=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def _launch_float_power_st_fast(x, e, out):
    # Scalar base `x` ** Tensor exponent `e`; stores fp32 directly (no fp64 cast).
    n_elements = e.numel()
    if n_elements == 0:
        return
    x = float(x)
    # x >= 0 (incl 0/1/+inf): use the lean single-transcendental corner (no sign
    # flip / NaN needed since the base is non-negative). Halves the exp/log cost.
    pos = x >= 0.0
    kern = (
        (float_power_st_pos_kernel_masked, float_power_st_pos_kernel)
        if pos
        else (float_power_st_fast_kernel_masked, float_power_st_fast_kernel)
    )
    block_size, num_warps, masked = _pick_fp_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        kern[0][grid](
            x,
            e,
            out,
            n_elements,
            BLOCK=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )
    else:
        grid = (n_elements // block_size,)
        kern[1][grid](
            x,
            e,
            out,
            BLOCK=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
        )


def _fast_scalar_ok(T):
    return T.is_contiguous() and not T.is_complex()


# ---------------------------------------------------------------------------
# Generic pointwise fallbacks (broadcast / non-contiguous / complex).
# ---------------------------------------------------------------------------


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def float_power_func(x, exponent):
    # In-place (out0 = input tensor, dtype preserved). Use the fully-vectorizable
    # corner implementation (handles negative base / integer-parity / inf / NaN
    # exactly, unlike the bare `_pow` extern which NaNs any negative base).
    return _float_power_corner(x.to(tl.float32), exponent.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def float_power_func_tensor_scalar(x, exponent):
    return _float_power_corner(x.to(tl.float32), exponent.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def float_power_tensor_tensor_func(x, exponent):
    # Functional (out-of-place) variant. The out0 tensor is nominally Double per
    # the float_power contract; on kunlunxin torch_xmlir stores it as float32, so
    # the trailing .to(tl.float64) truncates back to fp32 at store time.
    return _float_power_corner(x.to(tl.float32), exponent.to(tl.float32)).to(tl.float64)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def float_power_tensor_scalar_func(x, exponent):
    return _float_power_corner(x.to(tl.float32), exponent.to(tl.float32)).to(tl.float64)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def float_power_scalar_tensor_func(x, exponent):
    return _float_power_corner(x.to(tl.float32), exponent.to(tl.float32)).to(tl.float64)


def _prepare_out(out, shape, device):
    # PyTorch's float_power contract requires a Double out. On kunlunxin (XPU),
    # torch_xmlir hard-maps torch.float64 -> float32 at the tensor layer, so a
    # user-supplied Double out actually arrives here as float32 (there is no way
    # to tell it apart from a genuine float32). Accept float32/float64 as the
    # Double output; reject any other dtype (int/half/bf16/complex).
    if out.dtype not in (torch.float64, torch.float32):
        raise RuntimeError(
            f"the output given to float_power has dtype {out.dtype} "
            "but the operation's result requires dtype Double"
        )
    if out.device != device:
        raise RuntimeError(
            f"Expected out tensor to have device {device}, but got {out} instead"
        )
    if out.shape != shape:
        out.resize_(shape)
    return out


# ---------------------------------------------------------------------------
# Functional / in-place entries.
# ---------------------------------------------------------------------------


def float_power_tensor_tensor(A, exponent):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_TENSOR_TENSOR")
    # Equal-shape lean fast path: skip torch.broadcast_shapes (~8us host cost at
    # tiny sizes -- the dominant overhead vs aten there) when both operands
    # already share A.shape. This is the overwhelmingly common case.
    if (
        A.shape == exponent.shape
        and A.is_contiguous()
        and exponent.is_contiguous()
        and not A.is_complex()
        and not exponent.is_complex()
    ):
        out = torch.empty(A.shape, dtype=torch.float64, device=A.device)
        _launch_float_power_tt_fast(A, exponent, out)
        return out
    shape = torch.broadcast_shapes(A.shape, exponent.shape)
    out = torch.empty(shape, dtype=torch.float64, device=A.device)
    if _fast_tt_ok(A, exponent, shape):
        _launch_float_power_tt_fast(A, exponent, out)
        return out
    return float_power_tensor_tensor_func(A, exponent, out0=out)


def float_power_tensor_tensor_(A, exponent):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_TENSOR_TENSOR_")
    if _fast_tt_ok(A, exponent, A.shape):
        _launch_float_power_tt_fast(A, exponent, A)
        return A
    return float_power_func(A, exponent, out0=A)


def float_power_tensor_tensor_out(A, exponent, *, out):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_TENSOR_TENSOR_OUT")
    shape = torch.broadcast_shapes(A.shape, exponent.shape)
    _prepare_out(out, shape, A.device)
    if _fast_tt_ok(A, exponent, shape) and out.is_contiguous():
        _launch_float_power_tt_fast(A, exponent, out)
        return out
    return float_power_tensor_tensor_func(A, exponent, out0=out)


def float_power_tensor_scalar(A, exponent):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_TENSOR_SCALAR")
    out = torch.empty(A.shape, dtype=torch.float64, device=A.device)
    if _fast_scalar_ok(A):
        _launch_float_power_ts_fast(A, exponent, out)
        return out
    return float_power_tensor_scalar_func(A, exponent, out0=out)


def float_power_tensor_scalar_(A, exponent):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_TENSOR_SCALAR_")
    if _fast_scalar_ok(A):
        _launch_float_power_ts_fast(A, exponent, A)
        return A
    return float_power_func_tensor_scalar(A, exponent, out0=A)


def float_power_tensor_scalar_out(A, exponent, *, out):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_TENSOR_SCALAR_OUT")
    _prepare_out(out, A.shape, A.device)
    if _fast_scalar_ok(A) and out.is_contiguous():
        _launch_float_power_ts_fast(A, exponent, out)
        return out
    return float_power_tensor_scalar_func(A, exponent, out0=out)


def float_power_scalar_tensor(A, exponent):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_SCALAR_TENSOR")
    out = torch.empty(exponent.shape, dtype=torch.float64, device=exponent.device)
    if _fast_scalar_ok(exponent):
        _launch_float_power_st_fast(A, exponent, out)
        return out
    return float_power_scalar_tensor_func(A, exponent, out0=out)


def float_power_scalar_tensor_out(A, exponent, *, out):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_SCALAR_TENSOR_OUT")
    _prepare_out(out, exponent.shape, exponent.device)
    if _fast_scalar_ok(exponent) and out.is_contiguous():
        _launch_float_power_st_fast(A, exponent, out)
        return out
    return float_power_scalar_tensor_func(A, exponent, out0=out)
