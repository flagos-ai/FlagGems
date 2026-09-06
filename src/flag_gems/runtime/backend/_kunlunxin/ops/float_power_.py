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

# Kunlunxin(XPU) float_power / float_power_ vendor implementation.
#
# Why a vendor override is required (xpu3 libdevice, 2026-09-04):
#   * `triton.language.extra.xpu.libdevice.pow` maps (fp64, fp64) to
#     ("Unsupported", fp64) and fp64 exp2/log2 are "Unsupported" as well, so
#     the generic implementation (`_pow(x.to(tl.float64), e.to(tl.float64))`)
#     fails to link with `ld.lld: undefined symbol: Unsupported` on XPU.
#   * The generic in-place variants already cast to tl.float32 before pow
#     (lowering to the supported `_ZN3xpu3powEff`); the functional
#     `float_power.{Tensor_Tensor,Tensor_Scalar,Scalar}` (float64 out)
#     variants do not, so the benchmark's `torch.float_power(...)` call under
#     use_gems fails to compile end-to-end on XPU.
#   * Additionally, this TorchXPU build remaps every device-side float64
#     allocation to float32 (no fp64 hardware), so `torch.empty(..., float64)`
#     is a float32 tensor here; the functional variants' f64 output is
#     therefore float32 on the device.
#
# Implementation:
#   * Fast path (contiguous, equal-shape inputs): 1D big-tile kernel on the
#     SFU chain `r = exp2(e * log2(|x|))` (on this backend tl.exp2/tl.log2
#     are e^x / ln(x) numerically, so r == |x|^e), with in-kernel corner-case
#     handling (negative base x integer exponent => sign * |x|^e; negative
#     base x non-integer exponent => NaN; e == 0 => 1; x = 0 / +/-inf / NaN
#     fall out of the SFU chain). Mirrors the approach validated for
#     `_kunlunxin/ops/pow.py` (2026-08-15/19 probes) and the 2026-08-17
#     float_power_ candidate.
#   * Fallback: pointwise_dynamic fp32 pow (same math as the generic
#     in-place implementation) for broadcast / non-contiguous inputs.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim, triton_lang_extension as ext

_pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)

MIN_BLOCK = 2048
UNROLL_NUM = 16
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


# ---------------------------------------------------------------------------
# Fast 1D big-tile kernels (contiguous, equal-shape inputs).
# ---------------------------------------------------------------------------


@triton.jit
def float_power_tt_fast_kernel(x_ptr, e_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offset).to(tl.float32)
    e = tl.load(e_ptr + offset).to(tl.float32)
    r = tl.exp2(tl.log2(tl.abs(x)) * e)
    # corner cases (match ATen/C++ pow semantics; 2026-09-06: e_int via fptosi
    # instead of tl.floor -- an extern call costing ~1.2ms at 16M elements).
    # (+-1)^any == 1 (incl. 1^(+-inf/NaN)) and e == 0 -> 1 (incl. 0^0, NaN^0)
    r = tl.where((tl.abs(x) == 1.0) | (e == 0.0), 1.0, r)
    # Integer-exponent test via fptosi (exact for |e| < 2^24; fptosi saturates
    # outside that range, so mask |e| >= 2^24 (and inf/NaN) as "integer": every
    # representable value there is an even multiple -> no sign change).
    eb = e.to(tl.int32, bitcast=True)
    e_big = (eb & 0x7F800000) >= 0x4B800000
    e_int = e_big | (e.to(tl.int32).to(tl.float32) == e)
    e_odd = (e.to(tl.int32) & 1) != 0
    # sign of x (x < 0 or -0.0): bitcast sign bit of the fp32 value
    neg = x.to(tl.int32, bitcast=True) < 0
    # odd-integer exponent (|e| < 2^24, exact fptosi; big e is even -> no sign)
    r = tl.where(neg & e_int & e_odd & (~e_big), -r, r)
    # (x < 0) & (x > -inf) & (x != -0): only a strictly-negative FINITE base
    # with a non-integer exponent yields NaN (std::pow(-inf/_ -0, y) is
    # +-inf/+-0 by sign of y, never NaN; the old x > -inf test wrongly NaN-ed
    # std::pow(-0.0, 0.5/2.5) which is +0).
    r = tl.where(neg & (x < 0.0) & (x > -float("inf")) & (~e_int), float("nan"), r)
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
    r = tl.exp2(tl.log2(tl.abs(x)) * e)
    r = tl.where((tl.abs(x) == 1.0) | (e == 0.0), 1.0, r)
    eb = e.to(tl.int32, bitcast=True)
    e_big = (eb & 0x7F800000) >= 0x4B800000
    e_int = e_big | (e.to(tl.int32).to(tl.float32) == e)
    e_odd = (e.to(tl.int32) & 1) != 0
    neg = x.to(tl.int32, bitcast=True) < 0
    r = tl.where(neg & e_int & e_odd & (~e_big), -r, r)
    r = tl.where(neg & (x < 0.0) & (x > -float("inf")) & (~e_int), float("nan"), r)
    tl.store(out_ptr + offset, r, mask=mask)


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
    return (
        A.shape == shape
        and exponent.shape == shape
        and A.is_contiguous()
        and exponent.is_contiguous()
        and not A.is_complex()
        and not exponent.is_complex()
    )


# ---------------------------------------------------------------------------
# Generic pointwise fallbacks (broadcast / non-contiguous / complex).
# ---------------------------------------------------------------------------


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def float_power_func(x, exponent):
    # In-place (out0 = input tensor, dtype preserved). fp32 pow is supported
    # on XPU; same computation as the generic implementation.
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def float_power_func_tensor_scalar(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def float_power_tensor_tensor_func(x, exponent):
    # Functional variant (float64 out, which is float32 on this platform).
    # xpu3 libdevice has no fp64 pow: compute in fp32 (hardware powf) and
    # upcast to the output dtype.
    return _pow(x.to(tl.float32), exponent.to(tl.float32)).to(tl.float64)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def float_power_tensor_scalar_func(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32)).to(tl.float64)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def float_power_scalar_tensor_func(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32)).to(tl.float64)


def _prepare_out(out, shape, device):
    if out.dtype != torch.float64:
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
    if _fast_tt_ok(A, exponent, shape):
        _launch_float_power_tt_fast(A, exponent, out)
        return out
    return float_power_tensor_tensor_func(A, exponent, out0=out)


def float_power_tensor_scalar(A, exponent):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_TENSOR_SCALAR")
    out = torch.empty(A.shape, dtype=torch.float64, device=A.device)
    return float_power_tensor_scalar_func(A, exponent, out0=out)


def float_power_tensor_scalar_(A, exponent):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_TENSOR_SCALAR_")
    return float_power_func_tensor_scalar(A, exponent, out0=A)


def float_power_tensor_scalar_out(A, exponent, *, out):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_TENSOR_SCALAR_OUT")
    _prepare_out(out, A.shape, A.device)
    return float_power_tensor_scalar_func(A, exponent, out0=out)


def float_power_scalar_tensor(A, exponent):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_SCALAR_TENSOR")
    out = torch.empty(exponent.shape, dtype=torch.float64, device=exponent.device)
    return float_power_scalar_tensor_func(A, exponent, out0=out)


def float_power_scalar_tensor_out(A, exponent, *, out):
    logger.debug("GEMS_KUNLUNXIN FLOAT_POWER_SCALAR_TENSOR_OUT")
    _prepare_out(out, exponent.shape, exponent.device)
    return float_power_scalar_tensor_func(A, exponent, out0=out)