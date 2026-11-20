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

"""XPU override for ``ldexp`` / ``ldexp_`` / ``ldexp.out``.

The generic implementation is numerically correct on CUDA but trips three
TritonXPU/device quirks:

1. ``tl.exp2`` is lowered to the natural exponential on this backend
   (``tl.exp2(3.0) == 20.0855`` instead of ``8.0``).  Every scale therefore
   goes through the xpu libdevice ``exp2f`` extern.
2. ``exp2f`` is exact for ``y < 128`` (subnormal results included) but returns
   ``0`` instead of ``+inf`` once ``2**y`` overflows fp32, so the overflow
   range is special-cased.
3. fp32 arithmetic flushes subnormal results to zero *and* normalises the sign
   of zero (``-0.0 * 1.0 -> +0.0``), while ATen keeps ``-0.0`` and subnormal
   scales.  Results are re-signed with a bit-level ``copysign`` so ``-0.0`` /
   ``-2**-128``-style outputs keep the ATen sign bit.

The integral-exponent hot path avoids the ``ldexpf`` soft extern entirely and
builds the scale from two exponent-field factors (KernelOpt_Knowledge
technique ``xpu3-exponent-field-two-factor-split``), and ``ldexp.out`` writes
straight into the caller's contiguous buffer instead of materialising a
temporary result plus a copy kernel.  Kernels use the Kunlunxin
``pointwise_dynamic`` fork with a large 1D tile: the generic pointwise config
degenerates on the 16.7M-element core shapes (55 ms vs 0.8 ms at 4096x4096
fp32).
"""

import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.ops.copy import copy_
from flag_gems.ops.neg import neg
from flag_gems.ops.zeros_like import zeros_like
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping
from flag_gems.utils.type_utils import ELEMENTWISE_TYPE_PROMOTION_KIND, type_promotion

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_WRAP_INTEGER_EXPONENT = tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 11)

# The Kunlunxin pointwise fork with a large 1D tile keeps the 16.7M-element
# core shapes memory-bound; the generic pointwise config degenerates to a
# per-lane discrete access pattern here (55 ms vs 0.8 ms on 4096x4096 fp32).
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)

# ``exp2f`` on this device starts returning 0 once the fp32 result overflows
# (2**128 and above).  Constants used inside @triton.jit must stay literals or
# ``tl.constexpr`` instances, hence the local literal in ``_exp2_xpu``.
_EXP2_OVERFLOW = tl.constexpr(128.0)

# ---------------------------------------------------------------------------
# Flat fast path: fp16/bf16 source, integral exponent, dense same-shape operands
# ---------------------------------------------------------------------------
# The pointwise_dynamic codegen always emits a runtime tail mask plus a
# per-program stride chain, which costs ~2x on the half-dtype benchmark shapes
# (batch 20261119_01: 0.45-0.56x fp32, 1.05-1.26x half before the fast path).
# The fast path launches a lean 1-D kernel instead:
#   * the tail mask is a constexpr (``n % BLOCK == 0`` -> unmasked block DMA;
#     an always-true runtime mask is ~1.5x slower on this backend);
#   * fp16 sources use a single +-126 exponent factor: any |e| > 126 pushes the
#     result beyond the fp16 range for every nonzero fp16 input, so one factor
#     is exact after fp16 rounding;
#   * bf16 sources keep the two +-126 factors (+-252 total) because the bf16
#     range matches fp32;
#   * no subnormal rescale is needed: an fp16/bf16 subnormal is a normal fp32
#     number after promotion;
#   * the sign bit of x is OR-ed onto the result so -0.0 keeps its sign, as in
#     the fp32 path and in ATen.
_FLAT_BLOCK_MIN = 256
_FLAT_BLOCK_MAX = 16384
# Below this size the pointwise launcher is cheaper than the flat one
# (measured crossover on XPU6: n<=8 flat 5.2us vs pointwise 4.85us; n>=512 the
# flat kernel is 5-10% faster, and at n=64/128 it is 2.6x faster).
_FLAT_MIN_ELEMENTS = 256
_FLAT_DTYPES = (torch.float16, torch.bfloat16)
_FLAT_EXPONENT_DTYPES = (torch.int8, torch.int16, torch.int32)


@triton.jit
def _flat_pow2_scale(x, exponent, single: tl.constexpr):
    """``x * 2**exponent`` for a half-precision source promoted to fp32."""
    if single:
        exponent_hi = tl.maximum(tl.minimum(exponent, 126), -126)
        scale_hi = ((exponent_hi + 127) << 23).to(tl.float32, bitcast=True)
        return x * scale_hi
    exponent = tl.maximum(tl.minimum(exponent, 252), -252)
    exponent_hi = tl.maximum(tl.minimum(exponent, 126), -126)
    scale_hi = ((exponent_hi + 127) << 23).to(tl.float32, bitcast=True)
    scale_lo = (((exponent - exponent_hi) + 127) << 23).to(tl.float32, bitcast=True)
    return (x * scale_hi) * scale_lo


@libentry()
@triton.jit
def _ldexp_flat_kernel(
    x_ptr,
    exponent_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
    SINGLE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask, other=0.0)
        exponent = tl.load(exponent_ptr + offsets, mask, other=0)
    else:
        x = tl.load(x_ptr + offsets)
        exponent = tl.load(exponent_ptr + offsets)
    xf = x.to(tl.float32)
    result = _flat_pow2_scale(xf, exponent.to(tl.int32), SINGLE)
    # ATen keeps the sign of a flushed product (``-0.0 * 2**e`` stays -0.0);
    # the multiplier normalises zero signs, so OR the sign back (no-op for
    # every non-zero product).
    result = (
        result.to(tl.int32, bitcast=True)
        | (xf.to(tl.int32, bitcast=True) & -2147483648)
    ).to(tl.float32, bitcast=True)
    result = result.to(out_ptr.dtype.element_ty)
    if MASKED:
        tl.store(out_ptr + offsets, result, mask)
    else:
        tl.store(out_ptr + offsets, result)


def _flat_block_size(n_elements):
    block = _FLAT_BLOCK_MIN
    while block < n_elements and block < _FLAT_BLOCK_MAX:
        block *= 2
    return block


def _run_flat_ldexp(self, other, out):
    n_elements = self.numel()
    if n_elements == 0:
        return out
    block = _flat_block_size(n_elements)
    grid = (triton.cdiv(n_elements, block),)
    _ldexp_flat_kernel[grid](
        self,
        other,
        out,
        n_elements,
        BLOCK=block,
        MASKED=(n_elements % block) != 0,
        SINGLE=(self.dtype == torch.float16),
    )
    return out


def _flat_fast_ok(self, other, out, result_dtype):
    if result_dtype not in (torch.float16, torch.bfloat16):
        return False
    if self.dtype != result_dtype or not self.is_floating_point():
        return False
    if self.numel() < _FLAT_MIN_ELEMENTS:
        return False
    if not (self.is_contiguous() and other.is_contiguous()):
        return False
    if tuple(self.shape) != tuple(other.shape):
        return False
    if other.dtype not in _FLAT_EXPONENT_DTYPES:
        # int64 exponents keep the pointwise path: an i64 clamp + trunc pair
        # inside a masked 1-D kernel hits the known XPU lowering bug.
        return False
    if out is not None:
        if out.dtype != result_dtype or not out.is_contiguous():
            return False
        if out.is_conj() or out.is_neg():
            return False
        if tuple(out.shape) != tuple(self.shape):
            return False
    return True


@triton.jit
def _exp2_xpu(x):
    """2**x built from ``tl.exp``.

    Neither ``exp2`` flavour is usable on this backend: ``tl.exp2`` lowers to
    the natural exponential (``tl.exp2(3.0) == 20.0855``) and the libdevice
    ``exp2f`` extern has no XPU3 device symbol in the CI FlagTree build
    (``undefined symbol: Unsupported`` at link time).  ``tl.exp`` lowers on
    every build, so the scale is rebuilt as ``exp(x * ln2)`` and the overflow
    range (where ``exp2f`` used to return 0) is resolved explicitly.
    """
    return tl.where(x >= _EXP2_OVERFLOW, float("inf"), tl.exp(x * 0.6931471805599453))


@triton.jit
def _signed_f32(value, sign):
    """|value| with the sign of ``sign`` (``copysign`` as fp32 bit ops).

    ``libdevice.copysign`` has no XPU3 device symbol in the CI FlagTree build
    (the link fails with ``undefined symbol: Unsupported``), so the sign bit
    is OR-ed explicitly, as done in ``_ldexp32_xpu``.
    """
    value_bits = value.to(tl.float32).to(tl.int32, bitcast=True)
    sign_bits = sign.to(tl.float32).to(tl.int32, bitcast=True)
    return ((value_bits & 0x7FFFFFFF) | (sign_bits & -2147483648)).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _mul_pow2(xf, ef):
    """``xf * 2**ef`` for the fp32 path.

    ``2**ef`` overflows for ``ef >= 128``.  Going through the multiplier then
    would turn a subnormal ``xf`` into ``0 * inf = NaN`` (the XPU multiplier
    flushes subnormal inputs) while ATen keeps ``x * inf == +-inf`` for every
    nonzero ``x`` and only produces NaN for ``x == 0`` / NaN.  The overflow
    range is therefore resolved from the raw fp32 bits.
    """
    raw = xf.to(tl.int32, bitcast=True)
    exp_bits = (raw >> 23) & 0xFF
    mant_bits = raw & 0x7FFFFF
    special = ((exp_bits == 0xFF) & (mant_bits != 0)) | (
        (exp_bits == 0) & (mant_bits == 0)
    )
    overflow = ef >= _EXP2_OVERFLOW
    inf_value = tl.where(raw < 0, float("-inf"), float("inf"))
    inf_value = tl.where(special, float("nan"), inf_value)
    scale = tl.where(overflow, 1.0, _exp2_xpu(ef))
    return tl.where(overflow, inf_value, xf * scale)


@triton.jit
def _sign_fix_f32(value, sign):
    # fp32 results keep their dtype; copysign only accepts fp32 on this
    # backend, so wider (emulated) values are re-signed in fp32 and cast back.
    return _signed_f32(value, sign).to(value.dtype)


@triton.jit
def _ldexp32_xpu(x, exponent):
    """Exact fp32 ``x * 2**exponent`` built from two exponent-field factors.

    The xpu ``ldexpf`` extern is a per-lane soft call (~50x the memory floor at
    16.7M elements) and treats a subnormal ``x`` as zero.  Following
    ``KernelOpt_Knowledge`` (technique-xpu3-exponent-field-two-factor-split),
    the scale is assembled from two normal-range factors instead:

    * splitting ``e`` into ``e1 = clamp(e, +-126)`` and ``e - e1`` keeps both
      factors representable, so ``x * s1 * s2`` is exact for ``|e| <= 252``;
    * a subnormal/zero ``x`` is rescaled through its mantissa integer and the
      implicit ``-149`` exponent is folded into ``e`` (XPU3 flushes subnormal
      operands to zero, so they must never enter the multiplier);
    * the multiplier normalises zero signs, so the sign bit of ``x`` is OR-ed
      back onto the result (a no-op for every non-zero product).
    """
    bits = x.to(tl.int32, bitcast=True)
    tiny_or_zero = (bits & 0x7FFFFFFF) < 0x00800000
    mantissa = (bits & 0x7FFFFF).to(tl.float32)
    value = tl.where(tiny_or_zero, mantissa, x)
    exponent = tl.where(tiny_or_zero, exponent - 149, exponent)
    exponent = tl.maximum(tl.minimum(exponent, 252), -252)
    exponent_hi = tl.maximum(tl.minimum(exponent, 126), -126)
    scale_hi = ((exponent_hi + 127) << 23).to(tl.float32, bitcast=True)
    scale_lo = (((exponent - exponent_hi) + 127) << 23).to(tl.float32, bitcast=True)
    result = (value * scale_hi) * scale_lo
    return (result.to(tl.int32, bitcast=True) | (bits & -2147483648)).to(
        tl.float32, bitcast=True
    )


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def ldexp_func(x, y):
    if y.dtype.is_int() and x.dtype.is_floating():
        xf = x.to(tl.float32)
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
        return _signed_f32(_ldexp32_xpu(xf, exponent), xf)
    xf = x.to(tl.float32)
    return _signed_f32(_mul_pow2(xf, y.to(tl.float32)), xf)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def ldexp_fp64_func(x, y):
    if y.dtype.is_int() and x.dtype.is_floating():
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
        return _sign_fix_f32(tl_extra_shim.ldexp(x.to(tl.float64), exponent), x)
    return _sign_fix_f32(x.to(tl.float64) * _exp2_xpu(y.to(tl.float64)), x)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=config_,
)
@triton.jit
def ldexp_integral_func(x, y, wrap_exponent: tl.constexpr):
    exponent = y.to(tl.int32)
    if not wrap_exponent:
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
    xf = x.to(tl.float32)
    return _signed_f32(_ldexp32_xpu(xf, exponent), xf)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=config_,
)
@triton.jit
def ldexp_integral_fp64_func(x, y, wrap_exponent: tl.constexpr):
    exponent = y.to(tl.int32)
    if not wrap_exponent:
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
    return _sign_fix_f32(tl_extra_shim.ldexp(x.to(tl.float64), exponent), x)


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
    config=config_,
)
@triton.jit
def ldexp_complex_func(xr, xi, yr, yi):
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    yr = yr.to(tl.float32)
    yi = yi.to(tl.float32)
    scale = _exp2_xpu(yr)
    angle = yi * 0.6931471805599453
    cos_angle = tl.cos(angle)
    sin_angle = tl.sin(angle)
    pr = scale * cos_angle
    pi = tl.where(yi == 0, 0, scale * sin_angle)
    return xr * pr - xi * pi, xr * pi + xi * pr


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
    config=config_,
)
@triton.jit
def ldexp_complex_fp64_func(xr, xi, yr, yi):
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    yr = yr.to(tl.float32)
    yi = yi.to(tl.float32)
    scale = _exp2_xpu(yr)
    angle = yi * 0.693147180559945309417232121458176568
    cos_angle = tl.cos(angle)
    sin_angle = tl.sin(angle)
    pr = scale * cos_angle
    pi = tl.where(yi == 0, 0, scale * sin_angle)
    return xr * pr - xi * pi, xr * pi + xi * pr


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def ldexp_inplace_func(x, other):
    # IEEE: 0 * 2**(+inf) and ±inf * 2**(-inf) are NaN; _mul_pow2 already
    # resolves the overflow range from the fp32 bits, so no explicit
    # zero_times_inf guard is needed here.
    xf = x.to(tl.float32)
    return _signed_f32(_mul_pow2(xf, other.to(tl.float32)), xf)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def ldexp_inplace_fp64_func(x, other):
    return _sign_fix_f32(x.to(tl.float64) * _exp2_xpu(other.to(tl.float64)), x)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def ldexp_inplace_integral_func(x, other, wrap_exponent: tl.constexpr):
    exponent = other.to(tl.int32)
    if not wrap_exponent:
        exponent = tl.maximum(tl.minimum(other, 2147483647), -2147483648).to(tl.int32)
    xf = x.to(tl.float32)
    return _signed_f32(_ldexp32_xpu(xf, exponent), xf)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def ldexp_inplace_integral_fp64_func(x, other, wrap_exponent: tl.constexpr):
    exponent = other.to(tl.int32)
    if not wrap_exponent:
        exponent = tl.maximum(tl.minimum(other, 2147483647), -2147483648).to(tl.int32)
    return _sign_fix_f32(tl_extra_shim.ldexp(x.to(tl.float64), exponent), x)


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
    config=config_,
)
@triton.jit
def ldexp_inplace_complex_func(xr, xi, yr, yi):
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    yr = yr.to(tl.float32)
    yi = yi.to(tl.float32)
    scale = _exp2_xpu(yr)
    angle = yi * 0.6931471805599453
    cos_angle = tl.cos(angle)
    sin_angle = tl.sin(angle)
    pr = scale * cos_angle
    pi = tl.where(yi == 0, 0, scale * sin_angle)
    return xr * pr - xi * pi, xr * pi + xi * pr


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
    config=config_,
)
@triton.jit
def ldexp_inplace_complex_fp64_func(xr, xi, yr, yi):
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    yr = yr.to(tl.float32)
    yi = yi.to(tl.float32)
    scale = _exp2_xpu(yr)
    angle = yi * 0.693147180559945309417232121458176568
    cos_angle = tl.cos(angle)
    sin_angle = tl.sin(angle)
    pr = scale * cos_angle
    pi = tl.where(yi == 0, 0, scale * sin_angle)
    return xr * pr - xi * pi, xr * pi + xi * pr


@pointwise_dynamic(
    is_tensor=[True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ],
    config=config_,
)
@triton.jit
def ldexp_inplace_complex_integral_func(xr, xi, other):
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    scale = _exp2_xpu(other.to(tl.float32))
    return xr * scale - xi * 0.0, xr * 0.0 + xi * scale


@pointwise_dynamic(
    is_tensor=[True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ],
    config=config_,
)
@triton.jit
def ldexp_inplace_complex_integral_fp64_func(xr, xi, other):
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    scale = _exp2_xpu(other.to(tl.float32))
    return xr * scale - xi * 0.0, xr * 0.0 + xi * scale


def _result_dtype(self, other):
    _, result_dtype = type_promotion(
        self,
        other,
        type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    return result_dtype


def _result_device(self, other):
    if self.device == other.device:
        return self.device
    if (
        _WRAP_INTEGER_EXPONENT
        and self.is_floating_point()
        and not other.is_floating_point()
        and not other.is_complex()
    ):
        raise RuntimeError("Expected all tensors to be on the same device")
    if self.ndim == 0 and self.device.type == "cpu":
        return other.device
    if other.ndim == 0 and other.device.type == "cpu":
        return self.device
    raise RuntimeError(
        f"Expected all tensors to be on the same device, but found {self.device} "
        f"and {other.device}"
    )


def _is_exact_alias(left, right):
    return (
        left.data_ptr() == right.data_ptr()
        and left.storage_offset() == right.storage_offset()
        and left.shape == right.shape
        and left.stride() == right.stride()
    )


def _validate_out(self, other, out, result_dtype):
    result_device = _result_device(self, other)
    if out.device != result_device:
        raise RuntimeError(
            f"Expected out tensor to be on {result_device}, but got {out.device}"
        )
    if not torch.can_cast(result_dtype, out.dtype):  # allow-torch-compute
        raise RuntimeError(
            f"result type {result_dtype} can't be cast to the desired output "
            f"type {out.dtype}"
        )
    for operand in (self, other):
        if (
            operand.device == out.device
            and operand.untyped_storage().nbytes() > 0
            and operand.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
        ):
            if not _is_exact_alias(out, operand):
                raise RuntimeError(
                    "some elements of the input tensor and the written-to tensor "
                    "refer to a single memory location"
                )


def _complex_parts(tensor, real_dtype):
    if tensor.is_complex():
        physical = tensor.conj() if tensor.is_conj() else tensor
        parts = torch.view_as_real(physical)  # allow-torch-compute
        real = torch.empty(tensor.shape, dtype=real_dtype, device=tensor.device)
        imag = torch.empty_like(real)
        copy_(real, parts[..., 0])
        copy_(imag, neg(parts[..., 1]) if tensor.is_conj() else parts[..., 1])
        return real, imag
    real = torch.empty(tensor.shape, dtype=real_dtype, device=tensor.device)
    copy_(real, tensor)
    return real, zeros_like(real)


def _copy_result(out, result):
    if out.is_complex():
        physical = out.conj() if out.is_conj() else out
        parts = torch.view_as_real(physical)  # allow-torch-compute
        real, imag = _complex_parts(result, parts.dtype)
        copy_(parts[..., 0], real)
        copy_(parts[..., 1], neg(imag) if out.is_conj() else imag)
    else:
        copy_(out, result)
    return out


def _complex_ldexp(self, other, result_dtype):
    real_dtype = torch.float64 if result_dtype == torch.complex128 else torch.float32
    xr, xi = _complex_parts(self, real_dtype)
    yr, yi = _complex_parts(other, real_dtype)
    func = (
        ldexp_complex_fp64_func
        if result_dtype == torch.complex128
        else ldexp_complex_func
    )
    real, imag = func(xr, xi, yr, yi)
    result = torch.empty(real.shape, dtype=result_dtype, device=real.device)
    parts = torch.view_as_real(result)  # allow-torch-compute
    copy_(parts[..., 0], real)
    copy_(parts[..., 1], imag)
    return result


def _ldexp_impl(self, other, out=None):
    result_dtype = _result_dtype(self, other)
    if out is not None:
        _validate_out(self, other, out, result_dtype)
    device = _result_device(self, other)
    # CPU zero-dimensional operands are host scalar inputs, not device tensors
    # that may be passed to a GPU kernel unchanged.
    if self.device != device:
        self = torch.tensor(self.item(), dtype=self.dtype, device=device)
    if other.device != device:
        other = torch.tensor(other.item(), dtype=other.dtype, device=device)
    if self.is_complex() or other.is_complex():
        result = _complex_ldexp(self, other, result_dtype)
        if out is None:
            return result
        out.resize_(result.shape)
        return _copy_result(out, result)

    func = ldexp_fp64_func if result_dtype == torch.float64 else ldexp_func
    integral_exponent = (
        self.is_floating_point()
        and not other.is_floating_point()
        and not other.is_complex()
    )
    if integral_exponent:
        func = (
            ldexp_integral_fp64_func
            if result_dtype == torch.float64
            else ldexp_integral_func
        )
    if (
        integral_exponent
        and self.dtype in _FLAT_DTYPES
        and self.numel() >= _FLAT_MIN_ELEMENTS
        and _flat_fast_ok(self, other, out, result_dtype)
    ):
        # Lean 1-D path for the half-dtype dense case (batch 20261119_01).
        if out is not None:
            return _run_flat_ldexp(self, other, out)
        result = torch.empty(self.shape, dtype=result_dtype, device=device)
        return _run_flat_ldexp(self, other, result)
    if (
        out is not None
        and tuple(out.shape)
        == torch.broadcast_shapes(self.shape, other.shape)  # allow-torch-compute
        and out.dtype == result_dtype
        and out.is_contiguous()
        and not out.is_conj()
        and not out.is_neg()
    ):
        # Direct write into the caller's buffer (technique
        # xpu3-out-variant-direct-write): no temporary result and no copy
        # kernel.  Each lane loads before it stores, so exact aliasing with an
        # operand is safe; narrowed/non-contiguous/resized outs keep the
        # compute + copy path below.
        if integral_exponent:
            func(self, other, _WRAP_INTEGER_EXPONENT, out0=out)
        else:
            func(self, other, out0=out)
        return out
    result = (
        func(self, other, _WRAP_INTEGER_EXPONENT)
        if integral_exponent
        else func(self, other)
    )
    if out is None:
        return result
    out.resize_(result.shape)
    return _copy_result(out, result)


def ldexp(self, other):
    logger.debug("GEMS_KUNLUNXIN LDEXP")
    return _ldexp_impl(self, other)


def ldexp_out(self, other, *, out):
    logger.debug("GEMS_KUNLUNXIN LDEXP_OUT")
    return _ldexp_impl(self, other, out)


def _ldexp_result_dtype(self, other):
    self_dtype = (
        torch.float32
        if self.dtype == torch.bool
        or (not self.is_floating_point() and not self.is_complex())
        else self.dtype
    )
    return torch.promote_types(self_dtype, other.dtype)


def _write_complex(self, real, imag):
    physical = self.conj() if self.is_conj() else self
    output = torch.view_as_real(physical)  # allow-torch-compute
    copy_(output[..., 0], real)
    copy_(output[..., 1], neg(imag) if self.is_conj() else imag)


def _ldexp_complex_(self, other):
    promoted = _ldexp_result_dtype(self, other)
    real_dtype = torch.float64 if promoted == torch.complex128 else torch.float32
    xr, xi = _complex_parts(self, real_dtype)
    yr, yi = _complex_parts(other, real_dtype)
    func = (
        ldexp_inplace_complex_fp64_func
        if promoted == torch.complex128
        else ldexp_inplace_complex_func
    )
    real, imag = func(xr, xi, yr, yi)
    _write_complex(self, real, imag)


def _ldexp_complex_integral_(self, other):
    real_dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
    xr, xi = _complex_parts(self, real_dtype)
    func = (
        ldexp_inplace_complex_integral_fp64_func
        if self.dtype == torch.complex128
        else ldexp_inplace_complex_integral_func
    )
    real, imag = func(xr, xi, other)
    _write_complex(self, real, imag)


def ldexp_(self, other):
    logger.debug("GEMS_KUNLUNXIN LDEXP_")
    if has_internal_overlapping(self) == MemOverlap.Yes:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to tensor "
            "refers to a single memory location"
        )
    if (
        self.device == other.device
        and self.untyped_storage().nbytes() > 0
        and self.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()
        and not _is_exact_alias(self, other)
    ):
        temporary = torch.empty_like(other)
        if other.is_complex():
            real, imag = _complex_parts(
                other,
                torch.float64 if other.dtype == torch.complex128 else torch.float32,
            )
            _write_complex(temporary, real, imag)
        else:
            copy_(temporary, other)
        other = temporary

    result_dtype = _ldexp_result_dtype(self, other)
    if not torch.can_cast(result_dtype, self.dtype):  # allow-torch-compute
        raise RuntimeError(
            f"result type {result_dtype} can't be cast to the desired output "
            f"type {self.dtype}"
        )

    output_shape = torch.broadcast_shapes(  # allow-torch-compute
        self.shape, other.shape
    )
    if output_shape != self.shape:
        raise RuntimeError(
            f"output with shape {self.shape} doesn't match the broadcast shape "
            f"{output_shape}"
        )

    integral_exponent = not other.is_floating_point() and not other.is_complex()
    if self.is_complex():
        if integral_exponent:
            _ldexp_complex_integral_(self, other)
        else:
            _ldexp_complex_(self, other)
    else:
        if (
            integral_exponent
            and self.dtype in _FLAT_DTYPES
            and self.numel() >= _FLAT_MIN_ELEMENTS
            and _flat_fast_ok(self, other, None, self.dtype)
        ):
            # Lean 1-D in-place path for the half-dtype dense case.
            _run_flat_ldexp(self, other, self)
            return self
        if integral_exponent:
            func = (
                ldexp_inplace_integral_fp64_func
                if result_dtype == torch.float64
                else ldexp_inplace_integral_func
            )
        else:
            func = (
                ldexp_inplace_fp64_func
                if result_dtype == torch.float64
                else ldexp_inplace_func
            )
        if integral_exponent:
            func(self, other, _WRAP_INTEGER_EXPONENT, out0=self)
        else:
            func(self, other, out0=self)
    return self
