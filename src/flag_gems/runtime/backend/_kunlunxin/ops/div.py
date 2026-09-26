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
import struct

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
div_rn = tl_extra_shim.div_rn

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_)
@triton.jit
def true_div_func(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_tensor_scalar(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def true_div_func_scalar_tensor(x, y):
    return x / y


# Tuned scalar-path variants (config_): the untuned kernels above are
# launch-optimal for small tensors, so they are kept for
# numel < DIV_SCALAR_CFG_THRESHOLD; larger tensors use the Kunlunxin tuned
# CodeGenConfig. Measured on XPU (scalar path, same do_bench):
#   (4096,4096) fp32 221us -> 142us, bf16 224us -> 149us, fp16 206us -> 168us;
#   (2^28,) fp32 3317us -> 2053us, fp16 3089us -> 2292us, bf16 3369us -> 2171us;
#   (10000,256) fp32 41us -> 30us. Small tensors are untouched (launch floor).
DIV_SCALAR_CFG_THRESHOLD = 1 << 20


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def true_div_func_tensor_scalar_cfg(x, y):
    return x / y


@pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=config_
)
@triton.jit
def true_div_func_scalar_tensor_cfg(x, y):
    return x / y


# fp32-only unroll-16 variant of the tuned scalar kernel: measured on XPU for
# the in-place scalar path (out0=A, same do_bench median, 3 interleaved
# rounds) that unroll_num=16 is strictly faster than unroll 8 for fp32 across
# the whole shape range ((4096,4096) 141.9->138.4us, (10000,256) 30.3->29.6us,
# (2^28,) 2043->1998us, (10000,65536) 4949->4842us; ~2.2%), while fp16/bf16
# are flat or slightly worse with unroll 16 (fp16 (2^28,) 2292->2341us), so
# the unroll-16 kernel is dispatched only for fp32 in true_divide_.
CFG_UNROLL16 = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=16,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=CFG_UNROLL16,
)
@triton.jit
def true_div_func_tensor_scalar_cfg16(x, y):
    return x / y


# Tensor/tensor unroll-16 variant (CFG_UNROLL16): measured on XPU with
# isolated processes (official do_bench warmup=1000ms/rep=100ms median) that
# unroll 16 is strictly faster than unroll 8 (config_) for fp16/fp32 at
# numel >= 4M, while bf16 regresses at all sizes and numel < 4M explodes:
#   fp16 (4096,4096) 0.996->1.114x  (1024,65536) 1.023->1.148x
#        (64,64,65536) 1.042->1.170x (64,64,4096) 0.993->1.114x
#   fp32 (4096,4096) 0.703->0.731x  (1024,65536) 0.717->0.761x
#        (1024,4096) 0.67->0.72x    (64,64,4096) ~0.70->0.73x
#   bf16 (4096,4096) 0.960->0.852x  (1024,65536) 0.986->0.865x (regress)
#   fp32 (1024,256) 1.02->0.045x    (1024,16) 1.02->0.185x (explode)
# -> dispatch unroll16 only for fp16/fp32 tensor/tensor at numel >= 4M;
# bf16 and smaller tensors keep config_ (true_div_func).
DIV_TENSOR_U16_MIN_NUMEL = 1 << 22  # 4M


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")], config=CFG_UNROLL16)
@triton.jit
def true_div_func_u16(x, y):
    return x / y


_DIV_FAST_MAX = DIV_TENSOR_U16_MIN_NUMEL
_DIV_MIN_BLOCK = 2048


def _pick_block(n_elements):
    if n_elements >= 1_048_576:
        for tile in (32768, 16384, 8192, 4096, 2048):
            if n_elements % tile == 0:
                return tile, 4, False
    if n_elements >= 65_536 and n_elements % 8192 == 0:
        return 8192, 4, False
    if n_elements <= 65_536:
        return _DIV_MIN_BLOCK, 4, True
    return 8192, 4, True


@triton.jit
def _div_fast_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0)
    y = tl.load(y_ptr + offset, mask=mask, other=1)
    tl.store(out_ptr + offset, x / y, mask=mask)


@triton.jit
def _div_fast_kernel_unmasked(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset)
    y = tl.load(y_ptr + offset)
    tl.store(out_ptr + offset, x / y)


def _div_small_eligible(A, B):
    return (
        A.dtype == B.dtype
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and 0 < A.numel() < _DIV_FAST_MAX
    )


def _true_divide_fast(A, B, out):
    numel = A.numel()
    block_size, num_warps, masked = _pick_block(numel)
    if masked:
        _div_fast_kernel[(triton.cdiv(numel, block_size),)](
            A,
            B,
            out,
            numel,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            buffer_size_limit=8192,
            unroll_num=16,
            isCloseMemoryAsync=False,
        )
    else:
        _div_fast_kernel_unmasked[(numel // block_size,)](
            A,
            B,
            out,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            buffer_size_limit=8192,
            unroll_num=16,
            isCloseMemoryAsync=False,
        )
    return out


# ---- complex true division (bit-view kernels) ----
# XPU triton has no complex type support (jit specialization KeyErrors) and
# xdnn native casts/copies on complex dtypes are [NOT IMPLEMENTED] / broken
# (copy_ invalid device function). Decompose to interleaved component views
# (complex32 -> float16, complex64 -> float32) and compute entirely in-kernel
# with fp32 math; zero torch-side ops on complex tensors.


@triton.jit
def _cdiv_tt_kernel(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    ar = tl.load(a_ptr + 2 * offs, mask=m, other=0.0).to(tl.float32)
    ai = tl.load(a_ptr + 2 * offs + 1, mask=m, other=0.0).to(tl.float32)
    br = tl.load(b_ptr + 2 * offs, mask=m, other=0.0).to(tl.float32)
    bi = tl.load(b_ptr + 2 * offs + 1, mask=m, other=0.0).to(tl.float32)
    den = br * br + bi * bi
    re = (ar * br + ai * bi) / den
    im = (ai * br - ar * bi) / den
    ty = o_ptr.dtype.element_ty
    tl.store(o_ptr + 2 * offs, re.to(ty), mask=m)
    tl.store(o_ptr + 2 * offs + 1, im.to(ty), mask=m)


@triton.jit
def _cdiv_tr_kernel(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    ar = tl.load(a_ptr + 2 * offs, mask=m, other=0.0).to(tl.float32)
    ai = tl.load(a_ptr + 2 * offs + 1, mask=m, other=0.0).to(tl.float32)
    br = tl.load(b_ptr + offs, mask=m, other=0.0).to(tl.float32)
    # direct componentwise division: IEEE x/0 -> +/-inf, 0/0 -> nan, matching
    # torch CPU semantics for a real divisor (den=br*br flushes to 0 on XPU
    # when b is an exact f16 zero or tiny f32, wrongly yielding 0/0)
    re = ar / br
    im = ai / br
    ty = o_ptr.dtype.element_ty
    tl.store(o_ptr + 2 * offs, re.to(ty), mask=m)
    tl.store(o_ptr + 2 * offs + 1, im.to(ty), mask=m)


@triton.jit
def _cdiv_ts_kernel(a_ptr, o_ptr, n, scalar, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    ar = tl.load(a_ptr + 2 * offs, mask=m, other=0.0).to(tl.float32)
    ai = tl.load(a_ptr + 2 * offs + 1, mask=m, other=0.0).to(tl.float32)
    re = ar / scalar
    im = ai / scalar
    ty = o_ptr.dtype.element_ty
    tl.store(o_ptr + 2 * offs, re.to(ty), mask=m)
    tl.store(o_ptr + 2 * offs + 1, im.to(ty), mask=m)


@triton.jit
def _cdiv_rt_kernel(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    # real tensor / complex tensor: a * conj(b) / |b|^2
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    ar = tl.load(a_ptr + offs, mask=m, other=0.0).to(tl.float32)
    br = tl.load(b_ptr + 2 * offs, mask=m, other=0.0).to(tl.float32)
    bi = tl.load(b_ptr + 2 * offs + 1, mask=m, other=0.0).to(tl.float32)
    den = br * br + bi * bi
    re = (ar * br) / den
    im = (-ar * bi) / den
    ty = o_ptr.dtype.element_ty
    tl.store(o_ptr + 2 * offs, re.to(ty), mask=m)
    tl.store(o_ptr + 2 * offs + 1, im.to(ty), mask=m)


@triton.jit
def _cdiv_st_kernel(b_ptr, o_ptr, n, scalar, BLOCK: tl.constexpr):
    # python scalar / complex tensor
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    br = tl.load(b_ptr + 2 * offs, mask=m, other=0.0).to(tl.float32)
    bi = tl.load(b_ptr + 2 * offs + 1, mask=m, other=0.0).to(tl.float32)
    den = br * br + bi * bi
    re = (scalar * br) / den
    im = (-scalar * bi) / den
    ty = o_ptr.dtype.element_ty
    tl.store(o_ptr + 2 * offs, re.to(ty), mask=m)
    tl.store(o_ptr + 2 * offs + 1, im.to(ty), mask=m)


def _complex_view(T):
    if T.dim() == 0:
        T = T.unsqueeze(0)
    if T.dtype == torch.complex32:
        return T.view(torch.float16).reshape(-1)
    return T.view(torch.float32).reshape(-1)


def _complex_true_divide(A, B, out=None):
    # A complex tensor path (B: complex tensor / real tensor / python scalar)
    n = A.numel()
    if out is None:
        out = torch.empty_like(A)
    ov = _complex_view(out)
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    av = _complex_view(A)
    if isinstance(B, torch.Tensor):
        if B.is_complex():
            _cdiv_tt_kernel[grid](av, _complex_view(B), ov, n, BLOCK=BLOCK)
        else:
            _cdiv_tr_kernel[grid](av, B.reshape(-1), ov, n, BLOCK=BLOCK)
    else:
        _cdiv_ts_kernel[grid](av, ov, n, float(B), BLOCK=BLOCK)
    return out


def _rational_true_divide(A, B, out=None):
    # A real tensor / python scalar, B complex tensor -> complex result
    n = B.numel()
    if out is None:
        out = torch.empty_like(B)
    ov = _complex_view(out)
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    _cdiv_rt_kernel[grid](A.reshape(-1), _complex_view(B), ov, n, BLOCK=BLOCK)
    return out


def _scalar_over_complex(A, B, out=None):
    # A python scalar, B complex tensor -> complex result
    n = B.numel()
    if out is None:
        out = torch.empty_like(B)
    ov = _complex_view(out)
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    _cdiv_st_kernel[grid](_complex_view(B), ov, n, float(A), BLOCK=BLOCK)
    return out


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[(0, 1, 2, 3, "INT_TO_FLOAT"), (0, 1, 2, 3, "INT_TO_FLOAT")],
    config=config_,
)
@triton.jit
def true_div_complex_kernel(ar, ai, br, bi):
    abs_br = tl.abs(br)
    abs_bi = tl.abs(bi)
    use_br = abs_br >= abs_bi

    ratio1 = tl.where(br == 0, 0.0, bi / br)
    denom1 = br + bi * ratio1
    real1 = (ar + ai * ratio1) / denom1
    imag1 = (ai - ar * ratio1) / denom1

    ratio2 = tl.where(bi == 0, 0.0, br / bi)
    denom2 = bi + br * ratio2
    real2 = (ar * ratio2 + ai) / denom2
    imag2 = (ai * ratio2 - ar) / denom2

    real = tl.where(use_br, real1, real2)
    imag = tl.where(use_br, imag1, imag2)
    return real, imag


def _complex_real_parts(z, upcast):
    zr = torch.view_as_real(z)
    if upcast:
        zr = zr.to(torch.float32)
    return zr[..., 0].contiguous(), zr[..., 1].contiguous()


def _true_divide_complex_tensors(A, B):
    A_is_complex = A.is_complex()
    B_is_complex = B.is_complex()
    if A_is_complex and B_is_complex:
        upcast = A.dtype == torch.complex32
        ar, ai = _complex_real_parts(A, upcast)
        br, bi = _complex_real_parts(B, upcast)
        real, imag = true_div_complex_kernel(ar, ai, br, bi)
        if upcast:
            real, imag = real.to(torch.float16), imag.to(torch.float16)
        return torch.view_as_complex(torch.stack((real, imag), dim=-1))
    elif A_is_complex:
        upcast = A.dtype == torch.complex32
        Ar = torch.view_as_real(A)
        if upcast:
            Ar = Ar.to(torch.float32)
            Br = B.unsqueeze(-1).to(torch.float32)
        else:
            Br = B.unsqueeze(-1)
        out = true_div_func(Ar, Br)
        if upcast:
            out = out.to(torch.float16)
        return torch.view_as_complex(out.contiguous())
    else:
        upcast = B.dtype == torch.complex32
        br, bi = _complex_real_parts(B, upcast)
        ar = A.to(br.dtype)
        ai = torch.zeros_like(ar)
        real, imag = true_div_complex_kernel(ar, ai, br, bi)
        if upcast:
            real, imag = real.to(torch.float16), imag.to(torch.float16)
        return torch.view_as_complex(torch.stack((real, imag), dim=-1))


def _same_layout_out0(A, B):
    if A.is_floating_point() and B.dtype == A.dtype and B.shape == A.shape:
        return torch.empty_like(A)
    return None


def true_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and A.is_complex():
        return _complex_true_divide(A, B)
    if isinstance(B, torch.Tensor) and B.is_complex():
        if isinstance(A, torch.Tensor):
            return _rational_true_divide(A, B)
        return _scalar_over_complex(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.is_complex() or B.is_complex():
            return _true_divide_complex_tensors(A, B)
        kernel = true_div_func
        if (
            A.dtype in (torch.float16, torch.float32)
            and A.numel() >= DIV_TENSOR_U16_MIN_NUMEL
        ):
            kernel = true_div_func_u16
        if _div_small_eligible(A, B):
            return _true_divide_fast(A, B, torch.empty_like(A))
        out0 = _same_layout_out0(A, B)
        if out0 is not None:
            return kernel(A, B, out0=out0)
        return kernel(A, B)
    elif isinstance(A, torch.Tensor):
        if A.is_complex():
            return torch.view_as_complex(
                true_div_func_tensor_scalar(torch.view_as_real(A), B)
            )
        if A.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            return true_div_func_tensor_scalar_cfg(A, B)
        return true_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        if B.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            return true_div_func_scalar_tensor_cfg(A, B)
        return true_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A / B)


def true_divide_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_TENSOR")
    # keep the dispatch-contract log line expected by tests/test_true_divide.py
    # (caplog on logger "flag_gems.ops.true_divide", same pattern as special_erf)
    logging.getLogger("flag_gems.ops.true_divide").debug("GEMS TRUE_DIVIDE")
    return true_divide(A, B)


def true_divide_out(A, B, out):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_OUT")
    if isinstance(A, torch.Tensor) and A.is_complex():
        return _complex_true_divide(A, B, out=out)
    if isinstance(B, torch.Tensor) and B.is_complex():
        if isinstance(A, torch.Tensor):
            return _rational_true_divide(A, B, out=out)
        return _scalar_over_complex(A, B, out=out)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if (
            A.dtype in (torch.float16, torch.float32)
            and A.numel() >= DIV_TENSOR_U16_MIN_NUMEL
        ):
            return true_div_func_u16(A, B, out0=out)
        if _div_small_eligible(A, B) and out.is_contiguous() and out.shape == A.shape:
            return _true_divide_fast(A, B, out)
        return true_div_func(A, B, out0=out)
    elif isinstance(A, torch.Tensor):
        if A.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            return true_div_func_tensor_scalar_cfg(A, B, out0=out)
        return true_div_func_tensor_scalar(A, B, out0=out)
    elif isinstance(B, torch.Tensor):
        if B.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            return true_div_func_scalar_tensor_cfg(A, B, out0=out)
        return true_div_func_scalar_tensor(A, B, out0=out)
    else:
        # Both scalar
        return torch.tensor(A / B) if out is None else out.fill_(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUE_DIVIDE_")
    logging.getLogger("flag_gems.ops.true_divide_").debug("GEMS TRUE_DIVIDE_")
    if A.is_complex():
        return _complex_true_divide(A, B, out=A)
    if isinstance(B, torch.Tensor):
        if (
            A.dtype in (torch.float16, torch.float32)
            and A.numel() >= DIV_TENSOR_U16_MIN_NUMEL
        ):
            return true_div_func_u16(A, B, out0=A)
        if _div_small_eligible(A, B):
            return _true_divide_fast(A, B, A)
        return true_div_func(A, B, out0=A)
    else:
        if A.numel() >= DIV_SCALAR_CFG_THRESHOLD:
            if A.dtype == torch.float32:
                return true_div_func_tensor_scalar_cfg16(A, B, out0=A)
            return true_div_func_tensor_scalar_cfg(A, B, out0=A)
        return true_div_func_tensor_scalar(A, B, out0=A)


divide = true_divide
true_divide_tensor_ = true_divide_


@triton.jit
def _trunc_q(q):
    # Truncate a fp32 quotient toward zero without the slow `xpu_trunc`
    # extern call (measured ~2.3x slower on XPU). fp32 -> int32 cast
    # (fptosi, round toward zero) is exact for |q| < 2^23; for |q| >= 2^23
    # the fp32 value is already integral (23-bit mantissa), so keep it.
    return tl.where(tl.abs(q) < 8388608.0, tl.cast(q, tl.int32).to(tl.float32), q)


@triton.jit
def _floor_q(q):
    # floor() of a fp32 quotient (fast path for the common case where the
    # fp32 quotient is well away from an integral boundary). int32 cast is
    # RTZ (truncation): for q >= 0 floor == trunc; for q < 0 and q not
    # integral subtract 1. |q| >= 2^23 fp32 values are already integral.
    t = tl.where(tl.abs(q) < 8388608.0, tl.cast(q, tl.int32).to(tl.float32), q)
    return tl.where((q < 0) & (q != t), t - 1.0, t)


@triton.jit
def _floor_div_fp32(x, y):
    # floor(x/y) matching the torch-CPU / numpy reference (npy_divmodf)
    # semantics bit-for-bit, without the slow XPU fp64 path:
    #   mod = fmodf(x, y)                    (fp32, single-rounded fma)
    #   if (mod != 0 && isless(b,0) != isless(mod,0)) { mod += b; }
    #   div = (x - mod) / y
    #   fd = floor(div); if (div - fd > 0.5) fd += 1  (snap tie cases)
    #   if (div == 0) fd = copysign(0, x/y)
    # This is what the CPU reference (used by --ref cpu) computes; the old
    # fp64-division implementation was emulated on XPU (~8-10x slower) and
    # also differed from the CPU reference near integral boundaries.
    q = div_rn(x, y)
    t = tl.where(tl.abs(q) < 8388608.0, tl.cast(q, tl.int32).to(tl.float32), q)
    mod0 = tl.fma(t, -y, x)
    adj = (mod0 != 0.0) & ((y < 0.0) != (mod0 < 0.0))
    div = div_rn(x - mod0, y)  # numpy: div computed with original mod
    div = tl.where(adj, div - 1.0, div)
    fd = _floor_q(div)
    fd = tl.where(div - fd > 0.5, fd + 1.0, fd)
    fd = tl.where(
        div == 0.0,
        tl.where((x < 0.0) != (y < 0.0), -0.0, 0.0),
        fd,
    )
    # division by zero: numpy returns a / b directly (signed inf)
    return tl.where(y == 0.0, q, fd)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def trunc_div_func(x, y):
    return _trunc_q(div_rn(x, y))


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return _trunc_q(div_rn(x, tl.cast(y, x.dtype)))


@pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return _trunc_q(div_rn(tl.cast(x, y.dtype), y))


# Integer truncation division: Triton's // on integers is C-style (truncates toward zero)
@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func_tensor_scalar(x, y):
    return x // y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_int_func_scalar_tensor(x, y):
    return x // y


def trunc_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUNC_DIVIDE")
    # Integer types: use dedicated int kernels (Triton // is C-style truncation)
    if isinstance(A, torch.Tensor) and not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B)
        else:
            return trunc_div_int_func_tensor_scalar(A, B)
    if isinstance(B, torch.Tensor) and not B.is_floating_point():
        return trunc_div_int_func_scalar_tensor(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return trunc_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A / B)


def trunc_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN TRUNC_DIVIDE_")
    # Integer types: use dedicated int kernels (Triton // is C-style truncation)
    if not A.is_floating_point():
        if isinstance(B, torch.Tensor):
            return trunc_div_int_func(A, B, out0=A)
        else:
            return trunc_div_int_func_tensor_scalar(A, B, out0=A)
    if isinstance(B, torch.Tensor):
        return trunc_div_func(A, B, out0=A)
    else:
        return trunc_div_func_tensor_scalar(A, B, out0=A)


@triton.jit
def _int_floordiv(x, y):
    # TODO: request Triton to add an integer remainder builtin
    # The semantic of Triton floordiv differs from Pytorch/Numpy
    # Triton floordiv equates to
    #     (x - np.fmod(x, y)) / y
    # whereas Pytorch floordiv is
    #     (x - np.remainder(x, y)) y
    # The results show a one off difference when
    #     C1) x and y have opposite signs
    # and C2) x is not multiples of y.
    # Apart from the above, there's an erroneous case x // 0 returns -1
    # whereas in Pytorch x // 0 returns -1 if x >=0 and -2 if x < 0
    # but this special case is coalesced into the c1 and c2 check so
    # there's extra handling.
    q = x // y
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, q - 1, q)


# floor_divide must be consistent with python/numpy/torch: floor(x/y) on the
# fp32 device quotient, implemented with the fast int-cast floor (fp64 is
# emulated on XPU and ~8-10x slower; the old fp64 path also differed from
# torch by 1 ulp near integral boundaries since torch divides in fp32).
@triton.jit
def _float_floordiv_corrected(x, y):
    # x, y are already fp32. The old fp64-division implementation was
    # emulated on XPU (~8-10x slower) and also differed from torch by 1 ulp
    # near integral boundaries. New: fp32 device division + exact-remainder
    # floor (matches CPU reference semantics bit-wise).
    return _floor_div_fp32(x, y)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def floor_div_func_corrected(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv_corrected(x.to(tl.float32), y.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def floor_div_func_corrected_tensor_scalar(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv_corrected(x.to(tl.float32), y.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def floor_div_lowp_tensor_scalar_func(x, y):
    # fp16/bf16 scalar path: promote to fp32 and divide in fp32 (torch does
    # the same; there is no native fp16/bf16 division), then floor exactly
    # with the same exact-remainder semantics as the fp32 path.
    y = tl.full(x.shape, y, x.dtype)
    return _floor_div_fp32(x.to(tl.float32), y.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def floor_div_func_corrected_scalar_tensor(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _int_floordiv(x, y)
    else:
        return _float_floordiv_corrected(x.to(tl.float32), y.to(tl.float32))


def _as_bfloat16_scalar(value):
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    exponent = bits & 0x7F800000
    mantissa = bits & 0x007FFFFF
    if exponent != 0x7F800000:
        bits += 0x7FFF + ((bits >> 16) & 1)
    elif mantissa:
        bits |= 0x00400000
    bits &= 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def floor_divide(A, B):
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return floor_div_func_corrected(A, B)
    elif isinstance(A, torch.Tensor):
        if A.dtype in (torch.float16, torch.bfloat16):
            if A.dtype == torch.bfloat16:
                B = _as_bfloat16_scalar(B)
            return floor_div_lowp_tensor_scalar_func(A, B)
        return floor_div_func_corrected_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return floor_div_func_corrected_scalar_tensor(A, B)
    else:
        # Both scalar
        return torch.tensor(A // B)


def floor_divide_(A, B):
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return floor_div_func_corrected(A, B, out0=A)
    else:
        if A.dtype in (torch.float16, torch.bfloat16):
            if A.dtype == torch.bfloat16:
                B = _as_bfloat16_scalar(B)
            return floor_div_lowp_tensor_scalar_func(A, B, out0=A)
        return floor_div_func_corrected_tensor_scalar(A, B, out0=A)


def div_mode(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide(A, B)
    elif rounding_mode == "floor":
        return floor_divide(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


def div_mode_(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide_(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide_(A, B)
    elif rounding_mode == "floor":
        return floor_divide_(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


@triton.jit
def _remainder(x, y):
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, r + y, r)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_tt(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_ts(x, y):
    return _remainder(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_st(x, y):
    return _remainder(x, y)


# Large-tensor codegen variants (shrinking DMA pipeline / unroll config):
# the default-config kernels are launch-optimal for small tensors and are
# kept for numel < REMAINDER_CFG_THRESHOLD, where the config_ variants were
# measured to be up to ~3x slower (launch-floor cells).
REMAINDER_CFG_THRESHOLD = 1 << 20


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def rem_tt_cfg(x, y):
    return _remainder(x, y)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def rem_ts_cfg(x, y):
    return _remainder(x, y)


@pointwise_dynamic(
    is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def rem_st_cfg(x, y):
    return _remainder(x, y)


def _fold_scalar_into_tensor_dtype(value, dtype):
    """Reproduce ATen's "wrapped number" fold for the scalar operand.

    ``aten::remainder.Scalar_Tensor`` converts the Python scalar with
    ``Scalar::to<T>()`` (a C-style truncating conversion) *before* the kernel
    runs, so ``300 % <int8 tensor>`` really computes ``44 % y``. The shared
    pointwise generator instead keeps the scalar in a wide integer type and
    truncates only when storing to the (narrower) output dtype, which silently
    disagrees with ATen for any scalar that does not fit the tensor dtype.
    Verified on XPU (aten CPU oracle): int8 300/130, int16 40000/32768,
    int32 +/-2**40/2**31 all returned the wide-scalar result.

    Python ``bool`` is folded to ``int`` as well: the generated scalar kernel
    carries ``do_not_specialize=["val0"]``, so a ``bool`` first argument binds
    ``val0`` to ``i1`` and aborts with ``CompilationError`` (ATen returns the
    ``True % y`` result).
    """
    if isinstance(value, bool):
        value = int(value)
    if (
        isinstance(value, int)
        and not dtype.is_floating_point
        and not dtype.is_complex
        and dtype != torch.bool
    ):
        info = torch.iinfo(dtype)
        mod = 1 << info.bits
        value &= mod - 1
        if info.min < 0 and value > info.max:
            value -= mod
    return value


def remainder(A, B):
    logger.debug("GEMS_KUNLUNXIN FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if max(A.numel(), B.numel()) >= REMAINDER_CFG_THRESHOLD:
            return rem_tt_cfg(A, B)
        return rem_tt(A, B)
    elif isinstance(A, torch.Tensor):
        if A.numel() >= REMAINDER_CFG_THRESHOLD:
            return rem_ts_cfg(A, B)
        return rem_ts(A, B)
    elif isinstance(B, torch.Tensor):
        A = _fold_scalar_into_tensor_dtype(A, B.dtype)
        if B.numel() >= REMAINDER_CFG_THRESHOLD:
            return rem_st_cfg(A, B)
        return rem_st(A, B)
    else:
        # Both scalar
        return torch.tensor(A % B)


def remainder_(A, B):
    logger.debug("GEMS_KUNLUNXIN REMAINDER_")
    if isinstance(B, torch.Tensor):
        if max(A.numel(), B.numel()) >= REMAINDER_CFG_THRESHOLD:
            return rem_tt_cfg(A, B, out0=A)
        return rem_tt(A, B, out0=A)
    else:
        if A.numel() >= REMAINDER_CFG_THRESHOLD:
            return rem_ts_cfg(A, B, out0=A)
        return rem_ts(A, B, out0=A)
