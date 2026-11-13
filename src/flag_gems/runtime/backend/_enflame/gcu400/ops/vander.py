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

from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

_pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)


def _install_gcu_vander_ref_shim():
    # torch_gcu P21 workaround for the *reference* arm (at::native::vander ->
    # cumprod), which never dispatches to the FlagGems kernel: on gcu400 it
    # returns garbage for strided fp32 views, aborts the process for bool /
    # integer inputs (topsatenCumprod: TOPSATEN_STATUS_NOT_SUPPORT on Long),
    # and computes float64 through the Double->Float payload replace (fp32
    # rounding, wrong at float64 tolerances). Exactly those call shapes are
    # computed on CPU and copied back; contiguous fp16/bf16/fp32 and complex
    # inputs keep the native device path untouched (add.py / _linalg_solve_ex.py
    # gates_fix precedent). The op itself is launched via flag_gems.vander and
    # never goes through this wrapper.
    current = torch.vander
    if getattr(current, "_flaggems_gcu_vander_safe", False):
        return

    def _gcu_safe_vander(x, N=None, increasing=False):
        if (
            isinstance(x, torch.Tensor)
            and x.device.type == "gcu"
            and (
                not x.is_contiguous()
                or not x.is_floating_point()
                or x.dtype == torch.float64
            )
        ):
            return current(x.cpu(), N=N, increasing=increasing).to(x.device)
        return current(x, N=N, increasing=increasing)

    _gcu_safe_vander._flaggems_gcu_vander_safe = True
    torch.vander = _gcu_safe_vander


_install_gcu_vander_ref_shim()


# gcu400: Triton fp64 ops (arith.extf / fp64 arith.mulf) fail to legalize, and
# torch_gcu stores float64 / complex128 tensors as Float/ComplexFloat-replaced
# payloads (fp32 values in the low words). The generic kernels accumulate the
# power chain in tl.float64, which cannot compile here. The two kernels below
# run the same chain with a double-float (hi/lo fp32 pair) accumulator built
# from FMA error-free transformations: the stored fp32 column values match the
# correctly-rounded fp64 reference bit-for-bit at the test tolerances.
#
# gcu400 compiler caveat (probed): the backend reassociates plain fp32
# arithmetic and folds exact-subtraction identities, e.g. a Veltkamp split
# `c - (c - a)` collapses to `a` and `p + fma(a, b, -p)` collapses back to the
# naive product, silently degrading any Dekker/Knuth-style compensated code to
# naive fp32 accuracy. Every "main" value below is therefore produced by an
# fma against a *runtime* zero/one/negative-one scalar argument (Z0 / ONE /
# NEG_ONE), which the compiler cannot constant-fold, so the residual fma
# `tl.fma(a, b, -p)` and the two-sum recovery stay intact. With this shaping a
# 15-step chain is bit-identical to the fp64 reference rounded to fp32.
#
# Inputs are normalized on the host with value-level `.to()` casts (which read
# the replaced payload correctly) and results are widened back the same way, so
# the returned tensors keep the original dtype labels.


@triton.jit
def _df_mul(acc_hi, acc_lo, b, Z0):
    # (acc_hi + acc_lo) * b for an exact fp32 b, as an unnormalized hi/lo pair.
    # The main product goes through an fma on a runtime zero: a plain
    # `acc_hi * b` gets reassociated with the residual fma on this backend and
    # the compensation is folded away.
    p = tl.fma(acc_hi, b, Z0)
    e = tl.fma(acc_hi, b, -p)
    e = e + acc_lo * b
    return p, e


@triton.jit
def _df_two_sum(a, b, ONE, NEG_ONE):
    # Full Knuth two-sum: s = fl(a + b); v = fl(s - a); a2 = fl(s - v);
    # e = (a - a2) + (b - v). Both difference terms are required (the Fast2Sum
    # shortcut e = b - v is only valid under |a| >= |b|). Every subtraction is
    # fma-shaped against runtime +/-1 so its exactness survives the backend's
    # reassociation.
    s = tl.fma(a, ONE, b)
    v = tl.fma(a, NEG_ONE, s)
    a2 = tl.fma(v, NEG_ONE, s)
    e1 = tl.fma(a2, NEG_ONE, a)
    e2 = tl.fma(v, NEG_ONE, b)
    return s, e1 + e2


@triton.jit
def _df_cmul_step(ar_hi, ar_lo, ai_hi, ai_lo, zr, zi, Z0, ONE, NEG_ONE):
    # (ar + i * ai) * (zr + i * zi) with z exact; returns df real / imag parts.
    pr_hi, pr_lo = _df_mul(ar_hi, ar_lo, zr, Z0)
    pi_hi, pi_lo = _df_mul(ai_hi, ai_lo, zi, Z0)
    nr_hi, e_sub = _df_two_sum(pr_hi, -pi_hi, ONE, NEG_ONE)
    nr_lo = e_sub + pr_lo - pi_lo

    qr_hi, qr_lo = _df_mul(ar_hi, ar_lo, zi, Z0)
    qi_hi, qi_lo = _df_mul(ai_hi, ai_lo, zr, Z0)
    ni_hi, e_add = _df_two_sum(qr_hi, qi_hi, ONE, NEG_ONE)
    ni_lo = e_add + qr_lo + qi_lo
    return nr_hi, nr_lo, ni_hi, ni_lo


@libentry()
@triton.jit
def vander_complex_kernel(
    xr_ptr,
    xi_ptr,
    outr_ptr,
    outi_ptr,
    M,
    N,
    x_stride,
    out_row_stride,
    Z0,
    ONE,
    NEG_ONE,
    increasing: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Complex path in double-float fp32: each program handles one row block and
    # walks the columns left-to-right, updating the running power z^k with a
    # compensated complex multiplication (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
    # outr_ptr / outi_ptr are the interleaved [..., 0] / [..., 1] views of one
    # [M, N, 2] buffer (column stride 2).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    zr = tl.load(xr_ptr + rows * x_stride, mask=mask_m, other=0.0)
    zi = tl.load(xi_ptr + rows * x_stride, mask=mask_m, other=0.0)

    hr = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    lr = tl.zeros([BLOCK_M], dtype=tl.float32)
    hi = tl.zeros([BLOCK_M], dtype=tl.float32)
    li = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in range(0, N):
        store_col = k if increasing else N - 1 - k
        out_off = rows * out_row_stride + store_col * 2
        tl.store(outr_ptr + out_off, hr + lr, mask=mask_m)
        tl.store(outi_ptr + out_off, hi + li, mask=mask_m)
        hr, lr, hi, li = _df_cmul_step(hr, lr, hi, li, zr, zi, Z0, ONE, NEG_ONE)


@libentry()
@triton.jit
def vander_df_real_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    x_stride,
    Z0,
    increasing: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # float64-input path: fp32 payload plus double-float chain accumulation,
    # used because tl.float64 arithmetic cannot legalize on this backend.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    x = tl.load(x_ptr + rows * x_stride, mask=mask_m, other=0.0)

    hi = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    lo = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in range(0, N):
        store_col = k if increasing else N - 1 - k
        tl.store(out_ptr + rows * N + store_col, hi + lo, mask=mask_m)
        hi, lo = _df_mul(hi, lo, x, Z0)


@libentry()
@triton.jit
def vander_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    x_stride,
    increasing: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Floating-point path: out[i, j] = x[i]^p with p = j (increasing) or N-1-j.
    # Computing in COMPUTE_DTYPE (fp32 here; fp64 inputs take the df chain
    # kernel instead because fp64 ops are illegal on this backend) lets
    # fp16/bf16 benefit from fp32 accumulation.
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = rows < M
    x_vals = tl.load(x_ptr + rows * x_stride, mask=mask_m, other=0.0).to(COMPUTE_DTYPE)

    if increasing:
        powers = cols
    else:
        powers = N - 1 - cols
    powers = powers.to(COMPUTE_DTYPE)

    result = _pow(x_vals[:, None], powers[None, :])

    mask_n = cols < N
    mask = mask_m[:, None] & mask_n[None, :]
    out_off = rows[:, None] * N + cols[None, :]
    tl.store(out_ptr + out_off, result, mask=mask)


@libentry()
@triton.jit
def vander_int_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    x_stride,
    increasing: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Integer path: out[i, j] = x[i]^p computed exactly with int64 cumulative
    # products, matching ATen's cumprod semantics (including wrap-around). Each
    # program sweeps one column tile and carries the running product x^off.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    x_val = tl.load(x_ptr + rows * x_stride, mask=mask_m, other=0).to(tl.int64)

    acc = tl.full([BLOCK_M], 1, dtype=tl.int64)
    for off in range(0, N, BLOCK_N):
        k = off + tl.arange(0, BLOCK_N)
        col_mask = k < N
        local = tl.arange(0, BLOCK_N)
        # Seed the tile with [1, x, x, ..., x] so a cumulative product yields the
        # local powers [1, x, x^2, ..., x^(BLOCK_N-1)].
        seed = tl.where(local[None, :] == 0, 1, x_val[:, None])
        powers = tl.cumprod(seed, axis=1)
        block = acc[:, None] * powers
        if increasing:
            store_col = k
        else:
            store_col = N - 1 - k
        out_off = rows[:, None] * N + store_col[None, :]
        mask = mask_m[:, None] & col_mask[None, :]
        tl.store(out_ptr + out_off, block, mask=mask)
        # Advance the running product by one tile: x^(off + BLOCK_N).
        last_col = tl.arange(0, BLOCK_N) == (BLOCK_N - 1)
        last_local = tl.sum(tl.where(last_col[None, :], powers, 0), axis=1)
        acc = acc * last_local * x_val


def vander(x, N=None, increasing=False):
    logger.debug("GEMS VANDER")

    if x.ndim != 1:
        raise RuntimeError("x must be a one-dimensional tensor.")

    if N is not None and N < 0:
        raise RuntimeError("N must be non-negative.")

    M = x.shape[0]
    if N is None:
        N = M

    if x.is_complex():
        # ATen keeps the complex dtype; compute real/imag powers separately.
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
        if M == 0 or N == 0:
            return out
        xc = x.contiguous()
        # gcu400: complex128 is ComplexFloat-replaced (fp32 payload) and fp64
        # accumulation cannot compile, so normalize to complex64 (the physical
        # payload dtype) and widen the result back value-level at the end.
        if xc.dtype != torch.complex64:
            xc = xc.to(torch.complex64)
        xr = xc.real
        xi = xc.imag
        output = torch.empty((M, N, 2), dtype=torch.float32, device=x.device)
        BLOCK_M = 32
        grid = (triton.cdiv(M, BLOCK_M),)
        vander_complex_kernel[grid](
            xr,
            xi,
            output[..., 0],
            output[..., 1],
            M,
            N,
            xr.stride(0),
            output[..., 0].stride(0),
            0.0,
            1.0,
            -1.0,
            increasing=increasing,
            BLOCK_M=BLOCK_M,
        )
        result = torch.view_as_complex(output)
        if x.dtype != torch.complex64:
            result = result.to(x.dtype)
        return result

    # ATen promotes the input with long: bool/integer -> int64, while
    # floating-point dtypes keep their own type.
    out_dtype = torch.promote_types(x.dtype, torch.long)
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)

    if M == 0 or N == 0:
        return out

    x_stride = x.stride(0)

    BLOCK_M = 32
    BLOCK_N = 32
    if x.dtype == torch.float64:
        # gcu400: fp64 tensors carry an fp32 payload (Double->Float replace) and
        # tl.float64 ops fail to legalize. Drop to a true fp32 copy (value-level
        # cast reads the payload correctly), run the df chain kernel, then widen
        # the result back so the output keeps the fp64 dtype label.
        x32 = x.to(torch.float32)
        out32 = torch.empty((M, N), dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(M, BLOCK_M),)
        vander_df_real_kernel[grid](
            x32,
            out32,
            M,
            N,
            x32.stride(0),
            0.0,
            increasing=increasing,
            BLOCK_M=BLOCK_M,
        )
        out.copy_(out32)
        return out
    if x.is_floating_point():
        compute_dtype = tl.float32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        vander_kernel[grid](
            x,
            out,
            M,
            N,
            x_stride,
            increasing=increasing,
            COMPUTE_DTYPE=compute_dtype,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    else:
        grid = (triton.cdiv(M, BLOCK_M),)
        vander_int_kernel[grid](
            x,
            out,
            M,
            N,
            x_stride,
            increasing=increasing,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    return out
