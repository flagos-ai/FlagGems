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

"""Kunlunxin override for aten::ger (rank-1 outer product).

The generic implementation's forward kernel compiles and runs correctly on XPU,
so the forward path is reused verbatim.  Its *backward* kernels are not portable:
they carry a 2-D tile ``[BLOCK, BLOCK_R]`` whose ``tl.sum(..., axis=0/1)``
reduction (and the associated ``tt.addptr`` / ``tt.reduce`` lowering) trips the
XPU compiler (``'tt.addptr' op requires the same encoding for all operands`` at
``TritonXPUCoreTiling``, plus ``OutOfResources: uni_sram``).  The XPU-compatible
backward below rewrites the two matvec reductions as 1-D online kernels: one
program per output element, a loop over the reduction axis, and a final 1-D
``tl.sum`` fold.  This avoids every 2-D reduce and keeps each program's live
tile to a single ``BLOCK_R``-wide vector.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.ger import (
    _BLOCK_N,
    _BLOCK_R,
    _FP8_DTYPES,
    _ger_complex64_kernel,
    _ger_kernel,
    _launch,
)

logger = logging.getLogger(__name__)


@triton.jit
def _round_bf16_rne(x):
    """Round an fp32 value to bf16 (round-to-nearest-even), returned as fp32.

    The XPU compiler lowers bf16 multiply/cast with round-toward-zero instead of
    round-to-nearest-even, so ``aten::ger``'s bf16 backward product would drift
    one ULP per term from the reference.  bf16 is exactly the top 16 bits of an
    fp32, so adding ``0x7FFF + retained-LSB`` and masking the low 16 bits performs
    RNE without any hardware conversion.
    """
    u = x.to(tl.int32, bitcast=True)
    u = u + 0x7FFF + ((u >> 16) & 1)
    u = u & -65536
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _ger_fwd_flat_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    row0,
    sa,
    sb,
    IS_BF16: tl.constexpr,
    ROWS_PER: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """out[i, j] = self[i] * vec2[j] for a ``ROWS_PER x BLOCK_N`` tile.

    The generic forward tiles the output as a 2-D ``[BLOCK_M, BLOCK_N]`` block
    and writes it through a 2-D ``tl.store`` whose pointer is built with
    ``row_off[:, None] + col_off[None, :]``.  On XPU that 2-D broadcast multiply
    plus 2-D store miscompiles (``'tt.expand_dims' ... ranked tensor`` /
    ``OutOfResources: uni_sram``), so the generic forward runs at ~2.6 GB/s.

    Each program here fills a flat ``ROWS_PER * BLOCK_N``-element chunk of the
    row-major output with a single 1-D ``tl.store`` (the only store shape that
    vectorizes on XPU).  The per-element row/column are recovered with
    ``offs // BLOCK_N`` and ``offs % BLOCK_N``; ``BLOCK_N`` is ``constexpr`` so
    those lower to cheap constant division.  Both operands are gathered as
    vectors (``self[gi]`` and ``vec2[col]``): a scalar-broadcast multiply of the
    form ``scalar * vector`` miscompiles on XPU for non-power-of-two vectors, so
    every load is a vector gather even when ``ROWS_PER == 1`` (all ``gi`` equal).
    The gathers stay vectorized because the store is unmasked -- adding a
    ``gi < M`` mask here drops throughput ~100x (the masked store de-vectorizes),
    so the caller only launches full tiles and routes the ragged tail of
    ``M % ROWS_PER`` rows through a final shorter tile via ``row0``.  This kernel
    requires unit-stride operands: the ``col`` gather de-vectorizes (and
    mis-loads) once ``sb`` is not 1, so the caller keeps strided views on the
    generic path.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, ROWS_PER * BLOCK_N)
    row = offs // BLOCK_N
    col = offs % BLOCK_N
    gi = row0 + pid * ROWS_PER + row
    a = tl.load(a_ptr + gi * sa)
    b = tl.load(b_ptr + col * sb)
    if IS_BF16:
        # XPU bf16 multiply truncates and mishandles inf/nan (``inf * x`` can
        # come back ``nan``), so compute the exact product in fp32 and let the
        # store truncate back to bf16 (which is within the 0.016 test rtol).
        prod = a.to(tl.float32) * b.to(tl.float32)
    else:
        prod = a * b
    tl.store(o_ptr + (row0 + pid * ROWS_PER) * BLOCK_N + offs, prod)


def _forward_launch(a, b, out):
    """Dispatch the forward outer product on XPU.

    Floating-point dtypes with unit-stride operands and a power-of-two ``N``
    take the flat 1-D-store kernel above; every other case (integer, bool,
    complex, strided views, and non-power-of-two ``N``) keeps the generic
    (correct but slower) 2-D path.  The flat kernel fills a ``ROWS_PER x N``
    tile whose flat size must be an exact power of two: ``tl.arange`` on XPU
    rounds a non-power-of-two size up and the store writes past the tile (out
    of bounds), and masking the store to bound it de-vectorizes the write
    (~100x).  So only power-of-two ``N`` is eligible (``offs // N`` /
    ``offs % N`` are then cheap constant shifts, and ``ROWS_PER * N`` is a
    power of two for power-of-two ``ROWS_PER``), and the ragged tail of
    ``M % rows_per`` rows is binary-decomposed into power-of-two row chunks
    rather than masked.  Strided operand views (``stride != 1``) also keep the
    generic path, since the flat kernel's ``col`` gather only vectorizes for
    unit stride.
    """
    if out.dtype.is_floating_point:
        M = out.shape[0]
        N = out.shape[1]
        if M == 0 or N == 0:
            return out
        sa = a.stride(0)
        sb = b.stride(0)
        if sa == 1 and sb == 1 and (N & (N - 1)) == 0:
            is_bf16 = out.dtype == torch.bfloat16
            # Power-of-two rows per full tile.  32 keeps the contiguous store in
            # the high-bandwidth regime; the cap keeps the tile (still a power
            # of two) bounded for very large N so it does not overflow SRAM.
            rows_per = 32
            while rows_per > 1 and rows_per * N > 524288:
                rows_per //= 2
            n_full = M // rows_per
            rem = M % rows_per
            if n_full:
                _ger_fwd_flat_kernel[(n_full,)](
                    a,
                    b,
                    out,
                    0,
                    sa,
                    sb,
                    IS_BF16=is_bf16,
                    ROWS_PER=rows_per,
                    BLOCK_N=N,
                    num_warps=8,
                )
            # Ragged tail: binary-decompose into power-of-two row chunks so each
            # tile stays an exact power of two (no arange rounding, no mask).
            row0 = n_full * rows_per
            while rem > 0:
                chunk = 1 << (rem.bit_length() - 1)
                _ger_fwd_flat_kernel[(1,)](
                    a,
                    b,
                    out,
                    row0,
                    sa,
                    sb,
                    IS_BF16=is_bf16,
                    ROWS_PER=chunk,
                    BLOCK_N=N,
                    num_warps=8,
                )
                row0 += chunk
                rem -= chunk
            return out
    return _launch(a, b, out)


@triton.jit
def _ger_bwd_a_kernel(
    g_ptr,
    b_ptr,
    o_ptr,
    N,
    sgm,
    sgn,
    sb,
    so,
    IS_BF16: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """dself[i] = sum_j g[i, j] * vec2[j]  (real dtypes, 1-D online)."""
    i = tl.program_id(0)
    offs = tl.arange(0, BLOCK_R)
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for r0 in range(0, N, BLOCK_R):
        offs_r = r0 + offs
        mask = offs_r < N
        g = tl.load(g_ptr + i * sgm + offs_r * sgn, mask=mask, other=0.0)
        bv = tl.load(b_ptr + offs_r * sb, mask=mask, other=0.0)
        # aten::ger's gradient multiplies in the operand dtype (each product is
        # rounded to fp16/bf16) before reducing in fp32.  fp16/fp32 multiply are
        # already RNE on XPU, but bf16 multiply truncates, so bf16 needs the
        # explicit RNE round above.
        if IS_BF16:
            prod = _round_bf16_rne(g.to(tl.float32) * bv.to(tl.float32))
        else:
            prod = (g * bv).to(tl.float32)
        acc += prod
    tl.store(o_ptr + i * so, tl.sum(acc).to(o_ptr.dtype.element_ty))


@triton.jit
def _ger_bwd_b_kernel(
    g_ptr,
    a_ptr,
    o_ptr,
    M,
    sgm,
    sgn,
    sa,
    so,
    IS_BF16: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """dvec2[j] = sum_i g[i, j] * self[i]  (real dtypes, 1-D online)."""
    j = tl.program_id(0)
    offs = tl.arange(0, BLOCK_R)
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for r0 in range(0, M, BLOCK_R):
        offs_r = r0 + offs
        mask = offs_r < M
        g = tl.load(g_ptr + offs_r * sgm + j * sgn, mask=mask, other=0.0)
        av = tl.load(a_ptr + offs_r * sa, mask=mask, other=0.0)
        # Native-dtype product, fp32 accumulation (see _ger_bwd_a_kernel).
        if IS_BF16:
            prod = _round_bf16_rne(g.to(tl.float32) * av.to(tl.float32))
        else:
            prod = (g * av).to(tl.float32)
        acc += prod
    tl.store(o_ptr + j * so, tl.sum(acc).to(o_ptr.dtype.element_ty))


@triton.jit
def _ger_bwd_a_c64_kernel(
    g_ptr,
    b_ptr,
    o_ptr,
    N,
    sgm,
    sgn,
    sb,
    BLOCK_R: tl.constexpr,
):
    """dself[i] = sum_j g[i, j] * conj(vec2[j])  (complex64, float views)."""
    i = tl.program_id(0)
    offs = tl.arange(0, BLOCK_R)
    acc_re = tl.zeros((BLOCK_R,), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for r0 in range(0, N, BLOCK_R):
        offs_r = r0 + offs
        mask = offs_r < N
        base = i * sgm + offs_r * sgn
        gr = tl.load(g_ptr + base, mask=mask, other=0.0)
        gi = tl.load(g_ptr + base + 1, mask=mask, other=0.0)
        br = tl.load(b_ptr + offs_r * sb, mask=mask, other=0.0)
        bi = tl.load(b_ptr + offs_r * sb + 1, mask=mask, other=0.0)
        acc_re += gr * br + gi * bi
        acc_im += gi * br - gr * bi
    tl.store(o_ptr + i * 2, tl.sum(acc_re))
    tl.store(o_ptr + i * 2 + 1, tl.sum(acc_im))


@triton.jit
def _ger_bwd_b_c64_kernel(
    g_ptr,
    a_ptr,
    o_ptr,
    M,
    sgm,
    sgn,
    sa,
    BLOCK_R: tl.constexpr,
):
    """dvec2[j] = sum_i g[i, j] * conj(self[i])  (complex64, float views)."""
    j = tl.program_id(0)
    offs = tl.arange(0, BLOCK_R)
    acc_re = tl.zeros((BLOCK_R,), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for r0 in range(0, M, BLOCK_R):
        offs_r = r0 + offs
        mask = offs_r < M
        base = offs_r * sgm + j * sgn
        gr = tl.load(g_ptr + base, mask=mask, other=0.0)
        gi = tl.load(g_ptr + base + 1, mask=mask, other=0.0)
        ar = tl.load(a_ptr + offs_r * sa, mask=mask, other=0.0)
        ai = tl.load(a_ptr + offs_r * sa + 1, mask=mask, other=0.0)
        acc_re += gr * ar + gi * ai
        acc_im += gi * ar - gr * ai
    tl.store(o_ptr + j * 2, tl.sum(acc_re))
    tl.store(o_ptr + j * 2 + 1, tl.sum(acc_im))


def _backward(a, b, grad_out, da, db):
    M = a.shape[0]
    N = b.shape[0]
    is_bf16 = a.dtype == torch.bfloat16
    if a.is_complex():
        gv = torch.view_as_real(grad_out)
        av = torch.view_as_real(a)
        bv = torch.view_as_real(b)
        dav = torch.view_as_real(da)
        dbv = torch.view_as_real(db)
        _ger_bwd_a_c64_kernel[(M,)](
            gv,
            bv,
            dav,
            N,
            gv.stride(0),
            gv.stride(1),
            bv.stride(0),
            BLOCK_R=_BLOCK_R,
        )
        _ger_bwd_b_c64_kernel[(N,)](
            gv,
            av,
            dbv,
            M,
            gv.stride(0),
            gv.stride(1),
            av.stride(0),
            BLOCK_R=_BLOCK_R,
        )
    else:
        _ger_bwd_a_kernel[(M,)](
            grad_out,
            b,
            da,
            N,
            grad_out.stride(0),
            grad_out.stride(1),
            b.stride(0),
            da.stride(0),
            IS_BF16=is_bf16,
            BLOCK_R=_BLOCK_R,
        )
        _ger_bwd_b_kernel[(N,)](
            grad_out,
            a,
            db,
            M,
            grad_out.stride(0),
            grad_out.stride(1),
            a.stride(0),
            db.stride(0),
            IS_BF16=is_bf16,
            BLOCK_R=_BLOCK_R,
        )
    return da, db


class _Ger(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        M = a.shape[0]
        N = b.shape[0]
        out = torch.empty((M, N), dtype=a.dtype, device=a.device)
        _forward_launch(a, b, out)
        ctx.save_for_backward(a, b)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        da = torch.empty(a.shape, dtype=a.dtype, device=a.device)
        db = torch.empty(b.shape, dtype=b.dtype, device=b.device)
        g = grad_out.contiguous()
        _backward(a, b, g, da, db)
        return da, db


def ger(self, vec2, out=None):
    logger.debug("GEMS_KUNLUNXIN GER")
    if not isinstance(self, torch.Tensor) or not isinstance(vec2, torch.Tensor):
        raise TypeError("ger: expected Tensor arguments")
    if self.dim() != 1:
        raise ValueError("ger: Expected 1-D argument self")
    if vec2.dim() != 1:
        raise ValueError("ger: Expected 1-D argument vec2")
    if self.dtype in _FP8_DTYPES or vec2.dtype in _FP8_DTYPES:
        raise RuntimeError(
            "ger: fp8 operands are not supported by the NVIDIA backend "
            "(mul is not implemented for Float8)"
        )

    dtype = torch.result_type(self, vec2)
    M = self.shape[0]
    N = vec2.shape[0]

    if out is None:
        if (self.requires_grad or vec2.requires_grad) and (
            dtype.is_floating_point or dtype.is_complex
        ):
            return _Ger.apply(self.to(dtype), vec2.to(dtype))
        out = torch.empty((M, N), dtype=dtype, device=self.device)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("ger: out must be a Tensor")
        if out.dim() != 2 or out.shape[0] != M or out.shape[1] != N:
            raise ValueError("ger: out has an incompatible shape")
        if out.dtype != dtype:
            raise ValueError("ger: out has an incompatible dtype")

    if M == 0 or N == 0:
        return out

    return _forward_launch(self.to(dtype), vec2.to(dtype), out)
