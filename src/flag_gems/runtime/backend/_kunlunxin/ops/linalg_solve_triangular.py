# Copyright 2026, The FlagOS Contributors.
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
#
# Kunlunxin(XPU) backend implementation of linalg_solve_triangular.
#
# The general implementation (flag_gems/ops/linalg_solve_triangular.py) is not
# usable on the XPU backend:
#   1. `_small_diag_kernel_notle` (n<=16) returns zeros for every row after
#      the first (per-row `tl.debug_barrier()` + global INV buffer round-trip
#      is miscompiled on XPU).
#   2. `_kslice_trsm_kernel_notle` (16<n<=512) crashes at LLIR compile time
#      ("src1Type.getShape()[0] != 1" in TritonSDNNToLLVM): its update phase
#      uses tl.dot with a K_SLICE=8 operand (M)x32@32x8; the XPU MMA lowering
#      does not support dot-N < 16.
#   3. The XPU device has no native fp64: arithmetic on fp64 tensors is
#      silently carried out in fp32 (measured), so the 1e-6 residual bound of
#      test_residual_f64 can never be met with a plain fp32-path kernel. The
#      kernel below emulates double precision with double-single
#      (error-compensated fp32 hi/lo pairs).
#
# The kernel avoids every known XPU trap (verified empirically):
#   * no tl.dot (XPU MMA crashes for small N),
#   * no tl.sum at all (masked-lane reductions read adjacent memory),
#   * no 3-D tl.sum, no tl.trans, no value broadcast of 1-D tiles into 2-D
#     (they fail the TritonXPU passes),
#   * no tl.debug_barrier inside loops,
#   * no register-vector accumulator seeded by tl.zeros (XPU miscompiles the
#     cross-iteration carry; the accumulator is seeded by the loaded RHS row),
#   * no multi-CTA k-slices: concurrent CTAs writing disjoint columns of the
#     same rows corrupt each other's observed write-read ordering on XPU; the
#     RHS is processed one slice per launch (host-sequenced), each a single
#     CTA of width <= KS_SLICE lanes,
#   * no masks at all: a slice's lane width always equals the slice width.
#
# The substitution dot product over the already-solved window is a serial
# scalar-j chain of K-vectors (one lane per RHS column) - the load/store
# pattern proven on XPU by linalg_ldl_solve.
#
# Upper-triangular solves are reduced to lower solves on the host by flipping
# rows and columns of A and B (P A P lower when A upper); the kernel only
# implements the lower substitution.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

KS_SLICE = 64  # max RHS-column slice width (single-CTA lane width)


@libentry()
@triton.jit
def _trsm_slice_xpu_kernel(
    A_ptr,
    B_ptr,
    Ah_ptr,
    Al_ptr,
    Bh_ptr,
    Bl_ptr,
    N,
    K,
    rsa,
    csb,
    COL0,
    F64: tl.constexpr,
    UNIT: tl.constexpr,
    KS: tl.constexpr,
):
    """One lower-triangular solve of an RHS column-slice: X = A^-1 B.

    Single CTA; rows solved serially; the solved-window dot product is a
    serial scalar-j chain on K-vectors (no reduction, no mask: the lane width
    always equals the slice width).
    """
    cc = tl.arange(0, KS)
    cm = cc < K
    for row in range(N):
        if F64:
            acc_h = tl.load(Bh_ptr + row * csb + COL0 + cc, mask=cm, other=0.0)
            acc_l = tl.load(Bl_ptr + row * csb + COL0 + cc, mask=cm, other=0.0)
        else:
            acc = tl.load(B_ptr + row * csb + COL0 + cc, mask=cm, other=0.0)
        for j in range(row):
            if F64:
                ah = tl.load(Ah_ptr + row * rsa + j)
                al = tl.load(Al_ptr + row * rsa + j)
                xh = tl.load(Bh_ptr + j * csb + COL0 + cc, mask=cm, other=0.0)
                xl = tl.load(Bl_ptr + j * csb + COL0 + cc, mask=cm, other=0.0)
                # Dekker exact split of the product: ph + pl == ah*xh + al*xh
                # + ah*xl + al*xl to ~2^-48 (no fma dependency; a plain
                # re-subtraction (ah*xh - ph) is CSE-folded to 0 by XPU)
                sp = 8193.0
                ca = ah * sp
                ab = ca - ah
                ahh = ca - ab
                ahl = ah - ahh
                cx = xh * sp
                xb = cx - xh
                xhh = cx - xb
                xhl = xh - xhh
                ph = ah * xh
                pl = (ahh * xhh - ph) + (ahh * xhl + ahl * xhh) + (
                    ahl * xhl + ah * xl + al * xh + al * xl
                )
                # two-sum subtraction (carry into the lo part)
                s = acc_h - ph
                acc_l = (acc_h - s) - ph + acc_l + pl
                acc_h = s
            else:
                a = tl.load(A_ptr + row * rsa + j)
                x = tl.load(B_ptr + j * csb + COL0 + cc, mask=cm, other=0.0)
                acc = acc - a * x
        if F64:
            out_h = acc_h
            out_l = acc_l
            if not UNIT:
                dh = tl.load(Ah_ptr + row * rsa + row)
                dl = tl.load(Al_ptr + row * rsa + row)
                # division q for double-single: q1 + refinement step
                q1 = out_h / dh
                ph = q1 * dh
                pl = tl.fma(q1, dh, -ph) + q1 * dl
                r = out_h - ph
                re = (out_h - r) - ph
                r_l = out_l - pl - re
                q2 = r / dh
                out_h = q1 + q2
                out_l = (q1 - out_h) + q2
            tl.store(Bh_ptr + row * csb + COL0 + cc, out_h, mask=cm)
            tl.store(Bl_ptr + row * csb + COL0 + cc, out_l, mask=cm)
        else:
            t = acc
            if not UNIT:
                d = tl.load(A_ptr + row * rsa + row)
                t = t * (1.0 / d)
            tl.store(B_ptr + row * csb + COL0 + cc, t, mask=cm)


def _expand_fp64_inputs(A, B):
    """Split fp64 tensors into (hi, lo) fp32 pairs (value = hi + lo)."""
    A_hi = A.float()
    A_lo = (A - A_hi.double()).float()
    B_hi = B.float()
    B_lo = (B - B_hi.double()).float()
    return A_hi, A_lo, B_hi, B_lo


def _solve_lower(A, B, unitriangular):
    """Solve A X = B with A lower triangular (A, B contiguous)."""
    n, k = A.shape[-1], B.shape[-1]
    if n == 1:
        # scalar solve x = b / a (n=1: the 1-lane kernel fails to LLVM-translate)
        if not unitriangular:
            inv = (1.0 / A[..., 0, 0]).reshape(-1)
            X = B.reshape(-1, k) * inv[:, None]
            X = X.reshape(B.shape)
        else:
            X = B.clone()
        return X
    is_fp64 = A.dtype == torch.float64
    orig_shape = B.shape

    batch = 1
    for d in A.shape[:-2]:
        batch *= d
    A_view = A.reshape(batch, n, n)
    B_view = B.clone().reshape(batch, n, k)

    rsa = A_view.stride(1)
    csb = k
    if is_fp64:
        Ah, Al, Bh, Bl = _expand_fp64_inputs(A_view, B_view)
        for b in range(batch):
            col0 = 0
            while col0 < k:
                ks = min(KS_SLICE, k - col0)
                _trsm_slice_xpu_kernel[(1,)](
                    A_view[b],
                    B_view[b],
                    Ah[b],
                    Al[b],
                    Bh[b],
                    Bl[b],
                    n,
                    k,
                    rsa,
                    csb,
                    col0,
                    F64=True,
                    UNIT=bool(unitriangular),
                    KS=ks,
                    num_warps=4,
                )
                col0 += ks
        X = Bh.reshape(orig_shape).to(torch.float64) + Bl.reshape(orig_shape).to(
            torch.float64
        )
    else:
        for b in range(batch):
            col0 = 0
            while col0 < k:
                ks = min(KS_SLICE, k - col0)
                _trsm_slice_xpu_kernel[(1,)](
                    A_view[b],
                    B_view[b],
                    A_view[b],
                    A_view[b],
                    B_view[b],
                    B_view[b],
                    n,
                    k,
                    rsa,
                    csb,
                    col0,
                    F64=False,
                    UNIT=bool(unitriangular),
                    KS=ks,
                    num_warps=4,
                )
                col0 += ks
        X = B_view.reshape(orig_shape)
    return X


def linalg_solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None):
    """Solve A X = B (left) or X A = B (right) with triangular A on XPU."""
    logger.debug("GEMS KUNLUNXIN LINALG_SOLVE_TRIANGULAR")
    if A.dtype not in (torch.float32, torch.float64):
        raise ValueError("linalg_solve_triangular only supports float32 and float64")
    if B.dtype != A.dtype:
        raise ValueError("A and B must have the same dtype")
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError("A and B must be at least 2D")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be a square matrix")

    if A.numel() == 0 or B.numel() == 0:
        if out is not None:
            out.copy_(B)
            return out
        return B.clone()

    if not left:
        if A.shape[-1] != B.shape[-1]:
            raise ValueError("Shape mismatch for XA=B")
        result = linalg_solve_triangular(
            A.mT.contiguous(),
            B.mT.contiguous(),
            upper=not upper,
            left=True,
            unitriangular=unitriangular,
        )
        result = result.mT.contiguous()
        if out is not None:
            out.copy_(result)
            return out
        return result

    A = A.contiguous()
    B = B.contiguous()

    if upper:
        # P A P is lower triangular when A is upper; x = P (lower-solve(P b)).
        # For the LEFT solve the reversal acts on the ROW axis of B/X.
        A = A.flip(-2).flip(-1).contiguous()
        B = B.flip(-2)
        X = _solve_lower(A, B, unitriangular)
        X = X.flip(-2)
    else:
        X = _solve_lower(A, B, unitriangular)

    if out is not None:
        out.copy_(X)
        return out
    return X


def linalg_solve_triangular_out(
    A, B, *, upper, left=True, unitriangular=False, out=None
):
    return linalg_solve_triangular(
        A, B, upper=upper, left=left, unitriangular=unitriangular, out=out
    )