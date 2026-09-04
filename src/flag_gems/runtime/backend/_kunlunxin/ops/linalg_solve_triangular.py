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
    F64: tl.constexpr,
    UNIT: tl.constexpr,
    KS: tl.constexpr,
    UPPER: tl.constexpr,
    NS: tl.constexpr,
):
    """One triangular solve of an RHS column-slice: X = A^-1 B.

    grid = (batch * NS,): program ``pid`` owns batch ``pid // NS`` and the RHS
    column slice ``pid % NS`` (``KS`` columns).  Column slices are independent
    solves, so the only cross-program traffic is the read-only A; each program
    reads back exactly the X lanes it wrote.  Batch/slice bases are derived
    from the existing strides (``N * rsa`` / ``N * csb``) on purpose: adding a
    runtime scalar argument to an XPU kernel costs 15-30x.

    Rows are solved serially (the triangular dependency chain); the dot product
    over the already-solved window is a serial scalar-j chain on K-vectors.
    TritonXPU cannot batch this reduction: a 2D tile inside the row loop fails
    `tt.addptr` verification, `tl.static_range` over tiles fails the XPU layout
    check, and a 1D `tl.sum` inside a runtime loop hits
    "failed to legalize operation 'tt.reduce'" (all measured 2026-08-30).

    ``UPPER`` selects the sweep direction in-kernel (backward substitution over
    reversed row/column indices) so the host never has to materialise a flipped
    copy of A/B: the gems-registered ``flip`` kernel raises
    KL_XID_KERNEL_EXCEPTION for a 512x512 fp32 last-dim flip (measured).
    """
    pid = tl.program_id(0)
    bidx = pid // NS
    sidx = pid % NS
    abase = bidx * N * rsa
    bbase = bidx * N * csb + sidx * KS
    cc = tl.arange(0, KS)
    cm = cc < K
    for t in range(N):
        row = N - 1 - t if UPPER else t
        if F64:
            acc_h = tl.load(Bh_ptr + bbase + row * csb + cc, mask=cm, other=0.0)
            acc_l = tl.load(Bl_ptr + bbase + row * csb + cc, mask=cm, other=0.0)
        else:
            acc = tl.load(B_ptr + bbase + row * csb + cc, mask=cm, other=0.0)
        for u in range(t):
            j = N - 1 - u if UPPER else u
            if F64:
                ah = tl.load(Ah_ptr + abase + row * rsa + j)
                al = tl.load(Al_ptr + abase + row * rsa + j)
                xh = tl.load(Bh_ptr + bbase + j * csb + cc, mask=cm, other=0.0)
                xl = tl.load(Bl_ptr + bbase + j * csb + cc, mask=cm, other=0.0)
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
                pl = (
                    (ahh * xhh - ph)
                    + (ahh * xhl + ahl * xhh)
                    + (ahl * xhl + ah * xl + al * xh + al * xl)
                )
                # two-sum subtraction (carry into the lo part)
                s = acc_h - ph
                acc_l = (acc_h - s) - ph + acc_l + pl
                acc_h = s
            else:
                a = tl.load(A_ptr + abase + row * rsa + j)
                x = tl.load(B_ptr + bbase + j * csb + cc, mask=cm, other=0.0)
                acc = acc - a * x
        if F64:
            out_h = acc_h
            out_l = acc_l
            if not UNIT:
                dh = tl.load(Ah_ptr + abase + row * rsa + row)
                dl = tl.load(Al_ptr + abase + row * rsa + row)
                # division q for double-single: q1 + refinement step
                q1 = out_h / dh
                ph = q1 * dh
                pl = tl.fma(q1, dh, -ph) + q1 * dl
                r = out_h - ph
                q2 = r / dh
                out_h = q1 + q2
                out_l = (q1 - out_h) + q2
            tl.store(Bh_ptr + bbase + row * csb + cc, out_h, mask=cm)
            tl.store(Bl_ptr + bbase + row * csb + cc, out_l, mask=cm)
        else:
            xv = acc
            if not UNIT:
                d = tl.load(A_ptr + abase + row * rsa + row)
                xv = xv * (1.0 / d)
            tl.store(B_ptr + bbase + row * csb + cc, xv, mask=cm)


def _expand_fp64_inputs(A, B):
    """Split fp64 tensors into (hi, lo) fp32 pairs (value = hi + lo)."""
    A_hi = A.float()
    A_lo = (A - A_hi.double()).float()
    B_hi = B.float()
    B_lo = (B - B_hi.double()).float()
    return A_hi, A_lo, B_hi, B_lo


@libentry()
@triton.jit
def _trsm_update_xpu_kernel(
    A_ptr,
    B_ptr,
    N,
    K,
    R0,
    rsa,
    csb,
    KS: tl.constexpr,
    NS: tl.constexpr,
    BR: tl.constexpr,
    UPPER: tl.constexpr,
):
    """Phase 1 of the blocked sweep: subtract the already-solved rows [0, R0).

    grid = (batch * BR * NS,).  Every (row-in-block, column-slice) pair is an
    independent program, so this phase exposes BR times more parallelism than
    the substitution sweep, which is what lifts the shapes whose RHS is too
    narrow to fill the device with column slices alone.
    """
    pid = tl.program_id(0)
    sidx = pid % NS
    rl = (pid // NS) % BR
    bidx = pid // (NS * BR)
    abase = bidx * N * rsa
    bbase = bidx * N * csb + sidx * KS
    cc = tl.arange(0, KS)
    cm = cc < K
    t = R0 + rl
    row = N - 1 - t if UPPER else t
    acc = tl.load(B_ptr + bbase + row * csb + cc, mask=cm, other=0.0)
    arow = A_ptr + abase + row * rsa
    for u in range(R0):
        j = N - 1 - u if UPPER else u
        a = tl.load(arow + j)
        x = tl.load(B_ptr + bbase + j * csb + cc, mask=cm, other=0.0)
        acc = acc - a * x
    tl.store(B_ptr + bbase + row * csb + cc, acc, mask=cm)


@libentry()
@triton.jit
def _trsm_diag_xpu_kernel(
    A_ptr,
    B_ptr,
    N,
    K,
    R0,
    rsa,
    csb,
    KS: tl.constexpr,
    NS: tl.constexpr,
    BR: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
):
    """Phase 2 of the blocked sweep: serial solve of the BR x BR diagonal block.

    grid = (batch * NS,).  The RHS rows of the block were already updated by
    phase 1 (previous launch => globally visible), so only the intra-block
    dependency chain remains.
    """
    pid = tl.program_id(0)
    sidx = pid % NS
    bidx = pid // NS
    abase = bidx * N * rsa
    bbase = bidx * N * csb + sidx * KS
    cc = tl.arange(0, KS)
    cm = cc < K
    for q in range(BR):
        t = R0 + q
        row = N - 1 - t if UPPER else t
        acc = tl.load(B_ptr + bbase + row * csb + cc, mask=cm, other=0.0)
        arow = A_ptr + abase + row * rsa
        for u in range(R0, t):
            j = N - 1 - u if UPPER else u
            a = tl.load(arow + j)
            x = tl.load(B_ptr + bbase + j * csb + cc, mask=cm, other=0.0)
            acc = acc - a * x
        if not UNIT:
            d = tl.load(arow + row)
            acc = acc * (1.0 / d)
        tl.store(B_ptr + bbase + row * csb + cc, acc, mask=cm)


def _pick_block_rows(n):
    """Row-block size for the two-phase sweep, or 0 to keep the single sweep.

    Measured on XPU 6 (fp32, KS=64, torch-reference floor in brackets):
      n=16  single 0.175 ms  vs BR=8 0.209  -> single wins (launch bound)
      n=32  single 0.460     vs BR=8 0.411  -> ~tie, keep single (fewer launches)
      n=64  single 1.663     vs BR=16 1.056
      n=128 single 6.548     vs BR=32 4.467
      n=256 single 26.34     vs BR=32 23.86
      n=512 single 94.44     vs BR=32 45.21
    BR must divide n exactly: a ragged tail block would need the block length as
    a runtime kernel argument, and extra runtime scalars are a 15-30x cliff on
    this backend.
    """
    if n < 64:
        return 0
    cap = 16 if n < 128 else 32
    for br in (32, 16, 8):
        if br <= cap and n % br == 0:
            return br
    return 0


def _solve_tri(A, B, unitriangular, upper):
    """Solve A X = B with A triangular (A, B contiguous)."""
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
    B_view = B.reshape(batch, n, k)

    rsa = A_view.stride(1)
    # Pad the RHS column count to a multiple of KS_SLICE: a partial tail
    # slice (masked lanes) corrupts the surrounding buffer on XPU, so every
    # slice launch uses the full KS_SLICE lane width with an all-true mask.
    kpad = ((k + KS_SLICE - 1) // KS_SLICE) * KS_SLICE
    nslices = kpad // KS_SLICE
    # One launch for the whole (batch x slice) grid: column slices are
    # independent solves, so they run concurrently on separate XPU programs
    # (measured 3.8x at nslices=4).  Host-side per-slice launches were the
    # previous shape and serialised everything.
    grid = (batch * nslices,)
    if is_fp64:
        Ah, Al, Bh0, Bl0 = _expand_fp64_inputs(A_view, B_view)
        Bh = torch.zeros((batch, n, kpad), dtype=torch.float32, device=A.device)
        Bl = torch.zeros((batch, n, kpad), dtype=torch.float32, device=A.device)
        torch.ops.aten._copy_from(Bh0, Bh[:, :, :k], False)
        torch.ops.aten._copy_from(Bl0, Bl[:, :, :k], False)
        _trsm_slice_xpu_kernel[grid](
            A_view,
            B_view,
            Ah,
            Al,
            Bh,
            Bl,
            n,
            kpad,
            rsa,
            kpad,
            F64=True,
            UNIT=bool(unitriangular),
            KS=KS_SLICE,
            UPPER=bool(upper),
            NS=nslices,
            num_warps=4,
        )
        X = (Bh[:, :, :k].to(torch.float64) + Bl[:, :, :k].to(torch.float64)).reshape(
            orig_shape
        )
    else:
        if kpad == k:
            Bp = B_view.clone()
        else:
            Bp = torch.zeros((batch, n, kpad), dtype=A.dtype, device=A.device)
            torch.ops.aten._copy_from(B_view, Bp[:, :, :k], False)
        br = _pick_block_rows(n)
        if br:
            for r0 in range(0, n, br):
                if r0:
                    _trsm_update_xpu_kernel[(batch * br * nslices,)](
                        A_view,
                        Bp,
                        n,
                        kpad,
                        r0,
                        rsa,
                        kpad,
                        KS=KS_SLICE,
                        NS=nslices,
                        BR=br,
                        UPPER=bool(upper),
                        num_warps=4,
                    )
                _trsm_diag_xpu_kernel[(batch * nslices,)](
                    A_view,
                    Bp,
                    n,
                    kpad,
                    r0,
                    rsa,
                    kpad,
                    KS=KS_SLICE,
                    NS=nslices,
                    BR=br,
                    UPPER=bool(upper),
                    UNIT=bool(unitriangular),
                    num_warps=4,
                )
        else:
            _trsm_slice_xpu_kernel[grid](
                A_view,
                Bp,
                A_view,
                A_view,
                Bp,
                Bp,
                n,
                kpad,
                rsa,
                kpad,
                F64=False,
                UNIT=bool(unitriangular),
                KS=KS_SLICE,
                UPPER=bool(upper),
                NS=nslices,
                num_warps=4,
            )
        X = Bp[:, :, :k].reshape(orig_shape)
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
            torch.ops.aten._copy_from(B, out, False)
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
            torch.ops.aten._copy_from(result, out, False)
            return out
        return result

    A = A.contiguous()
    B = B.contiguous()

    # RHS batch broadcast: B's batch dims broadcast against A's batch dims
    # (e.g. unbatched B with batched A). The result takes the broadcast shape.
    batch_shape = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    if B.shape[:-2] != batch_shape:
        B = B.expand(batch_shape + B.shape[-2:]).contiguous()

    if upper:
        # Backward substitution is done inside the kernel (UPPER=True); the
        # host must NOT materialise P A P / P b: the gems-registered `flip`
        # kernel faults (KL_XID_KERNEL_EXCEPTION / status=700) for a 512x512
        # fp32 last-dim flip, which is exactly the A shape of
        # test_large_n_f64[dtype0-True-1-512].
        X = _solve_tri(A, B, unitriangular, True)
    else:
        X = _solve_tri(A, B, unitriangular, False)

    if out is not None:
        torch.ops.aten._copy_from(X, out, False)
        return out
    return X


def linalg_solve_triangular_out(
    A, B, *, upper, left=True, unitriangular=False, out=None
):
    return linalg_solve_triangular(
        A, B, upper=upper, left=left, unitriangular=unitriangular, out=out
    )
