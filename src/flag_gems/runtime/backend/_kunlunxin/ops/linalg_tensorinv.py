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

"""Kunlunxin backend override for linalg_tensorinv / linalg_tensorinv.out.

The generic _tinv_reg_kernel keeps a full BLOCK_N-by-BLOCK_N
register tile live across a data-dependent serial pivot loop.  The
TritonXPULegalize MLIR pipeline crashes on exactly that combination on this
backend (same class of failure as linalg_inv / linalg_eig:
arith.cmpi op failed to verify ...), so the generic entry point cannot
compile here.

tensorinv reduces to a single dense m x m inversion (m =
prod(shape[:ind])), so the override dispatches on m:

* m <= _TINV_SMALL_N — closed-form inverse computed from scalar element
  loads (_tinv_small), with no cross-lane reduction at all.  On this
  backend a full reduction is a fixed-cost cross-core synchronisation
  whatever the register width, so the reduce-free path removes the dominant
  term for tiny matrices.
* _TINV_SMALL_N < m <= _TINV_GJ_MAX — partial-pivoted Gauss-Jordan on
  separate 1-D row registers (_tinv_row_kernel).  The row-wise (1-D)
  formulation is deliberate: the XPU core-tiling pass rejects axis = 0
  reductions on 2-D tiles, reductions over broadcast-derived tiles, and any
  reduction whose operand already came out of a reduction, so a 2-D register
  tile cannot be compiled here.  Every reduction is a full reduce of a 1-D
  vector that is an elementwise function of a global load, which this backend
  tiles correctly.  The fully unrolled row-register form is validated up to
  m = 13 here: past that point the backend miscompiles the unrolled reduction
  chain (illegal memory access observed with the same kernel shape in
  linalg_inv), so larger m goes through the LU path below.
* m > _TINV_GJ_MAX — the LU factorisation + column substitution
  combination (_linalg_lu_factor + _xpu_lu_solve_serial): A^-1 is
  obtained by solving A X = I column by column, so no register tile
  survives across the pivot loop (m = 80 in the test matrix is well
  within the N <= 128 register-resident range).

fp16/bf16 inputs are upcast to fp32 for the arithmetic and downcast on store
via the kernel dtype-generic stores, mirroring the generic kernel internal
fp32 tile arithmetic.  A zero pivot (exactly singular input) propagates
inf/nan instead of a finite wrong inverse.
"""

import logging
import warnings

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_lu_solve import _xpu_lu_solve_serial
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .linalg_inv_ex import _xpu_identity_rhs
from .linalg_lu_factor import _linalg_lu_factor

logger = logging.getLogger(__name__)

_TINV_SMALL_N = 4
_TINV_GJ_MAX = 13


@libentry()
@triton.jit
def _cast_kernel(src, dst, numel, BLOCK: tl.constexpr):
    """Elementwise dtype cast run entirely on-device (no host-side aten cast)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    v = tl.load(src + offs, mask=mask, other=0.0)
    tl.store(dst + offs, v.to(dst.dtype.element_ty), mask=mask)


def _cast_tensor(x, dtype):
    """Cast x to dtype via a Triton kernel.

    Replaces x.to(dtype) (which dispatches to the aten _to_copy cast on
    the host path) so the dtype conversion for fp16/bf16 <-> fp32 stays inside a
    real XPU kernel and the host function only allocates.
    """
    if x.dtype == dtype:
        return x
    xc = x.contiguous()
    out = torch.empty(xc.shape, dtype=dtype, device=xc.device)
    numel = xc.numel()
    if numel == 0:
        return out
    block = 1024
    grid = (triton.cdiv(numel, block),)
    with torch_device_fn.device(xc.device):
        _cast_kernel[grid](xc, out, numel, BLOCK=block, num_warps=4)
    return out


def check_inv_input(A, ind):
    """Validate input for tensorinv: ind is strictly positive, A is >=2D, and
    prod(A.shape[:ind]) == prod(A.shape[ind:]). Raises RuntimeError on
    violation, matching torch.linalg.tensorinv behaviour.
    """
    # Checked first, as in torch: a non-positive ind is rejected before any
    # shape reasoning, so ind=0 cannot slip through to the prod comparison
    # below (which it would satisfy trivially for a 1x1 input, yielding a
    # silently wrong result instead of an error).
    if ind <= 0:
        raise RuntimeError(
            "linalg.tensorinv: Expected a strictly positive integer for "
            f"ind, but got {ind}"
        )
    if A.dim() < 2:
        raise RuntimeError(
            "linalg.tensorinv: Expected input to be at least 2D, " f"got {A.dim()}D"
        )
    if ind > A.dim():
        raise RuntimeError(
            "linalg.tensorinv: Expected 0 <= ind <= input.dim(), "
            f"got ind={ind} for input.dim()={A.dim()}"
        )
    m = 1
    for i in range(ind):
        m *= A.shape[i]
    n = 1
    for i in range(ind, A.dim()):
        n *= A.shape[i]
    if m != n:
        raise RuntimeError(
            "linalg.tensorinv: Expected self to satisfy the requirement "
            "prod(self.shape[ind:]) == prod(self.shape[:ind]), "
            f"but got {n} != {m}"
        )


@triton.jit
def _tinv_small(
    Ap,
    Op,
    s_ar,
    s_ac,
    n: tl.constexpr,
    BN: tl.constexpr,
):
    """Closed-form inverse for n <= 4 from scalar element loads.

    Loads are upcast to fp32 and stored back through the element type of
    Op, so the arithmetic runs in fp32 even for fp16/bf16 input.  A zero
    determinant (exactly singular input) yields inf/nan through 1.0 / det
    instead of a finite wrong inverse.
    """
    c = tl.arange(0, BN)
    cm = c < n
    base = Ap

    e = ()
    for i in tl.static_range(0, n):
        for j in tl.static_range(0, n):
            e = e + (tl.load(base + i * s_ar + j * s_ac).to(tl.float32),)
    z = e[0] * 0.0

    if n == 1:
        g = 1.0 / e[0]
        o0 = tl.where(c == 0, g, z)
        tl.store(Op + c, o0.to(Op.dtype.element_ty), mask=cm)

    if n == 2:
        det = e[0] * e[3] - e[1] * e[2]
        g = 1.0 / det
        o0 = tl.where(c == 0, e[3] * g, z)
        o0 = tl.where(c == 1, -e[1] * g, o0)
        o1 = tl.where(c == 0, -e[2] * g, z)
        o1 = tl.where(c == 1, e[0] * g, o1)
        tl.store(Op + c, o0.to(Op.dtype.element_ty), mask=cm)
        tl.store(Op + n + c, o1.to(Op.dtype.element_ty), mask=cm)

    if n == 3:
        det = (
            e[0] * (e[4] * e[8] - e[5] * e[7])
            - e[1] * (e[3] * e[8] - e[5] * e[6])
            + e[2] * (e[3] * e[7] - e[4] * e[6])
        )
        g = 1.0 / det
        m0 = e[4] * e[8] - e[5] * e[7]
        m1 = e[2] * e[7] - e[1] * e[8]
        m2 = e[1] * e[5] - e[2] * e[4]
        m3 = e[5] * e[6] - e[3] * e[8]
        m4 = e[0] * e[8] - e[2] * e[6]
        m5 = e[2] * e[3] - e[0] * e[5]
        m6 = e[3] * e[7] - e[4] * e[6]
        m7 = e[1] * e[6] - e[0] * e[7]
        m8 = e[0] * e[4] - e[1] * e[3]
        o0 = tl.where(c == 0, m0 * g, z)
        o0 = tl.where(c == 1, m1 * g, o0)
        o0 = tl.where(c == 2, m2 * g, o0)
        o1 = tl.where(c == 0, m3 * g, z)
        o1 = tl.where(c == 1, m4 * g, o1)
        o1 = tl.where(c == 2, m5 * g, o1)
        o2 = tl.where(c == 0, m6 * g, z)
        o2 = tl.where(c == 1, m7 * g, o2)
        o2 = tl.where(c == 2, m8 * g, o2)
        tl.store(Op + c, o0.to(Op.dtype.element_ty), mask=cm)
        tl.store(Op + n + c, o1.to(Op.dtype.element_ty), mask=cm)
        tl.store(Op + 2 * n + c, o2.to(Op.dtype.element_ty), mask=cm)

    if n == 4:
        # block inverse of [[P, Q], [R, S]] with 2x2 blocks
        detp = e[0] * e[5] - e[1] * e[4]
        gp = 1.0 / detp
        p00 = e[5] * gp
        p01 = -e[1] * gp
        p10 = -e[4] * gp
        p11 = e[0] * gp
        # T = R P^-1
        t00 = e[8] * p00 + e[9] * p10
        t01 = e[8] * p01 + e[9] * p11
        t10 = e[12] * p00 + e[13] * p10
        t11 = e[12] * p01 + e[13] * p11
        # U = T Q, Sm = S - U
        sm00 = e[10] - (t00 * e[2] + t01 * e[6])
        sm01 = e[11] - (t00 * e[3] + t01 * e[7])
        sm10 = e[14] - (t10 * e[2] + t11 * e[6])
        sm11 = e[15] - (t10 * e[3] + t11 * e[7])
        dets = sm00 * sm11 - sm01 * sm10
        gs = 1.0 / dets
        i00 = sm11 * gs
        i01 = -sm01 * gs
        i10 = -sm10 * gs
        i11 = sm00 * gs
        # W = P^-1 Q Si
        w00 = (p00 * e[2] + p01 * e[6]) * i00 + (p00 * e[3] + p01 * e[7]) * i10
        w01 = (p00 * e[2] + p01 * e[6]) * i01 + (p00 * e[3] + p01 * e[7]) * i11
        w10 = (p10 * e[2] + p11 * e[6]) * i00 + (p10 * e[3] + p11 * e[7]) * i10
        w11 = (p10 * e[2] + p11 * e[6]) * i01 + (p10 * e[3] + p11 * e[7]) * i11
        # TL = P^-1 + W R P^-1
        y00 = w00 * e[8] + w01 * e[12]
        y01 = w00 * e[9] + w01 * e[13]
        y10 = w10 * e[8] + w11 * e[12]
        y11 = w10 * e[9] + w11 * e[13]
        a00 = p00 + y00 * p00 + y01 * p10
        a01 = p01 + y00 * p01 + y01 * p11
        a10 = p10 + y10 * p00 + y11 * p10
        a11 = p11 + y10 * p01 + y11 * p11
        # BL = -Si R P^-1
        z00 = i00 * e[8] + i01 * e[12]
        z01 = i00 * e[9] + i01 * e[13]
        z10 = i10 * e[8] + i11 * e[12]
        z11 = i10 * e[9] + i11 * e[13]
        b00 = -(z00 * p00 + z01 * p10)
        b01 = -(z00 * p01 + z01 * p11)
        b10 = -(z10 * p00 + z11 * p10)
        b11 = -(z10 * p01 + z11 * p11)

        o0 = tl.where(c == 0, a00, z)
        o0 = tl.where(c == 1, a01, o0)
        o0 = tl.where(c == 2, -w00, o0)
        o0 = tl.where(c == 3, -w01, o0)
        o1 = tl.where(c == 0, a10, z)
        o1 = tl.where(c == 1, a11, o1)
        o1 = tl.where(c == 2, -w10, o1)
        o1 = tl.where(c == 3, -w11, o1)
        o2 = tl.where(c == 0, b00, z)
        o2 = tl.where(c == 1, b01, o2)
        o2 = tl.where(c == 2, i00, o2)
        o2 = tl.where(c == 3, i01, o2)
        o3 = tl.where(c == 0, b10, z)
        o3 = tl.where(c == 1, b11, o3)
        o3 = tl.where(c == 2, i10, o3)
        o3 = tl.where(c == 3, i11, o3)
        tl.store(Op + c, o0.to(Op.dtype.element_ty), mask=cm)
        tl.store(Op + n + c, o1.to(Op.dtype.element_ty), mask=cm)
        tl.store(Op + 2 * n + c, o2.to(Op.dtype.element_ty), mask=cm)
        tl.store(Op + 3 * n + c, o3.to(Op.dtype.element_ty), mask=cm)


@triton.jit
def _tinv_row_kernel(
    Ap,
    Op,
    s_ar,
    s_ac,
    n: tl.constexpr,
    BN: tl.constexpr,
):
    """Partial-pivoted Gauss-Jordan on 1-D row registers (5 <= n <= 32).

    The working matrix and the accumulated inverse are kept as fully unrolled
    Python tuples of n vectors of length BN; every reduction is a full
    reduce of a 1-D vector that is an elementwise function of a global load,
    the only shape the XPU core-tiling pass compiles here.  Loads are upcast
    to fp32 and stores go back through the element type of Op, so the
    arithmetic runs in fp32 even for fp16/bf16 input.  A zero pivot (exactly
    singular input) propagates inf/nan through 1.0 / piv instead of a
    finite wrong inverse.
    """
    c = tl.arange(0, BN)
    cm = c < n
    abase = Ap

    M = ()
    X = ()
    for i in tl.static_range(0, n):
        M = M + (
            tl.load(abase + i * s_ar + c * s_ac, mask=cm, other=0.0).to(tl.float32),
        )
        z = M[i] * 0.0
        X = X + (tl.where(c == i, z + 1.0, z),)

    for k in tl.static_range(0, n):
        # column k of the working matrix, one scalar per row
        col = ()
        for i in tl.static_range(0, n):
            col = col + (tl.sum(tl.where(c == k, M[i], 0.0), axis=0),)

        # partial pivoting: largest magnitude among rows k..n-1, ties -> lowest
        best = tl.abs(col[k])
        p = float(k) + 0.0 * best
        for i in tl.static_range(k + 1, n):
            a = tl.abs(col[i])
            gt = a > best
            best = tl.where(gt, a, best)
            p = tl.where(gt, float(i), p)

        # pivot row, its inverse image and the pivot value, chosen before the
        # swap
        pivrow = tl.where(p == 0, M[0], M[0] * 0.0)
        pivX = tl.where(p == 0, X[0], X[0] * 0.0)
        piv = tl.where(p == 0, col[0], col[0] * 0.0)
        for i in tl.static_range(1, n):
            pivrow = tl.where(p == i, M[i], pivrow)
            pivX = tl.where(p == i, X[i], pivX)
            piv = tl.where(p == i, col[i], piv)

        # swap rows k and p (a no-op when p == k)
        rowk = M[k]
        rowkX = X[k]
        nM = ()
        nX = ()
        for i in tl.static_range(0, n):
            if i == k:
                nM = nM + (tl.where(p == k, rowk, pivrow),)
                nX = nX + (tl.where(p == k, rowkX, pivX),)
            else:
                nM = nM + (tl.where(p == i, rowk, M[i]),)
                nX = nX + (tl.where(p == i, rowkX, X[i]),)
        M = nM
        X = nX

        # normalise the pivot row; a zero pivot propagates inf/nan
        rinv = 1.0 / piv
        rnk = pivrow * rinv
        rnXk = pivX * rinv
        nM = ()
        nX = ()
        for i in tl.static_range(0, n):
            if i == k:
                nM = nM + (rnk,)
                nX = nX + (rnXk,)
            else:
                # post-swap column-k entry of row i, from the pre-swap values
                m = tl.where(p == i, col[k], col[i])
                nM = nM + (M[i] - m * rnk,)
                nX = nX + (X[i] - m * rnXk,)
        M = nM
        X = nX

    for i in tl.static_range(0, n):
        tl.store(Op + i * n + c, X[i].to(Op.dtype.element_ty), mask=cm)


def linalg_tensorinv(A, ind=2, *, out=None):
    """Compute the multiplicative inverse of tensordot.

    Args:
        A: Input tensor with prod(A.shape[:ind]) == prod(A.shape[ind:]).
        ind: Index at which to compute the inverse of tensordot.

    Returns:
        Tensor X such that tensordot(A, X, ind) is the identity matrix, with
        shape A.shape[ind:] + A.shape[:ind].
    """
    logger.debug("GEMS_KUNLUNXIN LINALG_TENSORINV")
    check_inv_input(A, ind)

    shape = A.shape
    matrix_size = 1
    for i in range(ind):
        matrix_size *= shape[i]
    output_shape = shape[ind:] + shape[:ind]
    m = matrix_size

    if m == 0:
        # Empty input: both halves of the prod requirement are 0, so the
        # result is the empty output with shape[ind:] + shape[:ind] (matches
        # torch.linalg.tensorinv).
        res = torch.empty(output_shape, dtype=A.dtype, device=A.device)
        if out is None:
            return res
        out.copy_(res)
        return out

    # Flatten to a single dense m x m matrix in row-major flat order (same flat
    # element order as B = A.reshape(m, m) in the generic entry).
    if A.is_contiguous():
        src = A
        s0 = m
        s1 = 1
    else:
        src = A.reshape(m, m)
        st = src.stride()
        s0 = st[0]
        s1 = st[1]

    if m <= _TINV_GJ_MAX:
        # Single-kernel register path: closed-form for m <= 4 (no cross-lane
        # reduction), otherwise the 1-D row-register Gauss-Jordan.
        res = torch.empty(output_shape, dtype=A.dtype, device=A.device)
        res_flat = res.reshape(m, m)
        with torch_device_fn.device(A.device):
            if m <= _TINV_SMALL_N:
                bm = max(2, triton.next_power_of_2(m))
                _tinv_small[(1,)](src, res_flat, s0, s1, m, bm, num_warps=1)
            else:
                bn = 8 if m <= 8 else triton.next_power_of_2(m)
                _tinv_row_kernel[(1,)](src, res_flat, s0, s1, m, bn, num_warps=1)
        if out is None:
            return res
        out.copy_(res)
        return out

    # Large m: LU factorisation and substitution only run in fp32/fp64.
    # fp16/bf16 inputs are upcast for the solve and downcast on store, which
    # matches the generic kernel internal fp32 tile arithmetic.
    work_dtype = (
        torch.float32 if A.dtype in (torch.float16, torch.bfloat16) else A.dtype
    )

    A_flat = A.reshape(m, m)
    if A.dtype == work_dtype:
        A_work = A_flat.contiguous()
    else:
        A_work = _cast_tensor(A_flat, work_dtype)

    # LU factorisation with partial pivoting (returns compact LU + 1-indexed
    # pivots, A = P L U).
    lu, pivots = _linalg_lu_factor(A_work, True)

    # Solve A X = I column by column: X = A^-1.  The identity right-hand side
    # is filled on device so no torch.eye materialisation is involved.
    rhs = _xpu_identity_rhs(m, 1, work_dtype, A.device)  # (1, m, m)
    with torch_device_fn.device(A.device):
        inv = _xpu_lu_solve_serial(
            lu.reshape(1, m, m), pivots.reshape(1, m), rhs, 1, m, m
        )
    inv = inv.reshape(m, m)

    result = inv.reshape(output_shape)
    if result.dtype != A.dtype:
        result = _cast_tensor(result, A.dtype)

    if out is None:
        return result
    out.copy_(result)
    return out


# ATen scalar-type names, matching the wording torch uses in the
# checkLinalgCompatibleDtype error messages ("safely castable from Float
# dtype, but got result with dtype Long").
_INTEGRAL_DTYPES = frozenset(
    (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    )
)


_ATEN_DTYPE_NAMES = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.uint8: "Byte",
    torch.bool: "Bool",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


def _tensorinv_out_castable(A_dtype, out_dtype):
    """Mirror c10::canCast(A_dtype -> out_dtype) for the .out contract.

    The ATen linalg_tensorinv.out path validates the result with
    checkLinalgCompatibleDtype: no complex -> real, no floating -> integral,
    no non-bool -> bool; everything else (including fp16/bf16/fp64 outs for a
    floating input) is accepted and cast on write.
    """
    if out_dtype == A_dtype:
        return True
    if A_dtype.is_complex and not out_dtype.is_complex:
        return False
    if A_dtype.is_floating_point and out_dtype in _INTEGRAL_DTYPES:
        return False
    if out_dtype == torch.bool:
        return False
    return True


def _check_tensorinv_out(A, ind, out):
    """Validate out against the ATen linalg_tensorinv.out contract.

    Ordering mirrors the native path: result dtype castability first, then the
    same-device requirement, then the input validation (a bad ind never
    mutates the caller tensor), and only then the in-place resize for a shape
    mismatch, which replicates at::native::resize_output on this torch build
    (including its deprecation warning for non-empty tensors).
    """
    out_name = _ATEN_DTYPE_NAMES.get(out.dtype, str(out.dtype))
    a_name = _ATEN_DTYPE_NAMES.get(A.dtype, str(A.dtype))
    if not _tensorinv_out_castable(A.dtype, out.dtype):
        raise RuntimeError(
            "tensorinv: Expected result to be safely castable from "
            f"{a_name} dtype, but got result with dtype {out_name}"
        )
    if out.device != A.device:
        raise RuntimeError(
            "tensorinv: Expected result and input tensors to be on the same "
            f"device, but got result on {out.device} and input on {A.device}"
        )
    check_inv_input(A, ind)
    output_shape = A.shape[ind:] + A.shape[:ind]
    if tuple(out.shape) != output_shape:
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(out.shape)}, which does not match the required "
                f"output shape {list(output_shape)}. This behavior is "
                "deprecated, and in a future PyTorch release outputs will not "
                "be resized unless they have zero elements. You can explicitly "
                "reuse an out tensor t by resizing it, inplace, to zero "
                "elements with t.resize_(0).",
                stacklevel=2,
            )
        out.resize_(output_shape)


def linalg_tensorinv_out(A, ind=2, *, out=None):
    """Out-of-place variant of linalg_tensorinv: computes the tensor inverse
    and writes the result into the provided out tensor."""
    logger.debug("GEMS_KUNLUNXIN LINALG_TENSORINV_OUT")
    if out is not None:
        _check_tensorinv_out(A, ind, out)
    return linalg_tensorinv(A, ind=ind, out=out)


__all__ = ["linalg_tensorinv", "linalg_tensorinv_out"]
