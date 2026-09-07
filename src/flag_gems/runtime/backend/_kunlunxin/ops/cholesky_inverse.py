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

"""Kunlunxin backend override for cholesky_inverse.

The generic implementation pins one program per batch element and walks the
whole matrix with scalar loads/stores in triply nested loops (O(N^3) scalar
ops in a single program), which is unusable on TritonXPU (baseline speedup
~0.005x at N=64, ~0.0002x at N=128). This override computes the inverse with
a vectorized masked serial forward substitution plus a tiled tl.dot product,
which is what the TritonXPU backend can lower correctly:

  - no tl.gather: tt.gather is explicitly illegal on this backend;
  - tl.dot lowers correctly and is exact fp32 with input_precision="ieee"
    (probe at N=32..256, maxdiff <= 1.2e-5 vs fp64 reference);
  - serial per-row walks are vectorized across columns with vector width <=
    CBLK=64 lanes (128-wide unmasked vectors miscompile, and PAD=256 vector
    loops fail to lower in TritonXPUUnrollControl/uni_sram, reproduced at
    N=192/256);
  - 2D row/col masks INSIDE tl.dot operand loads blow the same
    uni_sram/UnrollControl pass (probed), so the dot-based product only runs
    for N % 64 == 0 shapes and stays fully mask-free; other shapes use the
    serial product kernel.

Algorithm (lower triangular factor L; an upper factor is first transposed
with a native transposed copy so the pipeline always runs on a lower one):

  X = L^{-1} via a masked serial vectorized triangular inverse (one program
  per (batch element, column band), forward substitution row by row);
  Out = X^T X as a tiled tl.dot over a native transposed copy of X
  (full symmetric output; no mirrored stores needed).

Accuracy: the masked serial inverse is accurate to ~5e-9 vs an fp64
reference at every size (verified N=32..256, batched and upper variants).
An earlier blocked variant (unmasked 64x64 diagonal blocks plus a tiled
tl.dot off-diagonal sweep) is NOT usable on this backend: it produced wrong
off-diagonal blocks (~3.5e-3 at N=128) from a broken tl.dot accumulate, so
only the two kernels below are kept.

dtype: only float32/float64 are legal per ATen; on the Kunlunxin xpu
backend fp64 inputs never reach the kernels (torch.linalg.cholesky
silently returns fp32 for fp64 requests), the kernels are dtype-generic.
"""

import torch
import triton
import triton.language as tl

_CBLK = 64  # max vector lanes for the serial kernels (compile safe;
# 128-wide unmasked vectors miscompile on TritonXPU)
_BS = 64  # blocked-path block size


@triton.jit
def _ci_tri_serial_kernel(
    A_ptr,
    X_ptr,
    N: tl.constexpr,
    CBLK: tl.constexpr,
    batch_stride,
    row_stride,
    col_stride,
):
    """Serial triangular inverse.

    Grid = (batch, cdiv(N, CBLK)); each program owns one column band
    [blk*CBLK, blk*CBLK+CBLK) and walks all rows serially; prior rows are
    re-read from global memory. No gather/dot/reduce.
    """
    b = tl.program_id(0)
    blk = tl.program_id(1)
    base = b * batch_stride
    js = blk * CBLK + tl.arange(0, CBLK)
    jmask = js < N
    dtype = A_ptr.dtype.element_ty

    for i in range(N):
        diag = tl.load(A_ptr + base + i * row_stride + i * col_stride)
        inv_d = 1.0 / diag
        acc = tl.zeros((CBLK,), dtype=dtype)
        for k in range(i):
            f = tl.load(A_ptr + base + i * row_stride + k * col_stride)
            pk = tl.load(
                X_ptr + base + k * row_stride + js * col_stride,
                mask=jmask,
                other=0.0,
            )
            acc += f * pk
        val = tl.where(js == i, inv_d, tl.where(js < i, -acc * inv_d, 0.0))
        tl.store(
            X_ptr + base + i * row_stride + js * col_stride,
            val,
            mask=jmask,
        )


@triton.jit
def _ci_mm_serial_kernel(
    A_ptr,
    Out_ptr,
    N: tl.constexpr,
    CBLK: tl.constexpr,
    batch_stride,
    a_row_stride,
    a_col_stride,
    o_row_stride,
    o_col_stride,
):
    """Symmetric product Out = A^T A for lower-triangular A (serial path).

    Grid = (batch, cdiv(N, CBLK)); each program owns one output column band.
    """
    b = tl.program_id(0)
    blk = tl.program_id(1)
    base = b * batch_stride
    js = blk * CBLK + tl.arange(0, CBLK)
    jmask = js < N
    dtype = A_ptr.dtype.element_ty

    for i in range(N):
        acc = tl.zeros((CBLK,), dtype=dtype)
        for k in range(i, N):
            f = tl.load(A_ptr + base + k * a_row_stride + i * a_col_stride)
            pk = tl.load(
                A_ptr + base + k * a_row_stride + js * a_col_stride,
                mask=jmask,
                other=0.0,
            )
            acc += f * pk
        tl.store(
            Out_ptr + base + i * o_row_stride + js * o_col_stride,
            acc,
            mask=(js <= i) & jmask,
        )
        tl.store(
            Out_ptr + base + js * o_row_stride + i * o_col_stride,
            acc,
            mask=(js < i) & jmask,
        )


@triton.jit
def _ci_mm_dot_kernel(
    AT_ptr,
    A_ptr,
    Out_ptr,
    N: tl.constexpr,
    TM: tl.constexpr,
    batch_stride,
):
    """Out = AT @ A (full matrix, no masks; N % TM == 0 by wrapper gate)."""
    b = tl.program_id(0)
    pm = tl.program_id(1)
    pn = tl.program_id(2)
    offm = pm * TM + tl.arange(0, TM)
    offn = pn * TM + tl.arange(0, TM)
    base = b * batch_stride
    acc = tl.zeros((TM, TM), dtype=A_ptr.dtype.element_ty)
    for k in range(0, N, TM):
        kk = k + tl.arange(0, TM)
        a = tl.load(AT_ptr + base + offm[:, None] * N + kk[None, :])
        bx = tl.load(A_ptr + base + kk[:, None] * N + offn[None, :])
        acc = tl.dot(a, bx, acc, input_precision="ieee")
    tl.store(Out_ptr + base + offm[:, None] * N + offn[None, :], acc)


def _run_serial(A):
    """Serial path: triangular inverse + symmetric product (any N)."""
    batch_size = A.shape[0]
    n = A.shape[1]
    batch_stride = A.stride(0) if batch_size > 1 else 0
    cblk = min(_CBLK, triton.next_power_of_2(n))
    num_bands = triton.cdiv(n, cblk)
    A_inv = torch.zeros_like(A)
    _ci_tri_serial_kernel[(batch_size, num_bands)](
        A,
        A_inv,
        n,
        cblk,
        batch_stride,
        A.stride(1),
        A.stride(2),
        num_warps=4 if cblk > 32 else 1,
        num_stages=1,
    )
    output = torch.empty_like(A)
    _ci_mm_serial_kernel[(batch_size, num_bands)](
        A_inv,
        output,
        n,
        cblk,
        batch_stride,
        A_inv.stride(1),
        A_inv.stride(2),
        output.stride(1),
        output.stride(2),
        num_warps=4 if cblk > 32 else 1,
        num_stages=1,
    )
    return output


def _run_blocked(A):
    """Blocked path (N % 64 == 0): vectorized masked serial inverse of the
    full matrix + tiled-dot symmetric product.

    NOTE: the original blocked implementation (diagonal 64x64 blocks via the
    unmasked ``_ci_diag_block_kernel`` plus ``_ci_offdiag_block_kernel`` for
    the off-diagonal block columns) is numerically wrong on TritonXPU: the
    off-diagonal blocks come out with ~3.5e-3 abs error at N=128 (exact
    sub-diagonal zeros from the unmasked diag kernel, plus a wrong
    ``tl.dot`` accumulate in the offdiag sweep), which propagates to
    ~2.4e-4 on the final product. The masked serial inverse below is accurate
    to ~5e-9 at every size (verified N=32..256, batch and upper variants);
    it is O(N^3) scalar work per band but stays within what the backend
    lowers correctly."""
    batch_size = A.shape[0]
    n = A.shape[1]
    batch_stride = n * n if batch_size > 1 else 0

    X = torch.zeros_like(A)
    _ci_tri_serial_kernel[(batch_size, n // _BS)](
        A,
        X,
        n,
        _BS,
        batch_stride,
        A.stride(1),
        A.stride(2),
        num_warps=4,
        num_stages=1,
    )

    X_T = X.transpose(-2, -1).contiguous()
    output = torch.empty_like(A)
    nt = n // _BS
    _ci_mm_dot_kernel[(batch_size, nt, nt)](
        X_T,
        X,
        output,
        n,
        _BS,
        batch_stride,
        num_warps=4,
        num_stages=1,
    )
    return output


def cholesky_inverse(a, upper=False):
    """Compute the inverse of a symmetric positive-definite matrix from its
    Cholesky factor (lower by default, or upper)."""
    assert a.dtype in (
        torch.float32,
        torch.float64,
    ), "cholesky_inverse only supports float32 and float64"

    if a.numel() == 0:
        return a

    shape = a.shape
    if len(shape) < 2:
        raise ValueError("Input must be at least 2D")

    n = shape[-1]
    m = shape[-2]
    if n != m:
        raise ValueError("Input must be a square matrix")

    batch_size = 1
    for dim in shape[:-2]:
        batch_size *= dim

    # Bring the factor to a lower-triangular view (upper via native copy).
    if upper:
        A = a.transpose(-2, -1).contiguous().reshape(batch_size, n, n)
    else:
        A = a.reshape(batch_size, n, n)
        if not A.is_contiguous():
            A = A.contiguous()

    if n >= _BS and n % _BS == 0:
        output = _run_blocked(A)
    else:
        output = _run_serial(A)

    return output.reshape(shape)
