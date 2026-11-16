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

"""Kunlunxin backend override for ``linalg_tensorinv`` / ``linalg_tensorinv.out``.

The generic ``_tinv_reg_kernel`` keeps a full ``BLOCK_N``-by-``BLOCK_N``
register tile live across a data-dependent serial pivot loop.  The
``TritonXPULegalize`` MLIR pipeline crashes on exactly that combination on this
backend (same class of failure as ``linalg_inv`` / ``linalg_eig``:
``'arith.cmpi' op failed to verify ...``), so the generic entry point cannot
compile here.

``tensorinv`` reduces to a single dense ``m x m`` inversion (``m =
prod(shape[:ind])``).  This override composes the two pieces that already
compile on the backend:

* ``_linalg_lu_factor`` — the O(k) pivoted elimination already shipped as the
  ``linalg_lu_factor`` override (pure Triton, no fallback), returning compact
  ``LU`` factors and 1-indexed pivots;
* ``_xpu_lu_solve_serial`` — the register-resident column substitution kernel
  from the generic ``linalg_lu_solve`` (``flag_gems.ops.linalg_lu_solve``),
  which is the XPU production path for ``N <= 128``.

``A^-1`` is obtained by solving ``A X = I`` column by column, so no register
tile survives across the pivot loop.  ``tensorinv`` has no batch axis, so this
is a single-matrix solve; the largest ``m`` in the test/benchmark matrix is 80
(well within the ``N <= 128`` register-resident range).  fp16/bf16 inputs are
upcast to fp32 for the factorization + solve and downcast on store via a
Triton cast kernel, mirroring the generic kernel's internal fp32 arithmetic.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_lu_solve import _xpu_lu_solve_serial
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .linalg_lu_factor import _linalg_lu_factor

logger = logging.getLogger(__name__)


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
    """Cast ``x`` to ``dtype`` via a Triton kernel.

    Replaces ``x.to(dtype)`` (which dispatches to the aten ``_to_copy`` cast on
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
    violation, matching torch.linalg.tensorinv's behaviour.
    """
    # Checked first, as in torch: a non-positive ind is rejected before any
    # shape reasoning, so ind=0 cannot slip through to the prod comparison
    # below (which it would satisfy trivially for a 1x1 input, yielding a
    # silently wrong result instead of an error).
    if ind <= 0:
        raise RuntimeError(
            "linalg.tensorinv: Expected a strictly positive integer for "
            f"'ind', but got {ind}"
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

    # The LU factorisation and substitution only run in fp32/fp64.  fp16/bf16
    # inputs are upcast for the solve and downcast on store, which matches the
    # generic kernel's internal fp32 tile arithmetic.
    work_dtype = (
        torch.float32 if A.dtype in (torch.float16, torch.bfloat16) else A.dtype
    )

    # Flatten to a single dense m x m matrix in row-major flat order (same flat
    # element order as B = A.reshape(m, m) in the generic entry).
    A_flat = A.reshape(m, m)
    if A.dtype == work_dtype:
        A_work = A_flat.contiguous()
    else:
        A_work = _cast_tensor(A_flat, work_dtype)

    # LU factorisation with partial pivoting (returns compact LU + 1-indexed
    # pivots, A = P L U).
    lu, pivots = _linalg_lu_factor(A_work, True)

    # Solve A X = I column by column: X = A^-1.
    eye = torch.eye(m, dtype=work_dtype, device=A.device)
    rhs = eye.unsqueeze(0)  # (1, m, m)
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


def linalg_tensorinv_out(A, ind=2, *, out=None):
    """Out-of-place variant of linalg_tensorinv: computes the tensor inverse
    and writes the result into the provided ``out`` tensor."""
    logger.debug("GEMS_KUNLUNXIN LINALG_TENSORINV_OUT")
    return linalg_tensorinv(A, ind=ind, out=out)


__all__ = ["linalg_tensorinv", "linalg_tensorinv_out"]
