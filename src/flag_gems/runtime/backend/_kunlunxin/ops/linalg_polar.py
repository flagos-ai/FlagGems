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
"""Kunlunxin linalg_polar (aten::linalg_polar) vendor override.

The right polar decomposition ``A = U H`` (``U`` orthogonal, ``H`` symmetric
positive-semidefinite) is computed from a thin SVD ``A = Up diag(S) Vh``:

    U = Up @ Vh
    H = Vh^T @ diag(S) @ Vh

The generic ``src/flag_gems/ops/linalg_polar.py`` routes into the generic
``ops/svd.py`` (``from flag_gems.ops.svd import svd``).  That generic SVD is
*not* usable on XPU3: for the small tall matrices this op feeds it
(``m > n``, ``max(m, n) <= 16``) every specialized kernel either fails to lower
(``_small_jacobi_svd`` -> ``ConvertTritonXPUToLLVM`` error, swallowed into
``_unsupported_svd``) or the dispatch raises ``NotImplementedError`` outright.
So HEAD raises ``NotImplementedError`` for every non-trivial shape (baseline
``tests/test_linalg_polar.py -m linalg_polar`` = 8 failed / 6 passed).

This override keeps the host-side contract of the generic implementation
(validation, empty handling, ``out=`` resize/copy) but reuses the *working*
backend one-sided Jacobi SVD (``_kunlunxin.ops.linalg_svd``) and Gems
``bmm``/pointwise kernels:

  - ``linalg_svd`` returns the thin factors; we deliberately **ignore** its
    ``Vh`` (it is ``U^T A / S`` with an unguarded ``1/S`` and so becomes NaN on
    rank-deficient input) and recompute ``Vh = U^T A * invS`` with an
    ``invS = where(S > eps, 1/S, 0)`` guard, which keeps the zero matrix exact
    (``U = H = 0``) for ``test_linalg_polar_rank_deficient``.
  - SVD column ordering is irrelevant for the polar factors (``H`` and ``U``
    are invariant under a joint permutation of ``(S, Up, Vh)``), so no host
    sort is required on our side -- ``linalg_svd`` already sorts internally.
  - ``Vh^T`` is materialised by ``bmm``'s internal ``.contiguous()``; the
    scaling/symmetrisation use ``pointwise_dynamic`` kernels (the same Triton
    helpers the generic implementation uses).

Platform constraints honoured: no ``tl.trans`` in a hand-written kernel (it is
a silent no-op on XPU3, so all transposes go through torch views consumed by
the backend ``bmm`` which sanitises strides), no CPU/ATen compute forwarding,
and fp64 never reaches the math (the platform down-casts fp64 to fp32 at
allocation, so ``linalg_polar`` here is fp32-only like the generic contract).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

# NOTE: absolute (not relative) imports on purpose.  ``pointwise_dynamic``
# re-emits this module's ``from X import Y`` lines into its generated kernel
# module, and a relative ``from .bmm import bmm`` becomes ``from bmm import
# bmm`` there, which fails at codegen import time.
from flag_gems.runtime.backend._kunlunxin.ops.bmm import bmm, bmm_out
from flag_gems.runtime.backend._kunlunxin.ops.linalg_svd import linalg_svd

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _scale_rows_kernel(value, scale):
    return value * scale


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _symmetrize_kernel(value, transposed):
    return 0.5 * (value + transposed)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_kernel(value):
    return value


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _inv_guard_kernel(value):
    return tl.where(value > 1.0e-20, 1.0 / value, 0.0)


def _validate_input(A):
    if A.ndim < 2:
        raise RuntimeError(
            "linalg.polar: The input tensor A must have at least 2 dimensions."
        )
    if A.shape[-2] < A.shape[-1]:
        raise RuntimeError(
            "linalg.polar: input must have at least as many rows as columns, "
            f"but got {A.shape[-2]} by {A.shape[-1]} matrices"
        )
    if A.dtype != torch.float32:
        raise TypeError(f"linalg_polar only supports float32 input, got {A.dtype}")


def _check_out(A, out, name):
    if out.dtype != A.dtype:
        raise RuntimeError(
            f"Expected out tensor {name} to have dtype {A.dtype}, got {out.dtype}"
        )
    if out.device != A.device:
        raise RuntimeError(
            f"Expected out tensor {name} to be on {A.device}, got {out.device}"
        )


def _polar_from_svd(A, batch_shape, m, n, out_U=None):
    """Return ``(U, H)`` for the flattened ``(B, m, n)`` view of ``A``.

    Reuses the backend one-sided Jacobi SVD for the singular triple and forms
    the polar factors with Gems matmuls.  ``A`` must be ``(B, m, n)`` shaped.

    ``out_U``, when given, must be a **contiguous** ``(batch, m, n)`` tensor
    sized to receive ``U``; the final matmul is then written straight into it
    (``bmm_out``) so the ``.out`` overload does not need a temp + copy.  ``H``
    is returned *unsymmetrised* so the caller can fuse the symmetrisation with
    the store into its own ``out`` buffer.
    """
    batch = A.shape[0]
    with torch_device_fn.device(A.device):
        Up, S, _ = linalg_svd(A, full_matrices=False)
    if Up.dim() == 2:  # linalg_svd squeezes the batch dim when B == 1
        Up = Up.unsqueeze(0)
        S = S.unsqueeze(0)

    # Vh = U^T A diag(1/S), with a rank-guarded reciprocal so a zero (or
    # rank-deficient) A yields Vh = 0 instead of NaN.
    UtA = bmm(Up.transpose(-2, -1), A)  # (B, n, n)
    invS = _inv_guard_kernel(S)  # (B, n)
    Vh = _scale_rows_kernel(UtA, invS.unsqueeze(-1))  # (B, n, n)

    # ``U`` is the last consumer of the singular triple (H only needs Vh/S), so
    # a contiguous caller buffer can be the matmul destination directly.
    if out_U is not None:
        U = bmm_out(Up.contiguous(), Vh.contiguous(), out_U.view(batch, m, n))
    else:
        U = bmm(Up, Vh)  # (B, m, n)

    # H = Vh^T diag(S) Vh (symmetrisation is applied by the caller).
    scaled = _scale_rows_kernel(Vh, S.unsqueeze(-1))  # diag(S) @ Vh
    H = bmm(Vh.transpose(-2, -1), scaled)  # (B, n, n)

    U = U.reshape(*batch_shape, m, n)
    H = H.reshape(*batch_shape, n, n)
    return U, H


def _linalg_polar_impl(A, out_U=None, out_H=None):
    _validate_input(A)
    m = A.shape[-2]
    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    U_shape = A.shape
    H_shape = (*batch_shape, n, n)

    if out_U is not None:
        _check_out(A, out_U, "U")
        _check_out(A, out_H, "H")
        if out_U.shape != U_shape:
            out_U.resize_(U_shape)
        if out_H.shape != H_shape:
            out_H.resize_(H_shape)

    if A.numel() == 0:
        U = out_U
        H = out_H
        if U is None:
            U = torch.empty_like(A, memory_format=torch.contiguous_format)
            H = torch.empty(H_shape, dtype=A.dtype, device=A.device)
        return U, H

    # ``linalg_svd``'s one-sided Jacobi kernel indexes A as dense row-major, so
    # a non-contiguous (e.g. transposed) input must be staged contiguous first;
    # ``reshape`` alone can keep the transposed strides as a view.
    A3 = A.contiguous().reshape(-1, m, n)
    # ``bmm_out`` needs a dense destination; a strided ``out`` buffer falls
    # back to an explicit pointwise copy of ``U`` after the matmul.
    direct_U = out_U is not None and out_U.is_contiguous()
    U, H = _polar_from_svd(A3, batch_shape, m, n, out_U if direct_U else None)

    if out_U is not None:
        if not direct_U:
            _copy_kernel(U, out0=out_U)
        # ``H`` is finished by the symmetrisation, so fuse that pointwise op
        # with the store into ``out_H`` (pointwise stores honour its strides).
        _symmetrize_kernel(H, H.transpose(-2, -1), out0=out_H)
        return out_U, out_H

    H = _symmetrize_kernel(H, H.transpose(-2, -1))
    return U.contiguous(), H.contiguous()


def linalg_polar(A):
    logger.debug("GEMS_KUNLUNXIN LINALG_POLAR")
    return _linalg_polar_impl(A)


def linalg_polar_out(A, *, U, H):
    logger.debug("GEMS_KUNLUNXIN LINALG_POLAR_OUT")
    return _linalg_polar_impl(A, U, H)
