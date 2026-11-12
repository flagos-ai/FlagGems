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

"""Kunlunxin backend override for ``linalg_polar``.

The generic implementation calls ``flag_gems.ops.svd``, whose generic Triton
kernel is not supported on this backend (unbound loop bounds / large register
tiles).  The backend already ships a working one-sided Jacobi SVD pipeline in
``_kunlunxin/ops/linalg_svd.py`` (``_osj_pipeline``) that produces, in the raw
``B`` buffer, the rotated columns ``A v_j`` (unnormalised) and, in the ``U``
buffer, the normalised left singular vectors ``Up``.

For a tall matrix ``A = Up diag(S) Vh`` (``m >= n``) the right polar factors are

    U = Up Vh                     (m x n, orthonormal columns)
    H = Vh^T diag(S) Vh           (n x n, symmetric positive semidefinite)

Since ``Up`` has orthonormal columns, ``Vh = diag(1/S) Up^T A``, and the column
norms of the raw ``B`` give ``S``.  All post-processing is done in Triton
(``tl.dot``), so no host-side linear algebra is used and no sorted singular
values are required (polar is invariant to the singular-value ordering, unlike
``svd``).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_polar import _check_out, _copy_kernel, _validate_input
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .linalg_svd import _osj_pipeline

logger = logging.getLogger(__name__)

# One-sided Jacobi sweep budget; the polar factors converge to well below the
# 1e-2 tolerance used by the accuracy suite for the supported shapes.
_SWEEPS = 12


@libentry()
@triton.jit
def _polar_osj_kernel(
    B,
    A,
    UOUT,
    HOUT,
    m,
    n,
    BSTRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Turn the raw rotated SVD columns ``B`` into the polar factors."""
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    rmask = rows < m
    cmask = cols < n
    mask = rmask[:, None] & cmask[None, :]
    srow = tl.where(rmask, rows, 0)
    scol = tl.where(cmask, cols, 0)

    # Masked loads may ignore ``other`` on this backend, so explicitly zero the
    # padding lanes after the (safely indexed) load.
    b = tl.load(B + srow[:, None] * BSTRIDE + scol[None, :], mask=mask, other=0.0)
    b = tl.where(mask, b, 0.0)
    a = tl.load(A + srow[:, None] * n + scol[None, :], mask=mask, other=0.0)
    a = tl.where(mask, a, 0.0)

    # Column norms of the raw rotated matrix are the singular values.
    s2 = tl.sum(b * b, axis=0)
    s = tl.sqrt(s2)
    inv = tl.where(s > 1.0e-20, 1.0 / s, 0.0)
    up = b * inv[None, :]

    # Vh = diag(1/S) Up^T A
    p = tl.dot(tl.trans(up), a, allow_tf32=False)
    vh = p * inv[:, None]

    # U = Up Vh
    u = tl.dot(up, vh, allow_tf32=False)

    # H = Vh^T diag(S) Vh, then symmetrise.
    svh = vh * s[:, None]
    h = tl.dot(tl.trans(vh), svh, allow_tf32=False)
    h = 0.5 * (h + tl.trans(h))

    tl.store(UOUT + rows[:, None] * n + cols[None, :], u, mask=mask)
    hmask = cmask[:, None] & cmask[None, :]
    tl.store(HOUT + rows[:, None] * n + cols[None, :], h, mask=hmask)


def _polar_osj(A):
    """Compute ``(U, H)`` for a contiguous ``(*, m, n)`` tensor with ``m >= n``."""
    orig_shape = A.shape
    m = A.shape[-2]
    n = A.shape[-1]
    if A.dim() == 2:
        A2 = A.unsqueeze(0)
    else:
        A2 = A.reshape(-1, m, n)
    batch = A2.shape[0]

    nw = n if n % 2 == 0 else n + 1
    NW = nw if (nw & (nw - 1)) == 0 else triton.next_power_of_2(nw)
    MP = triton.next_power_of_2(m)
    total = _SWEEPS * (nw - 1) * (nw // 2)
    block = max(16, triton.next_power_of_2(m))
    dev = A.device

    B = torch.empty((batch, MP * NW), device=dev, dtype=A.dtype)
    scratch = torch.empty((MP * NW,), device=dev, dtype=A.dtype)
    U3 = torch.empty((batch, m, n), device=dev, dtype=A.dtype)
    H3 = torch.empty((batch, n, n), device=dev, dtype=A.dtype)

    with torch_device_fn.device(dev):
        for b in range(batch):
            _osj_pipeline[(1,)](
                A2[b],
                B[b],
                scratch,
                m,
                n,
                nw,
                total,
                MP=MP,
                NW=NW,
                num_warps=1,
                num_stages=1,
            )
            _polar_osj_kernel[(1,)](
                B[b],
                A2[b],
                U3[b],
                H3[b],
                m,
                n,
                BSTRIDE=NW,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )

    U = U3.reshape(orig_shape)
    H = H3.reshape(*orig_shape[:-2], n, n)
    return U, H


def _linalg_polar_impl(A, out_U=None, out_H=None):
    _validate_input(A)
    n = A.shape[-1]
    U_shape = A.shape
    H_shape = (*A.shape[:-2], n, n)

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

    Ac = A.contiguous()
    U, H = _polar_osj(Ac)

    if out_U is not None:
        _copy_kernel(U, out0=out_U)
        _copy_kernel(H, out0=out_H)
        return out_U, out_H
    return U.contiguous(), H.contiguous()


def linalg_polar(A):
    logger.debug("GEMS_KUNLUNXIN LINALG_POLAR")
    return _linalg_polar_impl(A)


def linalg_polar_out(A, *, U, H):
    logger.debug("GEMS_KUNLUNXIN LINALG_POLAR_OUT")
    return _linalg_polar_impl(A, U, H)


__all__ = ["linalg_polar", "linalg_polar_out"]
