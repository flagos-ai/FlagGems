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
# 1e-2 tolerance used by the accuracy suite for the supported shapes (measured
# ~1e-6 at _SWEEPS=3..6 on the benchmark matrix).  A lower budget directly cuts
# the dominant _osj_pipeline runtime (98.8% of the op) for the loop-bound cases.
_SWEEPS = 6


@libentry()
@triton.jit
def _polar_vh_kernel(BP, AP, VH, m, n, BLOCK: tl.constexpr):
    """``Vh = diag(1/S^2) B^T A`` — one ``tl.dot`` per kernel.

    XPU's Triton pipeline hangs the device task when a kernel carries two or
    more ``tl.dot`` ops (measured: one dot runs, two+ never return and are
    killed by the driver watchdog), so the polar post-processing is split into
    three single-dot kernels.  Only constructs verified on this backend are
    used: row-vector broadcasts ``x * v[None, :]`` and ``tl.trans`` as a
    ``tl.dot`` operand (a standalone ``tl.trans`` and ``(BLOCK, 1)`` column
    broadcasts miscompile here).  ``BP`` / ``AP`` arrive zero-padded to
    ``BLOCK x BLOCK``; ``VH`` / ``UB`` are zero-initialised so the padding
    lanes the masked stores skip stay zero for the downstream kernels.
    """
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    rmask = rows < m
    cmask = cols < n
    mask = rmask[:, None] & cmask[None, :]

    b = tl.load(BP + rows[:, None] * BLOCK + cols[None, :])
    a = tl.load(AP + rows[:, None] * BLOCK + cols[None, :])

    # Column norms of the raw rotated matrix are the singular values, and
    # Vh = diag(1/S) Up^T A = diag(1/S^2) B^T A (Up = B diag(1/S)).
    s2 = tl.sum(b * b, axis=0)
    s = tl.sqrt(s2)
    inv = (1.0 / tl.maximum(s, 1.0e-20)) * (s > 1.0e-20)
    vh = tl.dot(tl.trans(b * inv[None, :] * inv[None, :]), a, allow_tf32=False)
    tl.store(VH + rows[:, None] * BLOCK + cols[None, :], vh, mask=mask)


@libentry()
@triton.jit
def _polar_u_kernel(BP, VH, UOUT, UB, m, n, BLOCK: tl.constexpr):
    """``U = Up Vh`` — one ``tl.dot``; ``UB`` is ``U`` padded to ``BLOCK``."""
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    rmask = rows < m
    cmask = cols < n
    mask = rmask[:, None] & cmask[None, :]

    b = tl.load(BP + rows[:, None] * BLOCK + cols[None, :])
    vh = tl.load(VH + rows[:, None] * BLOCK + cols[None, :])

    s2 = tl.sum(b * b, axis=0)
    s = tl.sqrt(s2)
    inv = (1.0 / tl.maximum(s, 1.0e-20)) * (s > 1.0e-20)
    up = b * inv[None, :]

    # U = Up Vh
    u = tl.dot(up, vh, allow_tf32=False)
    tl.store(UOUT + rows[:, None] * n + cols[None, :], u, mask=mask)
    tl.store(UB + rows[:, None] * BLOCK + cols[None, :], u, mask=mask)


@libentry()
@triton.jit
def _polar_h_kernel(UB, AP, HOUT, m, n, BLOCK: tl.constexpr):
    """``H = U^T A`` (polar: ``A = U H``, ``H`` symmetric PSD) — one ``tl.dot``."""
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    cmask = cols < n
    hmask = cmask[:, None] & cmask[None, :]

    u = tl.load(UB + rows[:, None] * BLOCK + cols[None, :])
    a = tl.load(AP + rows[:, None] * BLOCK + cols[None, :])

    # H = U^T A
    h = tl.dot(tl.trans(u), a, allow_tf32=False)
    tl.store(HOUT + rows[:, None] * n + cols[None, :], h, mask=hmask)


def _polar_osj(A, out_U=None, out_H=None):
    """Compute ``(U, H)`` for a contiguous ``(*, m, n)`` tensor with ``m >= n``.

    When contiguous ``out_U`` / ``out_H`` are supplied, the kernels write the
    polar factors directly into them (avoids a pair of copy launches for the
    small matrices that dominate the latency-bound core benchmark).
    """
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
    # Zero-padded (BLOCK x BLOCK) tiles for the polar post-processing kernels:
    # padding lanes must be zero in memory (masked loads may ignore ``other``
    # on this backend), and ``VH`` must be zero outside the masked store region
    # because the ``u`` / ``h`` kernels read the whole tile.
    BP = torch.zeros((batch, block, block), device=dev, dtype=A.dtype)
    AP = torch.zeros((batch, block, block), device=dev, dtype=A.dtype)
    VH = torch.zeros((batch, block, block), device=dev, dtype=A.dtype)
    UB = torch.zeros((batch, block, block), device=dev, dtype=A.dtype)
    if out_U is not None:
        U3 = out_U.reshape(batch, m, n)
        H3 = out_H.reshape(batch, n, n)
    else:
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
            BP[b, :MP, :NW].copy_(B[b].reshape(MP, NW))
            AP[b, :m, :n].copy_(A2[b])
            _polar_vh_kernel[(1,)](
                BP[b],
                AP[b],
                VH[b],
                m,
                n,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )
            _polar_u_kernel[(1,)](
                BP[b],
                VH[b],
                U3[b],
                UB[b],
                m,
                n,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )
            _polar_h_kernel[(1,)](
                UB[b],
                AP[b],
                H3[b],
                m,
                n,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )

    if out_U is not None:
        return out_U, out_H
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

    direct_out = out_U is not None and out_U.is_contiguous() and out_H.is_contiguous()
    if direct_out:
        return _polar_osj(Ac, out_U, out_H)

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
