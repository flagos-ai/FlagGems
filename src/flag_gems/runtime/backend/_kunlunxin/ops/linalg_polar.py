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
``B`` buffer, the rotated columns ``A v_j`` (unnormalised).  For a tall matrix
``A = Up diag(S) Vh`` (``m >= n``) the right polar factors are

    U = Up Vh                     (m x n, orthonormal columns)
    H = Vh^T diag(S) Vh           (n x n, symmetric positive semidefinite)

With ``T = B^T A``, ``Vh = diag(1/S^2) T`` and ``Up = B diag(1/S)``, so
``U = B diag(1/S^3) B^T A`` and ``H = U^T A``; the column norms of ``B`` are
the singular values ``S``.  All post-processing is done in Triton so no
host-side linear algebra is used and no sorted singular values are required
(polar is invariant to the singular-value ordering, unlike ``svd``).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_polar import _check_out, _copy_kernel, _validate_input
from flag_gems.runtime import torch_device_fn

from .linalg_svd import _osj_pipeline

logger = logging.getLogger(__name__)

# One-sided Jacobi sweep budget; the polar factors converge to well below the
# 1e-2 tolerance used by the accuracy suite for the supported shapes (measured
# ~1e-6 at _SWEEPS=3..6 on the benchmark matrix).  A lower budget directly cuts
# the dominant _osj_pipeline runtime (98.8% of the op) for the loop-bound cases.
_SWEEPS = 6


# NOTE(2026-11-12): every ``tl.dot`` operand below is a plain load (or
# ``tl.trans`` of a load) and every diagonal scaling happens in pure
# elementwise kernels.  Two platform constraints drive this:
#
# * flagtree's ``TritonSDNNOptimizeRcLayout`` crashes (``sdnn.core_specialize``
#   type mismatch, reported as OutOfResources) when a dot operand is produced
#   from an in-kernel reduction (``1/S`` derived from ``tl.sum``); the RC
#   layout it back-propagates cannot enter the reduction region.  Scaling is
#   therefore hoisted into pure kernels that write plain buffers, and the
#   reductions live only in kernels whose dot operands are plain loads.
# * the local sdnn-build Triton does not order dependent back-to-back dot
#   kernels (the second dot reads the first one's stale output) and
#   ``@libentry()`` makes it worse, so these kernels are plain ``@triton.jit``
#   and the single dot-to-dot edge (``U`` -> ``H``) is bridged by a torch copy
#   of the produced tile, the only reliable cross-kernel barrier there.
#
# Chain (per matrix, all ``tl.dot`` operands plain loads):
#
#     T (dot: B^T A)  ->  Bs/Vh (pure scalings)  ->  U (dot: Bs Vh)
#                    ->  UB.copy_(UB2)           ->  H (dot: U^T A)
#
# ``tl.sum`` axis reductions crash in a kernel with no ``tl.dot`` (MLIR
# UnrollControl type error) and feeding a ``tl.dot`` result into a second
# ``tl.dot`` hangs the device task (702/KEXCEPTION), so each dot kernel carries
# exactly one ``tl.dot``.


@triton.jit
def _polar_t_kernel(BP, AP, T, INV, m, n, ONE, BLOCK: tl.constexpr):
    """``T = B^T A`` and the reciprocal singular values ``1/S``.

    ``T`` is stored zero-padded to ``BLOCK x BLOCK`` (the ``n x n`` region is
    masked); ``INV`` holds the full ``BLOCK`` reciprocals for the pure scaling
    kernels below.  The dot operand is ``b * ONE`` with a *runtime scalar*
    (``ONE`` is passed as ``1.0``): a plain ``tl.trans(b)`` operand next to the
    ``tl.sum`` reduction miscompiles on the local sdnn-build pipeline, while an
    operand built from the reduction result (``b * inv[None, :]``) crashes
    flagtree's ``sdnn.core_specialize`` — the runtime scalar keeps the operand
    dynamic but load-derived, satisfying both.
    """
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)

    b = tl.load(BP + rows[:, None] * BLOCK + cols[None, :])
    a = tl.load(AP + rows[:, None] * BLOCK + cols[None, :])

    # Column norms of the raw rotated matrix are the singular values.
    s2 = tl.sum(b * b, axis=0)
    s = tl.sqrt(s2)
    inv = (1.0 / tl.maximum(s, 1.0e-20)) * (s > 1.0e-20)

    t = tl.dot(tl.trans(b * ONE), a, allow_tf32=False)
    tl.store(
        T + rows[:, None] * BLOCK + cols[None, :],
        t,
        mask=(rows[:, None] < n) & (cols[None, :] < n),
    )
    tl.store(INV + rows, inv)


@triton.jit
def _polar_bs_kernel(BP, INV, BS, BLOCK: tl.constexpr):
    """``Bs = B diag(1/S)`` — pure column scaling (no ``tl.dot``).

    ``INV`` is loaded as a ``(1, BLOCK)`` tile (a plain 1D load mixed with the
    2D ``BP`` load trips the addptr verifier on this backend).
    """
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    b = tl.load(BP + rows[:, None] * BLOCK + cols[None, :])
    inv = tl.load(INV + cols[None, :])
    tl.store(BS + rows[:, None] * BLOCK + cols[None, :], b * inv)


@triton.jit
def _polar_vh_kernel(T, INV, VH, BLOCK: tl.constexpr):
    """``Vh = diag(1/S^2) T`` — pure row scaling (no ``tl.dot``).

    ``INV`` is loaded as a ``(BLOCK, 1)`` tile; see ``_polar_bs_kernel``.
    """
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    t = tl.load(T + rows[:, None] * BLOCK + cols[None, :])
    inv = tl.load(INV + rows[:, None])
    tl.store(VH + rows[:, None] * BLOCK + cols[None, :], t * (inv * inv))


@triton.jit
def _polar_u_kernel(BS, VH, UOUT, UB, m, n, BLOCK: tl.constexpr):
    """``U = Bs Vh`` — one ``tl.dot`` on two plain loads.

    ``UB`` is ``U`` padded to ``BLOCK``; ``_polar_h_kernel`` consumes a torch
    copy of it.
    """
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    rmask = rows < m
    cmask = cols < n
    mask = rmask[:, None] & cmask[None, :]

    bs = tl.load(BS + rows[:, None] * BLOCK + cols[None, :])
    vh = tl.load(VH + rows[:, None] * BLOCK + cols[None, :])
    u = tl.dot(bs, vh, allow_tf32=False)
    tl.store(UOUT + rows[:, None] * n + cols[None, :], u, mask=mask)
    tl.store(UB + rows[:, None] * BLOCK + cols[None, :], u, mask=mask)


@triton.jit
def _polar_h_kernel(UB2, AP, HOUT, m, n, BLOCK: tl.constexpr):
    """``H = U^T A`` (polar: ``A = U H``, ``H`` symmetric PSD) — one ``tl.dot``."""
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    cmask = cols < n
    hmask = cmask[:, None] & cmask[None, :]

    u = tl.load(UB2 + rows[:, None] * BLOCK + cols[None, :])
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
        # ``B`` must be fully written before the copies below read it; the
        # zero-padded tiles are allocated after the sweeps complete so no
        # buffer is created while the osj kernel is still in flight.
        torch.cuda.synchronize()

        # Zero-padded (BLOCK x BLOCK) tiles for the polar post-processing
        # kernels: padding lanes must be zero in memory (masked loads may
        # ignore ``other`` on this backend), and the intermediate tiles stay
        # zero outside the masked store regions because the downstream kernels
        # read the full tile.
        BP = torch.zeros((batch, block, block), device=dev, dtype=A.dtype)
        AP = torch.zeros((batch, block, block), device=dev, dtype=A.dtype)
        T = torch.zeros((batch, block, block), device=dev, dtype=A.dtype)
        INV = torch.empty((batch, block), device=dev, dtype=A.dtype)
        BS = torch.zeros((batch, block, block), device=dev, dtype=A.dtype)
        VH = torch.zeros((batch, block, block), device=dev, dtype=A.dtype)
        UB = torch.zeros((batch, block, block), device=dev, dtype=A.dtype)
        # Torch copy of ``UB``: the only reliable cross-kernel barrier for the
        # dot-to-dot edge (``U`` -> ``H``) on this stack.
        UB2 = torch.zeros((batch, block, block), device=dev, dtype=A.dtype)
        if out_U is not None:
            U3 = out_U.reshape(batch, m, n)
            H3 = out_H.reshape(batch, n, n)
        else:
            U3 = torch.empty((batch, m, n), device=dev, dtype=A.dtype)
            H3 = torch.empty((batch, n, n), device=dev, dtype=A.dtype)

        for b in range(batch):
            BP[b, :MP, :NW].copy_(B[b].reshape(MP, NW))
            AP[b, :m, :n].copy_(A2[b])
            _polar_t_kernel[(1,)](
                BP[b],
                AP[b],
                T[b],
                INV[b],
                m,
                n,
                1.0,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )
            _polar_bs_kernel[(1,)](
                BP[b],
                INV[b],
                BS[b],
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )
            _polar_vh_kernel[(1,)](
                T[b],
                INV[b],
                VH[b],
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )
            _polar_u_kernel[(1,)](
                BS[b],
                VH[b],
                U3[b],
                UB[b],
                m,
                n,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )
            UB2[b].copy_(UB[b])
            _polar_h_kernel[(1,)](
                UB2[b],
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
