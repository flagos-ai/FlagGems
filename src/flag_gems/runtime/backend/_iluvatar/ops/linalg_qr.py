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

import logging

import torch

from flag_gems.ops.linalg_qr import (
    _FUSED_DIM,
    _FUSED_ELEM,
    _FUSED_M,
    _FUSED_TALL_M,
    _PANEL_IB,
    _TSQR_ASPECT,
    _TSQR_MAX_N,
    _TSQR_MIN_M,
    _blocked_qr,
    _identity_kernel,
    _launch_larfb,
    _launch_larft,
)
from flag_gems.ops.linalg_qr import _linalg_qr as _generic_linalg_qr
from flag_gems.ops.linalg_qr import _triu_copy, _validate_mode, _validate_out

logger = logging.getLogger(__name__)


def _use_safe_q_assembly(A, mode):
    """True when the generic fp32 fused Q-assembly kernel would be used.

    The generic _assemble_q_fused_kernel (single launch: identity + reverse
    panel loop with cross-iteration global write->read of Q) miscompiles on
    this backend: the W1 = V_p^H Q accumulation produces NaN from the first
    panel whenever the panel is full, and Q comes out all-NaN.  Everything
    else in the generic implementation (fused single-kernel path, TSQR,
    single-panel assembly, all of geqrt/larft/larfb) is verified correct here,
    so only the blocked multi-panel fp32 Q assembly is rerouted.
    """
    if mode == "r" or A.dim() < 2 or A.element_size() != 4:
        return False
    m, n = A.shape[-2], A.shape[-1]
    if m == 0 or n == 0:
        return False
    k = min(m, n)
    if (k + _PANEL_IB - 1) // _PANEL_IB < 2:
        return False  # P == 1 uses the single-panel kernel, which is fine
    qcols = k if mode == "reduced" else m
    is_ts = (
        (m >= _TSQR_ASPECT * n)
        and (m >= _TSQR_MIN_M)
        and (n <= _TSQR_MAX_N)
        and mode == "reduced"
    )
    fits_fused = (
        m <= _FUSED_M
        and n <= _FUSED_DIM
        and m * n <= _FUSED_ELEM
        and qcols * m <= _FUSED_ELEM
    )
    if is_ts:
        fits_fused = fits_fused and (m <= _FUSED_TALL_M)
    return not fits_fused and not is_ts


def _assemble_q_safe(V, tau, Tbuf, m, n, k, qcols, ib, B, out):
    """Stream-ordered Q assembly: identity kernel + one larfb launch per panel
    (reverse order), so every cross-panel dependency is ordered by the stream
    instead of an intra-kernel global write->read."""
    P = (k + ib - 1) // ib
    kk_last = (P - 1) * ib
    ib_last = min(ib, k - kk_last)
    if kk_last + ib_last >= n:
        # the last panel never got a T from _blocked_qr (no trailing update)
        Vp = V[:, kk_last:m, kk_last : kk_last + ib_last]
        taup = tau[:, kk_last : kk_last + ib_last]
        Tp = Tbuf[:, kk_last : kk_last + ib_last, kk_last : kk_last + ib_last]
        _launch_larft(Vp, taup, Tp, m, kk_last, ib_last, B)
    sQb, sQm, sQn = out.stride()
    grid_e = (m * qcols + 1023) // 1024
    _identity_kernel[(B * grid_e,)](out, m, qcols, grid_e, sQb, sQm, sQn, BLOCK=1024)
    for kk in reversed(range(0, k, ib)):
        ib_active = min(ib, k - kk)
        Vp = V[:, kk:m, kk : kk + ib_active]
        Tp = Tbuf[:, kk : kk + ib_active, kk : kk + ib_active]
        _launch_larfb(Vp, Tp, out[:, kk:m, :], m - kk, qcols, ib_active, B, upper=True)


def _linalg_qr_blocked_safe(A, mode, out=None):
    """Generic blocked Householder path, but with _assemble_q_safe for Q."""
    batch_shape = A.shape[:-2]
    m, n = A.shape[-2], A.shape[-1]
    k = min(m, n)
    B = 1
    for d in batch_shape:
        B *= d
    W = A.reshape(B, m, n)
    qcols = k if mode == "reduced" else m
    out_Q = out_R = None
    if out is not None:
        out_Q, out_R = out
        out_Q = out_Q.reshape(B, m, qcols)
        out_R = out_R.reshape(B, qcols if mode == "complete" else k, n)

    V = torch.zeros(B, m, k, dtype=W.dtype, device=W.device)
    tau = torch.empty(B, k, dtype=W.dtype, device=W.device)
    Tbuf = torch.empty(B, k, k, dtype=W.dtype, device=W.device)
    W = W.clone()
    _blocked_qr(W, V, tau, Tbuf, m, n, k)

    Q = (
        out_Q
        if out_Q is not None
        else torch.empty(B, m, qcols, dtype=W.dtype, device=W.device)
    )
    _assemble_q_safe(V, tau, Tbuf, m, n, k, qcols, _PANEL_IB, B, Q)

    rrows = qcols if mode == "complete" else k
    R = (
        out_R
        if out_R is not None
        else torch.empty(B, rrows, n, dtype=W.dtype, device=W.device)
    )
    _triu_copy(W, R, R.shape[-2], n, B)

    if mode == "reduced":
        return Q.reshape(*batch_shape, m, k), R.reshape(*batch_shape, k, n)
    return Q.reshape(*batch_shape, m, m), R.reshape(*batch_shape, m, n)


def _linalg_qr_impl(A, mode, out=None):
    """Routing shared by linalg_qr / linalg_qr_out (no logging)."""
    # the safe-assembly route bypasses the generic entry; validate here so an
    # invalid mode raises instead of being silently treated as non-reduced
    _validate_mode(mode)
    if out is not None and A.dim() >= 2:
        _validate_out(out, A.dtype, A.shape[:-2], A.shape[-2], A.shape[-1], mode)
    if _use_safe_q_assembly(A, mode):
        return _linalg_qr_blocked_safe(A, mode, out=out)
    return _generic_linalg_qr(A, mode, out=out)


def linalg_qr(A, mode="reduced", *, out=None):
    logger.debug("GEMS_ILUVATAR LINALG_QR")
    return _linalg_qr_impl(A, mode, out=out)


def linalg_qr_out(A, mode="reduced", *, Q, R):
    logger.debug("GEMS_ILUVATAR LINALG_QR_OUT")
    return _linalg_qr_impl(A, mode, out=(Q, R))
