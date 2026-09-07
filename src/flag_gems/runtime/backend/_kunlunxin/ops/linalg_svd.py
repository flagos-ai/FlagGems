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

"""Kunlunxin backend override for ``linalg_svd``.

One-sided Jacobi (Hestenes) SVD on the XPU backend.

Pipeline
--------
A single ``_osj_pipeline`` kernel (grid ``(1,)``, one program per batch
element) performs the whole one-sided Jacobi process:

1. fill: copy ``A`` into a zero-padded ``(MP, NW)`` workspace ``B``
   (``MP = next_pow2(m)``, ``nw = n + (n & 1)``, ``NW = next_pow2(nw)``)
   with element-wise ``if``-guarded scalar stores.
2. ``sweeps`` x ``(nw-1)`` cyclic rotations on the columns of ``B``
   (``B -> B * J`` with ``J`` the plane rotation of the ``(p, q)`` columns),
   expressed as one flattened runtime loop of ``sweeps*(nw-1)*(nw//2)``
   steps.  After convergence the columns of ``B`` are orthogonal and
   ``B = U S``.
3. ``U = B * diag(1/S)`` with ``S = ||B[:, j]||``.
4. Host: ``Bc = B.cpu()``; the column norms ``S`` are computed on the host
   (``Bc.double().norm(dim=1)``); ``U`` is gathered on device and
   ``Vh = S^{-1} U^H A`` is formed on device (``U`` orthonormal, exact
   identity ``A = U S Vh``).

Why the implementation looks like this (Triton-XPU backend limitations
discovered while porting, see also the op black list):

- The Triton XPU compiler cannot legalize ``tt.reduce`` when a ``tl.sum``
  lives inside two nested *runtime* loops, so the pipeline keeps at most one
  runtime loop with a ``tl.sum`` at a time (the loops are sequential, never
  nested); the batch dimension is handled by launching one program per
  batch element (grid ``(1,)``).
- Multi-program grids that race a ``tl.sum`` (e.g. ``(batch, NW)``) produce
  racy stores even for a fixed input, so every kernel that reduces uses a
  single program.  Element-wise grids such as the original 3-D fill
  ``(batch, MP, NW)`` are legal but extremely slow on this backend
  (0.76 ms for a 64-element copy), which is why the fill lives inside the
  pipeline kernel instead.
- Vector ``tl.load(..., mask=..., other=...)`` is miscompiled on this
  backend; masks are expressed with element-wise ``if`` instead.
- Scalar stores of a warp-reduction result (``tl.store(S_ptr + j, s)`` after
  ``tl.sum``) are corrupted ~1/3 of the time on this backend with
  allocation-layout-dependent values (the same store sequence is correct on
  CUDA).  The ``S`` vector is therefore computed on the host from the
  Triton-produced ``B`` (``S = ||B[:, j]||`` is an O(mn) reduction, not an
  SVD), while ``U`` is still normalized on device (vector stores).
- An unmasked vector store whose address has the form ``base + p + rows*NW``
  (``rows = tl.arange(0, MP)``) writes phantom lanes past the tensor: the
  emitted store covers a fixed ~2 KB window at ``base`` regardless of the
  tensor's true extent, so the elements immediately *above* the tensor
  (e.g. a neighboring allocation) are silently overwritten with rotation
  data.  Every such store therefore carries an explicit
  ``mask = rows < MP``, which limits the store to the real ``MP`` rows
  (a masked store yields bit-identical in-bounds results, and zero extra
  stores; a row-padded ``(2*MP, NW)`` allocation does *not* contain the
  phantom writes, they escape past the padding).
- Kernel launches are asynchronous and launch-order completion is not
  guaranteed; the D2H ``B.cpu()`` copy doubles as the synchronization
  barrier for the pipeline launch.

dtype: float32 only (matching the generic linalg_svd contract).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _osj_pipeline(
    A_ptr, B_ptr, U_ptr, m, n, nw, total, MP: tl.constexpr, NW: tl.constexpr
):
    """Fill ``B``, run the one-sided Jacobi sweeps, and write ``U = B/S``."""
    rows = tl.arange(0, MP)
    ring = nw - 1
    half = nw // 2
    msk = rows < MP  # store mask: on this backend unmasked vector stores write
    # a fixed ~2KB window (the phantom lanes land in the memory block right
    # above the tensor); the mask limits the store to the real MP rows.

    # 1. fill: B[r, c] = A[r, c] if r < m and c < n else 0 (scalar stores)
    for r in range(0, MP):
        for c in range(0, NW):
            val = 0.0
            if (r < m) and (c < n):
                val = tl.load(A_ptr + r * n + c)
            tl.store(B_ptr + r * NW + c, val)

    # 2. cyclic one-sided Jacobi sweeps (flattened (sweep, s, j) schedule):
    #    rotate column pair (p, q) so that their inner product becomes zero.
    for t in range(0, total):
        s = (t // half) % ring
        j = t % half
        p = tl.where(j == 0, 0, (j + ring - s - 1) % ring + 1)
        q = (nw - 1 - j + ring - s - 1) % ring + 1
        ap = tl.load(B_ptr + p + rows * NW)
        aq = tl.load(B_ptr + q + rows * NW)
        alpha = tl.sum(ap * ap)
        beta = tl.sum(aq * aq)
        gamma = tl.sum(ap * aq)
        eps = 1.0e-20
        threshold = 1.0e-7 * tl.sqrt(alpha * beta + eps)
        active = tl.abs(gamma) > threshold
        safe_gamma = tl.where(active, gamma, 1.0)
        tau = (beta - alpha) / (2.0 * safe_gamma)
        sign_tau = tl.where(tau >= 0.0, 1.0, -1.0)
        t_rot = sign_tau / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
        c = tl.rsqrt(1.0 + t_rot * t_rot)
        s_rot = t_rot * c
        c = tl.where(active, c, 1.0)
        s_rot = tl.where(active, s_rot, 0.0)
        tl.store(B_ptr + p + rows * NW, c * ap - s_rot * aq, mask=msk)
        tl.store(B_ptr + q + rows * NW, s_rot * ap + c * aq, mask=msk)

    # 3. U = B * diag(1/S) with S = ||B[:, j]|| (vector stores only)
    for j in range(0, nw):
        v = tl.load(B_ptr + j + rows * NW)
        sv = tl.sqrt(tl.sum(v * v))
        inv = tl.where(sv > 1.0e-20, 1.0 / sv, 0.0)
        tl.store(U_ptr + j + rows * NW, v * inv, mask=msk)


def _osj_svd_impl(A, sweeps=12, full_matrices=False):
    """One-sided Jacobi SVD; returns ``(U, S, Vh)`` per torch convention."""
    dev = A.device
    if A.dim() == 2:
        A = A.unsqueeze(0)
    batch, m, n = A.shape
    nw = n if n % 2 == 0 else n + 1
    NW = nw if (nw & (nw - 1)) == 0 else triton.next_power_of_2(nw)
    MP = triton.next_power_of_2(m)

    # workspace (zero-padded) and U, both produced by the single pipeline
    # kernel (one launch per batch element, grid (1,)).
    B = torch.empty((batch, MP, NW), device=dev, dtype=A.dtype)
    U = torch.empty((batch, MP, NW), device=dev, dtype=A.dtype)
    total = sweeps * (nw - 1) * (nw // 2)
    for b in range(batch):
        _osj_pipeline[(1,)](
            A[b], B[b], U[b], m, n, nw, total, MP=MP, NW=NW,
            num_warps=1, num_stages=1,
        )

    # S = column norms of B, computed on host (scalar-store workaround);
    # the D2H copy is also the completion barrier for the pipeline kernel.
    Bc = B.cpu().double()
    S = Bc.norm(dim=1).to(device=dev, dtype=A.dtype)  # (batch, NW)

    k = min(m, n)
    S_sorted, idx = torch.sort(S, dim=-1, descending=True)
    S_sorted = S_sorted[:, :k]
    idxg = idx.unsqueeze(1).expand(-1, MP, -1)
    U = torch.gather(U, 2, idxg)[:, :m, :k].contiguous()

    # Vh = S^{-1} U^H A   (exact identity A = U diag(S) Vh when U orthonormal)
    UtA = torch.matmul(U.transpose(-2, -1), A)
    Vh = UtA * (1.0 / S_sorted).unsqueeze(-1)

    if full_matrices:
        if m > k:
            pad = torch.zeros((batch, m, m - k), device=dev, dtype=A.dtype)
            U = torch.cat([U, pad], dim=2)
        if n > k:
            pad = torch.zeros((batch, n - k, n), device=dev, dtype=A.dtype)
            Vh = torch.cat([Vh, pad], dim=1)
    Vh = Vh.contiguous()

    if batch == 1:
        return U[0], S_sorted[0], Vh[0]
    return U, S_sorted, Vh


def linalg_svd(A, full_matrices=True, *, driver=None):
    """Triton (XPU) implementation of ``torch.linalg.svd``.

    Returns ``(U, S, Vh)`` with ``A = U @ diag(S) @ Vh``.  Only ``float32``
    is supported (matches the generic ``linalg_svd`` contract).
    """
    logger.debug("GEMS LINALG_SVD (kunlunxin)")
    if A.dtype != torch.float32:
        raise TypeError(f"linalg_svd only supports float32 input, got {A.dtype}")
    return _osj_svd_impl(A, full_matrices=full_matrices)