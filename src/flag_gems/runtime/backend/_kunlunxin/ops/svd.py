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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from flag_gems.ops.svd import (  # noqa: F401  (generic fallback for other shapes)
    SVDResult,
    _is_float32_cuda_matrix,
    _is_low_precision_cuda_matrix,
    _svd_shape,
    svd as _generic_svd,
)

# The generic rank-1 path caps BLOCK_R at 1024 and sums the columns in a
# loop.  On XPU the loop-carried scalar accumulation (`norm_sq += tl.sum(...)`)
# is mis-compiled for BLOCK_R > 128, inflating the singular value.  Use a
# single, power-of-two block that covers the whole column/row so the reduction
# runs once, which the XPU backend lowers correctly.
_RANK1_XPU_BLOCK_LIMIT = 1024


@libentry()
@triton.jit
def _rank1_svd_kernel(
    A,
    U,
    S,
    V,
    M: tl.constexpr,
    N: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_R)
    eps = 1.1920928955078125e-7
    a_base = A + pid * M * N
    norm_sq = tl.full((), 0.0, dtype=tl.float32)

    if TALL:
        for base in range(0, M, BLOCK_R):
            rows = base + offsets
            mask = rows < M
            vals = tl.load(a_base + rows * N, mask=mask, other=0.0).to(tl.float32)
            norm_sq += tl.sum(vals * vals)

        norm = tl.sqrt(norm_sq)
        denom = tl.maximum(norm, eps)
        tl.store(S + pid, norm)
        tl.store(V + pid, 1.0)

        u_base = U + pid * M
        for base in range(0, M, BLOCK_R):
            rows = base + offsets
            mask = rows < M
            vals = tl.load(a_base + rows * N, mask=mask, other=0.0).to(tl.float32)
            tl.store(u_base + rows, vals / denom, mask=mask)
    else:
        for base in range(0, N, BLOCK_R):
            cols = base + offsets
            mask = cols < N
            vals = tl.load(a_base + cols, mask=mask, other=0.0).to(tl.float32)
            norm_sq += tl.sum(vals * vals)

        norm = tl.sqrt(norm_sq)
        denom = tl.maximum(norm, eps)
        tl.store(S + pid, norm)
        tl.store(U + pid, 1.0)

        v_base = V + pid * N
        for base in range(0, N, BLOCK_R):
            cols = base + offsets
            mask = cols < N
            vals = tl.load(a_base + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(v_base + cols, vals / denom, mask=mask)


def _rank1_svd_xpu(input):
    batch, m, n = _svd_shape(input)
    a = input.contiguous().reshape(batch, m, n)
    u = torch.empty((batch, m, 1), dtype=input.dtype, device=input.device)
    s = torch.empty((batch, 1), dtype=input.dtype, device=input.device)
    v = torch.empty((batch, n, 1), dtype=input.dtype, device=input.device)
    if batch != 0:
        rows = max(m, n)
        # Single power-of-two block covering the whole column/row: keeps the
        # reduction out of the loop, sidestepping the XPU loop-accumulation bug.
        block_r = triton.next_power_of_2(rows)
        with torch_device_fn.device(input.device):
            _rank1_svd_kernel[(batch,)](
                a,
                u,
                s,
                v,
                m,
                n,
                TALL=n == 1,
                BLOCK_R=block_r,
                num_warps=1 if block_r <= 64 else 4,
            )
    return (
        u.reshape(*input.shape[:-2], m, 1),
        s.reshape(*input.shape[:-2], 1),
        v.reshape(*input.shape[:-2], n, 1),
    )


@triton.jit
def _osj_pipeline(
    A_ptr,
    B_ptr,
    U_ptr,
    V_ptr,
    m,
    n,
    nw,
    total,
    MP: tl.constexpr,
    NW: tl.constexpr,
):
    """Fill ``B``, run the one-sided Jacobi sweeps, write ``U = B/S`` and
    accumulate the rotation matrix ``V`` (initially identity, right-multiplied
    by the same column rotations as ``B`` so ``A V = U S``)."""
    b = tl.program_id(0)
    A_ptr = A_ptr + b * m * n
    B_ptr = B_ptr + b * MP * NW
    U_ptr = U_ptr + b * MP * NW
    V_ptr = V_ptr + b * NW * NW

    rows = tl.arange(0, MP)
    ring = nw - 1
    half = nw // 2
    msk = rows < MP

    for r in range(0, MP):
        for c in range(0, NW):
            val = 0.0
            if (r < m) and (c < n):
                val = tl.load(A_ptr + r * n + c)
            tl.store(B_ptr + r * NW + c, val)

    # V = identity (NW x NW)
    for r in range(0, NW):
        for c in range(0, NW):
            val = 1.0 if (r == c) else 0.0
            tl.store(V_ptr + r * NW + c, val)

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
        # Same rotation on V's columns p,q (scalar loop: a second arange of a
        # different size is mis-lowered by the XPU backend when MP != NW).
        for r in range(0, NW):
            vp = tl.load(V_ptr + p + r * NW)
            vq = tl.load(V_ptr + q + r * NW)
            tl.store(V_ptr + p + r * NW, c * vp - s_rot * vq)
            tl.store(V_ptr + q + r * NW, s_rot * vp + c * vq)

    for j in range(0, nw):
        v = tl.load(B_ptr + j + rows * NW)
        sv = tl.sqrt(tl.sum(v * v))
        inv = tl.where(sv > 1.0e-20, 1.0 / sv, 0.0)
        tl.store(U_ptr + j + rows * NW, v * inv, mask=msk)


def _complete_u(u_thin, m, k):
    """Complete thin left factor ``u_thin [batch, m, k]`` to an orthonormal
    ``[batch, m, m]`` (first ``k`` columns kept) via modified Gram-Schmidt
    against identity columns ``e_k..e_{m-1}``.

    The orthogonalization is expressed with on-device element-wise ops and
    reductions (``*``/``.sum(-1)``/``.norm``) — the same primitives already used
    for the singular-value sort/norm post-processing — rather than a dedicated
    Triton kernel, because the XPU backend cannot legalize a ``tt.reduce`` that
    feeds a loop-carried column inside a nested runtime loop. The core Jacobi
    SVD itself stays in ``_osj_pipeline`` (a real Triton kernel)."""
    dev = u_thin.device
    dtype = u_thin.dtype
    squeezed = u_thin.dim() == 2
    if squeezed:
        u_thin = u_thin.unsqueeze(0)
    batch = u_thin.shape[0]
    eye = (
        torch.eye(m, m, dtype=dtype, device=dev)[:, k:]
        .unsqueeze(0)
        .expand(batch, m, m - k)
    )
    full = torch.cat([u_thin, eye], dim=-1).contiguous()  # [batch, m, m]
    for j in range(k, m):
        v = full[:, :, j]
        for p in range(j):
            prev = full[:, :, p]
            coeff = (prev * v).sum(dim=-1, keepdim=True)
            v = v - prev * coeff
        v = v / torch.clamp(v.norm(dim=-1, keepdim=True), min=1.0e-20)
        full[:, :, j] = v
    if squeezed:
        return full[0]
    return full


def _osj_svd_xpu(A, sweeps=12, full_matrices=False):
    """Runtime-loop one-sided Jacobi SVD, device-side only.

    The generic small/gram/cyclic Jacobi kernels fully unroll via
    ``tl.static_range`` and blow the LM stack ("Failed to tune buffer size").
    This kernel uses ``tl.range`` (a runtime loop) so it compiles, and the
    right singular vectors ``V`` are accumulated directly in the kernel (the
    product of orthogonal rotations) instead of reconstructed via ``U^T A / S``
    (which is ill-conditioned for near-zero singular values).

    Returns ``(U, S, V)`` with ``U [..., m, k]``, ``S [..., k]``, ``V [..., n, k]``
    for the thin SVD, or ``U [..., m, m]``, ``V [..., n, n]`` when
    ``full_matrices=True``, per ``torch.svd`` convention.
    """
    dev = A.device
    if A.dim() == 2:
        A = A.unsqueeze(0)
    batch, m, n = A.shape
    k = min(m, n)

    # Zero matrix: torch.svd returns U=I, S=0, V=I (any orthogonal basis works).
    if bool((A == 0).all()):
        eye_m = torch.eye(m, k, dtype=A.dtype, device=dev)
        eye_n = torch.eye(n, k, dtype=A.dtype, device=dev)
        u_full = eye_m.expand(batch, m, k).contiguous()
        s = torch.zeros(batch, k, dtype=A.dtype, device=dev)
        v_full = eye_n.expand(batch, n, k).contiguous()
        if full_matrices:
            if m > k:
                u_full = _complete_u(u_full, m, k)
            if n > k:
                v_full = _complete_u(v_full, n, k)
        if batch == 1:
            return u_full[0], s[0], v_full[0]
        return u_full, s, v_full

    nw = n if n % 2 == 0 else n + 1
    NW = nw if (nw & (nw - 1)) == 0 else triton.next_power_of_2(nw)
    MP = triton.next_power_of_2(m)

    B = torch.empty((batch, MP, NW), device=dev, dtype=A.dtype)
    U = torch.empty((batch, MP, NW), device=dev, dtype=A.dtype)
    V = torch.empty((batch, NW, NW), device=dev, dtype=A.dtype)
    total = sweeps * (nw - 1) * (nw // 2)
    A = A.contiguous()
    _osj_pipeline[(batch,)](
        A,
        B,
        U,
        V,
        m,
        n,
        nw,
        total,
        MP=MP,
        NW=NW,
        num_warps=1,
        num_stages=1,
    )

    # singular values = column norms of the swept B (device-side)
    S = B.norm(dim=1)

    # Sort only the n real columns. The dummy/padding columns also have S == 0
    # and would otherwise interleave with the real null-space columns after a
    # global sort, leaking zero vectors into the full (some=False) factors.
    S_sorted, idx = torch.sort(S[:, :n], dim=-1, descending=True)
    S_sorted = S_sorted[:, :k]

    idxg_u = idx.unsqueeze(1).expand(-1, MP, -1)
    U = torch.gather(U, 2, idxg_u)[:, :m, :k].contiguous()

    idxg_v = idx.unsqueeze(1).expand(-1, NW, -1)
    V = torch.gather(V, 2, idxg_v).contiguous()  # [batch, NW, n]

    if full_matrices:
        V_full = V[:, :n, :n].contiguous()
        U_full = _complete_u(U, m, k) if m > k else U
        if batch == 1:
            return U_full[0], S_sorted[0], V_full[0]
        return U_full, S_sorted, V_full

    V = V[:, :n, :k].contiguous()
    if batch == 1:
        return U[0], S_sorted[0], V[0]
    return U, S_sorted, V


def svd(input, some=True, compute_uv=True):
    # The generic Jacobi kernels (small/gram/cyclic) fully unroll via
    # tl.static_range and blow the LM stack ("Failed to tune buffer size").
    # Route the thin SVD (some=True, f32, k>=3) through a runtime-loop one-sided
    # Jacobi instead.  Large rank-1 stays on the dedicated single-block kernel.
    if _is_low_precision_cuda_matrix(input) and some and compute_uv:
        _, m, n = _svd_shape(input)
        if min(m, n) >= 3:
            u, s, v = _osj_svd_xpu(input.to(torch.float32))
            return SVDResult(u.to(input.dtype), s.to(input.dtype), v.to(input.dtype))
    if _is_float32_cuda_matrix(input) and not some and compute_uv:
        _, m, n = _svd_shape(input)
        if min(m, n) >= 3 and max(m, n) <= 64:
            return SVDResult(*_osj_svd_xpu(input, full_matrices=True))
    if _is_float32_cuda_matrix(input) and some and compute_uv:
        _, m, n = _svd_shape(input)
        k = min(m, n)
        if k == 1 and max(m, n) > _RANK1_XPU_BLOCK_LIMIT:
            return SVDResult(*_rank1_svd_xpu(input))
        if k >= 3:
            return SVDResult(*_osj_svd_xpu(input))
    return _generic_svd(input, some, compute_uv)
