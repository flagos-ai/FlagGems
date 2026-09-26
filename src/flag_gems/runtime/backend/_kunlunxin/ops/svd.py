import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from .bmm import bmm

logger = logging.getLogger(__name__)

SVDResult = namedtuple("SVDResult", ["U", "S", "V"])

_EPS = 1.1920928955078125e-7


@triton.jit
def _osj_svd_pipeline(
    A_ptr, B_ptr, U_ptr, S_ptr, m, n, nw, total, MP: tl.constexpr, NW: tl.constexpr
):
    """One-sided Jacobi SVD on a tall matrix (m >= n).

    Fills ``B`` with the (zero padded) input, runs cyclic Jacobi sweeps to make
    the columns mutually orthogonal, then writes normalized columns to ``U`` and
    the column norms (singular values) to ``S``.  Everything runs in fp32.
    """
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

    for j in range(0, nw):
        v = tl.load(B_ptr + j + rows * NW)
        sv = tl.sqrt(tl.sum(v * v))
        inv = tl.where(sv > 1.0e-20, 1.0 / sv, 0.0)
        tl.store(U_ptr + j + rows * NW, v * inv, mask=msk)
        tl.store(S_ptr + j, sv)


@triton.jit
def _rank1_svd_kernel(
    A_ptr, U_ptr, S_ptr, V_ptr, m, n, TALL: tl.constexpr, BLOCK: tl.constexpr
):
    """Rank-1 SVD for min(m, n) == 1 matching torch.svd's degenerate convention.

    TALL (n == 1, column vector): U = A / ||A||, S = ||A||, V = [[1]].
    Otherwise (m == 1, row vector): V = A / ||A||, S = ||A||, U = [[1]].
    For the all-zero input this yields the zero long-side factor and a unit
    scalar on the short side, exactly like ``torch.svd``.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    a_base = A_ptr + pid * m * n
    eps = 1.1920928955078125e-7
    if TALL:
        mask = offs < m
        vals = tl.load(a_base + offs * n, mask=mask, other=0.0)
        norm = tl.sqrt(tl.sum(vals * vals))
        denom = tl.maximum(norm, eps)
        tl.store(S_ptr + pid, norm)
        tl.store(V_ptr + pid, 1.0)
        tl.store(U_ptr + pid * m + offs, vals / denom, mask=mask)
    else:
        mask = offs < n
        vals = tl.load(a_base + offs, mask=mask, other=0.0)
        norm = tl.sqrt(tl.sum(vals * vals))
        denom = tl.maximum(norm, eps)
        tl.store(S_ptr + pid, norm)
        tl.store(U_ptr + pid, 1.0)
        tl.store(V_ptr + pid * n + offs, vals / denom, mask=mask)


def _rank1_svd(a, batch, m, n):
    """a: (batch, m, n) fp32 contiguous with min(m, n) == 1.

    Returns thin U (batch, m, 1), S (batch, 1), Vh (batch, 1, n).
    """
    dev = a.device
    u = torch.empty((batch, m, 1), device=dev, dtype=torch.float32)
    s = torch.empty((batch, 1), device=dev, dtype=torch.float32)
    v = torch.empty((batch, n, 1), device=dev, dtype=torch.float32)
    block = triton.next_power_of_2(max(m, n))
    for b in range(batch):
        _rank1_svd_kernel[(1,)](
            a[b], u[b], s[b], v[b], m, n, TALL=(n == 1), BLOCK=block, num_warps=1
        )
    vh = v.transpose(-2, -1).contiguous()
    return u, s, vh


_OSJ_MAX_MP = 512


def _gram_thin_factors(w, batch, m, n, want_uv, sweeps):
    """Thin SVD of a tall (batch, m, n) matrix via its small n x n Gram matrix.

    Used when ``m`` is too large for the row-parallel Jacobi pipeline.  The Gram
    matrix ``G = w^T w`` is (batch, n, n); its (symmetric PSD) SVD gives the
    right singular vectors ``V`` and squared singular values.  ``U = w V / S``.
    Returns ``(U, S, Vh)`` in the tall orientation (U (batch, m, n), S (batch,
    n), Vh (batch, n, n)); ``U``/``Vh`` are ``None`` when ``want_uv`` is False.
    """
    dev = w.device
    G = bmm(w.transpose(-2, -1).contiguous(), w)
    nw = n if n % 2 == 0 else n + 1
    NW = nw if (nw & (nw - 1)) == 0 else triton.next_power_of_2(nw)
    MPg = triton.next_power_of_2(n)
    Bg = torch.empty((batch, MPg, NW), device=dev, dtype=torch.float32)
    Ug = torch.zeros((batch, MPg, NW), device=dev, dtype=torch.float32)
    Sg = torch.zeros((batch, NW), device=dev, dtype=torch.float32)
    total = sweeps * (nw - 1) * (nw // 2)
    for b in range(batch):
        _osj_svd_pipeline[(1,)](
            G[b],
            Bg[b],
            Ug[b],
            Sg[b],
            n,
            n,
            nw,
            total,
            MP=MPg,
            NW=NW,
            num_warps=1,
            num_stages=1,
        )
    Sg_sorted, idx = torch.sort(Sg, dim=-1, descending=True)
    Sg_sorted = Sg_sorted[:, :n].contiguous()
    S = torch.sqrt(torch.clamp(Sg_sorted, min=0.0))
    if not want_uv:
        return None, S, None
    idxg = idx.unsqueeze(1).expand(-1, MPg, -1)
    V = torch.gather(Ug, 2, idxg)[:, :n, :n].contiguous()
    inv_s = torch.where(S > 1.0e-20, 1.0 / S, torch.zeros_like(S))
    U = bmm(w, V) * inv_s.unsqueeze(1)
    Vh = V.transpose(-2, -1).contiguous()
    return U, S, Vh


def _osj_thin(a, batch, m0, n0, want_uv=True, sweeps=12):
    """One-sided Jacobi thin SVD of a batch of fp32 matrices.

    ``a`` is (batch, m0, n0) contiguous.  Returns thin factors in the original
    orientation: U (batch, m0, k), S (batch, k), Vh (batch, k, n0) with
    ``a = U @ diag(S) @ Vh``.  When ``want_uv`` is False only S is meaningful
    (U/Vh are returned as ``None``).
    """
    dev = a.device
    k = min(m0, n0)
    transposed = n0 > m0
    w = a.transpose(-2, -1).contiguous() if transposed else a
    m, n = (n0, m0) if transposed else (m0, n0)

    if triton.next_power_of_2(m) > _OSJ_MAX_MP:
        Uw, S_sorted, Vhw = _gram_thin_factors(w, batch, m, n, want_uv, sweeps)
        if not want_uv:
            return None, S_sorted, None
        if transposed:
            U_a = Vhw.transpose(-2, -1).contiguous()
            Vh_a = Uw.transpose(-2, -1).contiguous()
        else:
            U_a = Uw
            Vh_a = Vhw
        return U_a, S_sorted, Vh_a

    nw = n if n % 2 == 0 else n + 1
    NW = nw if (nw & (nw - 1)) == 0 else triton.next_power_of_2(nw)
    MP = triton.next_power_of_2(m)

    B = torch.empty((batch, MP, NW), device=dev, dtype=torch.float32)
    U = torch.zeros((batch, MP, NW), device=dev, dtype=torch.float32)
    Sbuf = torch.zeros((batch, NW), device=dev, dtype=torch.float32)
    total = sweeps * (nw - 1) * (nw // 2)
    for b in range(batch):
        _osj_svd_pipeline[(1,)](
            w[b],
            B[b],
            U[b],
            Sbuf[b],
            m,
            n,
            nw,
            total,
            MP=MP,
            NW=NW,
            num_warps=1,
            num_stages=1,
        )

    S_sorted, idx = torch.sort(Sbuf, dim=-1, descending=True)
    S_sorted = S_sorted[:, :k].contiguous()

    if not want_uv:
        return None, S_sorted, None

    idxg = idx.unsqueeze(1).expand(-1, MP, -1)
    Uw = torch.gather(U, 2, idxg)[:, :m, :k].contiguous()
    inv_s = torch.where(S_sorted > 1.0e-20, 1.0 / S_sorted, torch.zeros_like(S_sorted))
    Vhw = bmm(Uw.transpose(-2, -1), w) * inv_s.unsqueeze(-1)

    if transposed:
        U_a = Vhw.transpose(-2, -1).contiguous()
        Vh_a = Uw.transpose(-2, -1).contiguous()
    else:
        U_a = Uw
        Vh_a = Vhw
    return U_a, S_sorted, Vh_a


def _ortho_complete(Q, out_cols, tol=1.0e-6):
    """Return (batch, dim, out_cols) with orthonormal columns.

    Columns of ``Q`` (batch, dim, k) seed the basis; degenerate / missing
    columns are completed with orthonormalized standard basis vectors so the
    result always has ``out_cols`` orthonormal columns (matching torch.svd's
    identity-fill convention for zero singular values).
    """
    batch, dim, k = Q.shape
    dev = Q.device
    eye = torch.eye(dim, device=dev, dtype=Q.dtype).unsqueeze(0).expand(batch, dim, dim)
    cand = torch.cat([Q, eye], dim=2)
    ncand = k + dim
    out = torch.zeros(batch, dim, out_cols, device=dev, dtype=Q.dtype)
    filled = [0] * batch
    for c in range(ncand):
        v = cand[:, :, c].clone()
        for j in range(out_cols):
            qj = out[:, :, j]
            coeff = (qj * v).sum(dim=1, keepdim=True)
            v = v - coeff * qj
        norm = torch.sqrt((v * v).sum(dim=1))
        vn = v / torch.clamp(norm, min=tol)
        for b in range(batch):
            if filled[b] < out_cols and norm[b].item() > tol:
                out[b, :, filled[b]] = vn[b]
                filled[b] += 1
    return out


def _batch_dims(input):
    m = input.shape[-2]
    n = input.shape[-1]
    batch = 1
    for d in input.shape[:-2]:
        batch *= d
    return batch, m, n


def _empty_result(input, some, compute_uv):
    _, m, n = _batch_dims(input)
    k = min(m, n)
    thin = compute_uv and some
    u_cols = k if thin else m
    v_cols = k if thin else n
    lead = input.shape[:-2]
    u = torch.empty((*lead, m, u_cols), dtype=input.dtype, device=input.device)
    s = torch.empty((*lead, k), dtype=torch.float32, device=input.device).to(
        input.real.dtype if input.is_complex() else input.dtype
    )
    v = torch.empty((*lead, n, v_cols), dtype=input.dtype, device=input.device)
    return u, s, v


def _real_svd(input, some, compute_uv):
    lead = input.shape[:-2]
    batch, m, n = _batch_dims(input)
    k = min(m, n)
    a = input.contiguous().reshape(batch, m, n).to(torch.float32)

    if not compute_uv:
        _, s, _ = _osj_thin(a, batch, m, n, want_uv=False)
        u = torch.zeros((batch, m, m), device=a.device, dtype=torch.float32)
        v = torch.zeros((batch, n, n), device=a.device, dtype=torch.float32)
        return (
            u.reshape(*lead, m, m),
            s.reshape(*lead, k),
            v.reshape(*lead, n, n),
        )

    if k == 1:
        u, s, vh = _rank1_svd(a, batch, m, n)
    else:
        u, s, vh = _osj_thin(a, batch, m, n, want_uv=True)
        v_thin = vh.transpose(-2, -1).contiguous()
        u = _ortho_complete(u, k)
        v_thin = _ortho_complete(v_thin, k)
        vh = v_thin.transpose(-2, -1).contiguous()

    v = vh.transpose(-2, -1).contiguous()
    if not some:
        u = _ortho_complete(u, m)
        v = _ortho_complete(v, n)

    return (
        u.reshape(*lead, m, u.shape[-1]),
        s.reshape(*lead, k),
        v.reshape(*lead, n, v.shape[-1]),
    )


def _complex_svd(input, some, compute_uv):
    lead = input.shape[:-2]
    batch, m, n = _batch_dims(input)
    k = min(m, n)
    a = input.contiguous().reshape(batch, m, n)
    ar = a.real.contiguous().to(torch.float32)
    ai = a.imag.contiguous().to(torch.float32)

    top = torch.cat([ar, -ai], dim=-1)
    bot = torch.cat([ai, ar], dim=-1)
    R = torch.cat([top, bot], dim=-2)

    _, s_r, vh_r = _osj_thin(R, batch, 2 * m, 2 * n, want_uv=True)
    s = s_r[:, 0::2][:, :k].contiguous()

    v_full = vh_r.transpose(-2, -1)
    v_cols = v_full[:, :, 0::2][:, :, :k].contiguous()
    vcr = v_cols[:, :n, :].contiguous()
    vci = v_cols[:, n : 2 * n, :].contiguous()

    if not compute_uv:
        u = torch.empty((*lead, m, m), dtype=input.dtype, device=input.device)
        v = torch.empty((*lead, n, n), dtype=input.dtype, device=input.device)
        return u, s.reshape(*lead, k), v

    ur = bmm(ar, vcr) - bmm(ai, vci)
    ui = bmm(ar, vci) + bmm(ai, vcr)
    inv_s = torch.where(s > 1.0e-20, 1.0 / s, torch.zeros_like(s)).unsqueeze(-2)
    ur = ur * inv_s
    ui = ui * inv_s

    u = torch.complex(ur, ui).to(input.dtype)
    v = torch.complex(vcr, vci).to(input.dtype)
    return (
        u.reshape(*lead, m, k),
        s.reshape(*lead, k),
        v.reshape(*lead, n, k),
    )


def svd(input, some=True, compute_uv=True):
    logger.debug("GEMS_KUNLUNXIN SVD")
    if 0 in input.shape[-2:]:
        return SVDResult(*_empty_result(input, some, compute_uv))
    if input.is_complex():
        return SVDResult(*_complex_svd(input, some, compute_uv))
    if input.dtype in (torch.float16, torch.bfloat16):
        u, s, v = _real_svd(input.to(torch.float32), some, compute_uv)
        return SVDResult(u.to(input.dtype), s.to(input.dtype), v.to(input.dtype))
    return SVDResult(*_real_svd(input, some, compute_uv))
