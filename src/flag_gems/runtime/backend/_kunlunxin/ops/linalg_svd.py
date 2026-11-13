import logging
import os

import torch
import triton
import triton.language as tl

from .bmm import bmm

logger = logging.getLogger(__name__)

try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))

# Raw cluster payload (`sj_svd.xpu`): the whole one-sided Jacobi SVD -- fill,
# sweeps with all `nw/2` disjoint rotations of a round rotated concurrently by
# one core each, singular values, descending rank permutation and U -- inside a
# single program. It exists because the equivalent triton tile formulation is
# rejected by the XPU backend (see solution/linalg_svd/report.md appendices B
# and C), and it also drops the host round trip (fp64 norm) plus torch.sort and
# torch.gather from the tail.
_SJ_MAX = 64


def _payload_ok(mp, nw):
    return (
        _TLE_OK
        and (os.environ.get("TRITON_XPU_ARCH", "3") == "3")
        and mp <= _SJ_MAX
        and nw <= _SJ_MAX
    )


if _TLE_OK:

    @tle.raw.dialect(
        "xpu3",
        object=os.path.join(os.path.dirname(_HERE), "payload", "obj", "sj_svd.o"),
        arch=3,
    )
    def sj_svd(A, U, S, Vh, m, n, mp, nw, sweeps, bid): ...

    @triton.jit(do_not_specialize=["m", "n", "sweeps"])
    def _sj_svd_kernel(A, U, S, Vh, m, n, mp, nw, sweeps):
        tle.raw.call(sj_svd, (A, U, S, Vh, m, n, mp, nw, sweeps, tl.program_id(0)))


@triton.jit
def _osj_pipeline(
    A_ptr, B_ptr, U_ptr, m, n, nw, total, MP: tl.constexpr, NW: tl.constexpr
):
    """Fill ``B``, run the one-sided Jacobi sweeps, and write ``U = B/S``."""
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


def _osj_svd_impl(A, sweeps=12, full_matrices=False):
    """One-sided Jacobi SVD; returns ``(U, S, Vh)`` per torch convention."""
    dev = A.device
    if A.dim() == 2:
        A = A.unsqueeze(0)
    batch, m, n = A.shape
    nw = n if n % 2 == 0 else n + 1
    NW = nw if (nw & (nw - 1)) == 0 else triton.next_power_of_2(nw)
    MP = triton.next_power_of_2(m)

    B = torch.empty((batch, MP, NW), device=dev, dtype=A.dtype)
    U = torch.empty((batch, MP, NW), device=dev, dtype=A.dtype)
    total = sweeps * (nw - 1) * (nw // 2)
    for b in range(batch):
        _osj_pipeline[(1,)](
            A[b],
            B[b],
            U[b],
            m,
            n,
            nw,
            total,
            MP=MP,
            NW=NW,
            num_warps=1,
            num_stages=1,
        )

    Bc = B.cpu().double()
    S = Bc.norm(dim=1).to(device=dev, dtype=A.dtype)

    k = min(m, n)
    S_sorted, idx = torch.sort(S, dim=-1, descending=True)
    S_sorted = S_sorted[:, :k]
    idxg = idx.unsqueeze(1).expand(-1, MP, -1)
    U = torch.gather(U, 2, idxg)[:, :m, :k].contiguous()

    UtA = bmm(U.transpose(-2, -1), A)
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


def _osj_svd_payload(A, sweeps=10, full_matrices=False):
    """Payload path: one cluster payload produces ordered ``U``, ``S`` and
    ``Vh`` -- no host round trip, no ``torch.sort`` / ``torch.gather`` / ``bmm``.
    The host only slices to ``k`` and pads for ``full_matrices``.

    ``sweeps=10`` is the measured convergence floor: singular *values* saturate
    by 8 sweeps, but the singular *vectors* of an ill-conditioned spectrum need
    10 (``test_linalg_svd_orthonormal`` fails at 8 with gram error ~30, passes
    at 10 with ~6e-5). 10 keeps the full test margin while cutting the sweep
    loop -- and thus the (nw-1)*sweeps cluster barriers -- by 1/6 versus the 12
    the pointer path used.
    """
    dev = A.device
    if A.dim() == 2:
        A = A.unsqueeze(0)
    batch, m, n = A.shape
    MP = triton.next_power_of_2(max(m, 2))
    NW = triton.next_power_of_2(max(n, 2))
    k = min(m, n)

    U = torch.empty((batch, MP, NW), device=dev, dtype=A.dtype)
    S_sorted = torch.empty((batch, NW), device=dev, dtype=A.dtype)
    Vh = torch.empty((batch, NW, NW), device=dev, dtype=A.dtype)
    for b in range(batch):
        _sj_svd_kernel[(1,)](A[b], U[b], S_sorted[b], Vh[b], m, n, MP, NW, sweeps)

    S_sorted = S_sorted[:, :k]
    U = U[:, :m, :k].contiguous()
    Vh = Vh[:, :k, :n]

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
    logger.debug("GEMS_KUNLUNXIN LINALG_SVD")
    if A.dtype != torch.float32:
        raise TypeError(f"linalg_svd only supports float32 input, got {A.dtype}")
    m = A.shape[-2] if A.dim() >= 2 else 1
    n = A.shape[-1] if A.dim() >= 1 else 1
    mp = triton.next_power_of_2(max(m, 2))
    nw = triton.next_power_of_2(max(n, 2))
    if _payload_ok(mp, nw):
        return _osj_svd_payload(A, full_matrices=full_matrices)
    return _osj_svd_impl(A, full_matrices=full_matrices)
