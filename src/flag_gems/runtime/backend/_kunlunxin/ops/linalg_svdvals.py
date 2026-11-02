import logging

import numpy as np
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _osj_svals_pipeline(
    A_ptr, B_ptr, m, n, nw, total, MP: tl.constexpr, NW: tl.constexpr
):
    """Fill ``B`` and run the one-sided Jacobi sweeps (singular values only).

    Identical to the rotation part of ``linalg_svd._osj_pipeline`` (same cyclic
    schedule, same ``tl.sum``-based ``alpha/beta/gamma`` reductions, same
    ``mask=rows < MP`` store guard); the ``U = B * diag(1/S)`` normalization
    that ``linalg_svd`` needs is omitted because ``svdvals`` only requires
    ``S = ||B[:, j]||`` (computed on the host from ``B``).
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


# --- device-side descending bitonic sort -------------------------------------
# ``tl.sort``/``tl.flip``/``tl.split``/``tl.join`` and 2D/3D reductions are all
# unusable on this backend (GlobalEncodingAttr UNREACHABLE), so the sorted
# singular values are produced by a hand-rolled bitonic network. The stage
# kernel uses only shifted affine loads + min/max/where; the per-stage
# direction patterns are precomputed on the host.
#
# NOTE(kunlunxin): the stage kernel silently miscompiles for *tiles narrower
# than 64 lanes* (data-dependent wrong values; single-stage micro-tests fail
# for NW <= 32 and pass for every NW >= 64), so rows are padded up to
# ``_DEV_SORT_MIN_NW`` lanes; the padding zeros are negative-free, and
# ``S >= 0`` (column norms), so they always sort to the tail (unchanged from
# the previous host ``np.sort``, which sorted the same zero-padded row).
#
# NOTE(kunlunxin): a launch may also run at most ONE program and ONE stage --
# grid > 1 (>= ~9 programs) or a single program looping over rows/stages is
# silently corrupted as well, so the host launches one program per (stage, row).
_DEV_SORT_MIN_NW = 64


@triton.jit
def _svd_sort_stage_kernel(
    S_ptr,
    TAB_ptr,
    st: tl.int32,
    d: tl.constexpr,
    NW: tl.constexpr,
    PAD: tl.constexpr,
    TSTR: tl.constexpr,
):
    """One bitonic compare-exchange stage (descending). One program per launch."""
    pid = tl.program_id(0)
    offs = tl.arange(0, NW)
    base = S_ptr + pid * (NW + 2 * PAD) + PAD
    x = tl.load(base + offs)
    xf = tl.load(base + offs + d)
    xb = tl.load(base + offs - d)
    is_lo = tl.load(TAB_ptr + st * TSTR + offs) > 0.5
    v = tl.where(is_lo, xf, xb)
    mn = tl.minimum(x, v)
    mx = tl.maximum(x, v)
    cond = tl.load(TAB_ptr + st * TSTR + NW + offs) > 0.5
    tl.store(base + offs, tl.where(cond, mx, mn))


def _device_desc_sort(S, k, dev, dtype):
    """Sort ``S`` (values >= 0) descending on the device.

    Args:
        S: ``[batch, NW]`` CPU f64 (NW a power of two, trailing columns zero).
        k: number of leading (= largest) values to keep.
        dev: target device.
        dtype: output dtype.

    Returns:
        ``[batch, k]`` f32 device tensor = the k largest values of each row,
        in descending order.

    The stage kernel contains shifted-affine loads (``base + offs +/- d``); on
    this backend such kernels are silently corrupted when a launch runs more
    than one program or when one program executes more than one stage, so we
    launch exactly one program per (stage, row).  The per-stage ``is_lo``/``cond``
    direction patterns are precomputed on the host and transferred as one table.
    """
    batch, NW = S.shape
    NWs = max(NW, _DEV_SORT_MIN_NW)
    PAD = NWs // 2
    buf = torch.zeros(batch, NWs + 2 * PAD, device=dev, dtype=torch.float32)
    buf[:, PAD : PAD + NW] = S.to(device=dev, dtype=torch.float32)
    offs = np.arange(NWs)
    patterns = []
    for lev in range(1, NWs.bit_length()):
        stage = 1 << lev
        for s in range(lev):
            d = 1 << (lev - s - 1)
            is_lo = ((offs % (2 * d)) < d).astype(np.float32)
            block_odd = ((offs // stage) % 2 == 1).astype(np.float32)
            cond = (np.logical_xor(is_lo > 0.5, block_odd > 0.5)).astype(np.float32)
            patterns.append((d, stage, is_lo, cond))
    tab = torch.tensor(
        np.stack([np.concatenate([p[2], p[3]]) for p in patterns], axis=0),
        device=dev,
        dtype=torch.float32,
    )
    for st, (d, stage, _, _) in enumerate(patterns):
        for b in range(batch):
            _svd_sort_stage_kernel[(1,)](
                buf[b : b + 1],
                tab,
                st,
                d,
                NW=NWs,
                PAD=PAD,
                TSTR=2 * NWs,
                num_warps=4,
            )
    return buf[:, PAD : PAD + k].contiguous()


def _osj_svals_impl(A, sweeps=12):
    """One-sided Jacobi singular values only; returns ``S`` (descending)."""
    dev = A.device
    if A.dim() == 2:
        A = A.unsqueeze(0)
    batch, m, n = A.shape
    nw = n if n % 2 == 0 else n + 1
    NW = nw if (nw & (nw - 1)) == 0 else triton.next_power_of_2(nw)
    MP = triton.next_power_of_2(m)

    B = torch.empty((batch, MP, NW), device=dev, dtype=A.dtype)
    total = sweeps * (nw - 1) * (nw // 2)
    for b in range(batch):
        _osj_svals_pipeline[(1,)](
            A[b],
            B[b],
            m,
            n,
            nw,
            total,
            MP=MP,
            NW=NW,
            num_warps=1,
            num_stages=1,
        )

    S = B.cpu().double().norm(dim=1)  # [batch, NW] f64 CPU, values >= 0

    k = min(m, n)
    S_sorted = _device_desc_sort(S, k, dev, A.dtype)

    if batch == 1:
        return S_sorted[0]
    return S_sorted


def linalg_svdvals(A: torch.Tensor, driver: str = None) -> torch.Tensor:
    """Computes the singular values of a matrix (Kunlunxin XPU).

    Args:
        A: Input tensor of shape (*, m, n) where * is zero or more batch dimensions.
        driver: Accepted for API compatibility; the one-sided Jacobi pipeline
            does not use a solver driver selection.

    Returns:
        Singular values in descending order, shape (*, min(m, n)).
    """
    logger.debug("GEMS_KUNLUNXIN LINALG_SVDVALS")
    if A.dtype != torch.float32:
        raise TypeError(f"linalg_svdvals only supports float32 input, got {A.dtype}")
    if not A.is_contiguous():
        A = A.contiguous()

    if A.dim() not in (2, 3):
        orig_shape = A.shape
        m, n = orig_shape[-2:]
        k = min(m, n)
        A = A.reshape(-1, m, n)
        S = _osj_svals_impl(A)
        return S.reshape(*orig_shape[:-2], k)

    return _osj_svals_impl(A)
