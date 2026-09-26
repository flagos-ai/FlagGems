import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _te_rmsnorm_bwd_dx_1d_kernel(
    dx_ptr,
    dz_ptr,
    x_ptr,
    gamma_ptr,
    rsigma_ptr,
    N,
    zero_centered_gamma: tl.constexpr,
    C: tl.constexpr,
    NEED_TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * N
    rsigma = tl.load(rsigma_ptr + pid).to(tl.float32)
    acc = tl.zeros([C], dtype=tl.float32)

    for off in range(0, N, C):
        cols = off + tl.arange(0, C)
        if NEED_TAIL:
            cmask = cols < N
            x = tl.load(x_ptr + row + cols, mask=cmask, other=0.0).to(tl.float32)
            dz = tl.load(dz_ptr + row + cols, mask=cmask, other=0.0).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
            if zero_centered_gamma:
                gamma = gamma + 1.0
            acc += x * rsigma * dz * gamma
        else:
            x = tl.load(x_ptr + row + cols).to(tl.float32)
            dz = tl.load(dz_ptr + row + cols).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols).to(tl.float32)
            if zero_centered_gamma:
                gamma = gamma + 1.0
            acc += x * rsigma * dz * gamma

    c1 = tl.sum(acc) / N

    for off in range(0, N, C):
        cols = off + tl.arange(0, C)
        if NEED_TAIL:
            cmask = cols < N
            x = tl.load(x_ptr + row + cols, mask=cmask, other=0.0).to(tl.float32)
            dz = tl.load(dz_ptr + row + cols, mask=cmask, other=0.0).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
            if zero_centered_gamma:
                gamma = gamma + 1.0
            x_hat = x * rsigma
            dx = rsigma * (dz * gamma - x_hat * c1)
            tl.store(dx_ptr + row + cols, dx, mask=cmask)
        else:
            x = tl.load(x_ptr + row + cols).to(tl.float32)
            dz = tl.load(dz_ptr + row + cols).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols).to(tl.float32)
            if zero_centered_gamma:
                gamma = gamma + 1.0
            x_hat = x * rsigma
            dx = rsigma * (dz * gamma - x_hat * c1)
            tl.store(dx_ptr + row + cols, dx)


@triton.jit
def _te_rmsnorm_bwd_dgamma_kernel(
    dgamma_partial_ptr,
    dz_ptr,
    x_ptr,
    rsigma_ptr,
    M,
    N,
    BM: tl.constexpr,
    C: tl.constexpr,
):
    n0 = tl.program_id(0) * C
    mi = tl.program_id(1)
    m0 = mi * BM
    cols = tl.arange(0, C)
    cmask = (n0 + cols) < N
    acc = tl.zeros([C], dtype=tl.float32)
    for r in range(0, BM):
        m = m0 + r
        base = m * N + n0
        x = tl.load(x_ptr + base + cols, mask=cmask, other=0.0).to(tl.float32)
        dz = tl.load(dz_ptr + base + cols, mask=cmask, other=0.0).to(tl.float32)
        rsigma = tl.load(rsigma_ptr + m).to(tl.float32)
        acc += tl.where(cmask, dz * x * rsigma, 0.0)
    tl.store(dgamma_partial_ptr + mi * N + n0 + cols, acc, mask=cmask)


@triton.jit
def _te_rmsnorm_bwd_dgamma_reduce_kernel(
    dgamma_ptr,
    dgamma_partial_ptr,
    P,
    N,
    C: tl.constexpr,
):
    n0 = tl.program_id(0) * C
    cols = n0 + tl.arange(0, C)
    cmask = cols < N
    acc = tl.zeros([C], dtype=tl.float32)
    for i in range(0, P):
        acc += tl.load(dgamma_partial_ptr + i * N + cols, mask=cmask, other=0.0).to(
            tl.float32
        )
    tl.store(dgamma_ptr + cols, acc, mask=cmask)


def _dgamma_bm_size(M):
    block = min(M, _DGM_BM_MAX)
    while block > 1 and M % block != 0:
        block //= 2
    return max(1, block)


_DGM_BM_MAX = 128


def _ln_bwd_col_size(N):
    cap = min(N, 8192)
    return 1 << (cap.bit_length() - 1)


def _dx_col_size(N):
    if N <= 16:
        return 16
    return min(max(1 << (N - 1).bit_length(), 1024), 8192)


_FWD_TILE_N_MAX = 4096
_FWD_TILE_ELEMS = 65536
_FWD_ROW_BLOCK = 64 * 128


@triton.jit
def _te_rmsnorm_fwd_tile2d_kernel(
    Y,
    INV_RMS,
    X,
    W,
    eps: tl.constexpr,
    TILE_M: tl.constexpr,
    N: tl.constexpr,
):
    pid = tl.program_id(0)

    n_off = tl.arange(0, N)
    w = tl.load(W + n_off).to(tl.float32)

    m_off = pid * TILE_M + tl.arange(0, TILE_M)
    offs = m_off[:, None] * N + n_off[None, :]

    x = tl.load(X + offs).to(tl.float32)

    var = tl.sum(x * x, axis=1) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    y = (x * rrms[:, None] * w[None, :]).to(Y.dtype.element_ty)
    tl.store(Y + offs, y)
    tl.store(INV_RMS + m_off, rrms)


@triton.jit
def _te_rmsnorm_fwd_row_kernel(
    Y,
    INV_RMS,
    X,
    W,
    N,
    eps,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    X += pid * N
    Y += pid * N

    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            x = tl.load(X + cols).to(tl.float32)
        sum_sq += x * x
    var = tl.sum(sum_sq, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    tl.store(INV_RMS + pid, rrms)

    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        if NEED_MASK:
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            y = (x * rrms * w).to(Y.dtype.element_ty)
            tl.store(Y + cols, y, mask=mask)
        else:
            x = tl.load(X + cols).to(tl.float32)
            w = tl.load(W + cols).to(tl.float32)
            y = (x * rrms * w).to(Y.dtype.element_ty)
            tl.store(Y + cols, y)


def _fwd_tile_m(N, M):
    """TILE_M for the unmasked 2D tile kernel, or None if not applicable.

    Mirrors rms_norm_forward's tile selection (TILE_M=32 preference; the
    [TILE_M, N] fp32 tile must stay within _FWD_TILE_ELEMS) but refuses tiles
    wider than _FWD_TILE_N_MAX columns: the [8, 8192] tile that the rms_norm
    65536-element budget admits when N == 8192 crashes the XPU vectorizer.
    """
    if N > _FWD_TILE_N_MAX:
        return None
    if N <= 256:
        for cand in (32, 16):
            if M % cand == 0:
                return cand
        return None
    tm = 32
    while tm * N > _FWD_TILE_ELEMS:
        tm //= 2
    while tm >= 2:
        if M % tm == 0:
            return tm
        tm //= 2
    return None


def te_rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    ln_out: torch.Tensor = None,
    quantizer=None,
    otype: torch.dtype = None,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    """TE-aligned RMSNorm forward (XPU backend local): y = x/sqrt(mean(x^2)+eps) * w,
    returns (y, None, rsigma).  See the module comment above for why the generic
    and rms_norm-reuse paths are not usable on this backend.
    """
    del sm_margin
    if zero_centered_gamma:
        weight = weight.to(torch.float32) + 1.0
    N = input.shape[-1]
    x = input.contiguous()
    M = x.numel() // N
    w = weight.contiguous()
    y = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=x.device)
    rsigma = torch.empty_strided((M,), (1,), dtype=torch.float32, device=x.device)

    with torch_device_fn.device(x.device):
        tile_m = _fwd_tile_m(N, M)
        if tile_m is not None:
            _te_rmsnorm_fwd_tile2d_kernel[(M // tile_m,)](
                y, rsigma, x, w, eps, tile_m, N
            )
        else:
            need_mask = (N % _FWD_ROW_BLOCK) != 0
            _te_rmsnorm_fwd_row_kernel[(M,)](
                y, rsigma, x, w, N, eps, _FWD_ROW_BLOCK, need_mask
            )

    if otype is not None and y.dtype != otype:
        y = y.to(otype)
    if ln_out is not None:
        ln_out.copy_(y)
        return ln_out, None, rsigma
    return y, None, rsigma


def te_rmsnorm_bwd(
    dz: torch.Tensor,
    x: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    del sm_margin
    original_shape = x.shape
    N = gamma.shape[0]
    x_2d = x.reshape(-1, N).contiguous()
    dz_2d = dz.reshape(-1, N).contiguous()
    rsigma = rsigma.contiguous()
    M = x_2d.shape[0]
    dx = torch.empty_like(x_2d)
    dgamma = torch.empty_like(gamma)

    bc = _ln_bwd_col_size(N)
    cdx = _dx_col_size(N)

    bm = _dgamma_bm_size(M)
    p = M // bm
    dgamma_partial = torch.empty((p, N), dtype=torch.float32, device=x.device)

    with torch_device_fn.device(x.device):
        _te_rmsnorm_bwd_dx_1d_kernel[(M,)](
            dx,
            dz_2d,
            x_2d,
            gamma,
            rsigma,
            N,
            zero_centered_gamma=zero_centered_gamma,
            C=cdx,
            NEED_TAIL=(N % cdx != 0),
            num_warps=4,
            isCloseUnrollControl=True,
            isCloseVectorization=False,
        )
        _te_rmsnorm_bwd_dgamma_kernel[(triton.cdiv(N, bc), p)](
            dgamma_partial,
            dz_2d,
            x_2d,
            rsigma,
            M,
            N,
            BM=bm,
            C=bc,
            num_warps=4,
            isCloseUnrollControl=True,
        )
        _te_rmsnorm_bwd_dgamma_reduce_kernel[(triton.cdiv(N, bc),)](
            dgamma,
            dgamma_partial,
            p,
            N,
            C=bc,
            num_warps=4,
            isCloseUnrollControl=True,
        )

    return dx.reshape(original_shape), dgamma


def _patch_generic_wrapper():
    """Route direct calls to the generic wrapper (flag_gems.ops.te_rmsnorm
    module) to this backend override.

    Tests and benchmarks import ``te_rmsnorm_bwd`` from
    ``flag_gems.ops.te_rmsnorm`` (bypassing the top-level ``flag_gems``
    registry that SpecOpRegistrar patches), so the generic Triton kernel
    would still be hit on XPU (it cannot compile there: the 2D
    ``tl.sum(..., axis=0)`` reduction in ``rmsnorm_bwd_dgamma_kernel`` is
    rejected by the XPU legalizer with
    ``axis must not be 0 for 2D+ shapes``).  Patching the module
    attribute at import time keeps the change backend-local: the generic
    module source is untouched and other vendor backends are unaffected
    (this module is only imported for the kunlunxin backend).
    """
    try:
        import sys

        _generic_module = sys.modules.get("flag_gems.ops.te_rmsnorm")
        if _generic_module is not None:
            if hasattr(_generic_module, "te_rmsnorm_bwd"):
                _generic_module.te_rmsnorm_bwd = te_rmsnorm_bwd
            if hasattr(_generic_module, "te_rmsnorm_fwd"):
                _generic_module.te_rmsnorm_fwd = te_rmsnorm_fwd
        import flag_gems.ops as _ops

        if hasattr(_ops, "te_rmsnorm_bwd"):
            _ops.te_rmsnorm_bwd = te_rmsnorm_bwd
        if hasattr(_ops, "te_rmsnorm_fwd"):
            _ops.te_rmsnorm_fwd = te_rmsnorm_fwd
    except ImportError:
        pass


_patch_generic_wrapper()


__all__ = ["te_rmsnorm_bwd", "te_rmsnorm_fwd"]
