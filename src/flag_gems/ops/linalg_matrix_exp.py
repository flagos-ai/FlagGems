import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.exp import exp
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_THETA_18 = {
    torch.float32: 3.010066362817634e00,
    torch.float64: 1.090863719290036e00,
}

_T18_B = [
    [
        0.0,
        -1.00365581030144618291e-01,
        -8.02924648241156932449e-03,
        -8.92138498045729985177e-04,
        0.0,
    ],
    [
        0.0,
        3.97849749499645077844e-01,
        1.36783778460411720168e00,
        4.98289622525382669416e-01,
        -6.37898194594723280150e-04,
    ],
    [
        -1.09676396052962061844e01,
        1.68015813878906206114e00,
        5.71779846478865511061e-02,
        -6.98210122488052056106e-03,
        3.34975017086070470649e-05,
    ],
    [
        -9.04316832390810593223e-02,
        -6.76404519071381882256e-02,
        6.75961301770459654925e-02,
        2.95552570429315521194e-02,
        -1.39180257516060693404e-05,
    ],
    [
        0.0,
        0.0,
        -9.23364619367118555360e-02,
        -1.69364939002081722752e-02,
        -1.40086798182036094347e-05,
    ],
]

_T18_COEFF_CACHE = {}

_MATRIX_EXP_SMALL_N_MAX = 16


def _get_t18_coeff(dtype, device):
    key = (dtype, str(device))
    coeff = _T18_COEFF_CACHE.get(key)
    if coeff is None:
        coeff = torch.tensor(_T18_B, dtype=dtype).to(device)
        _T18_COEFF_CACHE[key] = coeff
    return coeff


@triton.jit
def _t18_lincomb_row(COEFF, k, p0, p1, p2, p3, p4):
    c0 = tl.load(COEFF + k * 5 + 0)
    c1 = tl.load(COEFF + k * 5 + 1)
    c2 = tl.load(COEFF + k * 5 + 2)
    c3 = tl.load(COEFF + k * 5 + 3)
    c4 = tl.load(COEFF + k * 5 + 4)
    return c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3 + c4 * p4


@libentry()
@triton.jit
def _matrix_exp_small_kernel(
    A,
    OUT,
    COEFF,
    N,
    THETA,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    offs = pid * N * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < N) & (cols[None, :] < N)
    a = tl.load(A + offs, mask=mask, other=0.0)

    norm = tl.max(tl.sum(tl.abs(a), axis=0), axis=0)
    s = tl.maximum(tl.ceil(tl.log2(norm / THETA)), 0.0)
    s = tl.minimum(s, 4096.0)
    s = tl.where(norm != norm, 0.0, s)
    s_i = s.to(tl.int32)
    scale = tl.exp2(-s)

    a_s = a * scale
    a2 = tl.dot(a_s, a_s, input_precision="ieee", out_dtype=A.dtype.element_ty)
    a3 = tl.dot(a2, a_s, input_precision="ieee", out_dtype=A.dtype.element_ty)
    a6 = tl.dot(a3, a3, input_precision="ieee", out_dtype=A.dtype.element_ty)
    p0 = (rows[:, None] == cols[None, :]).to(A.dtype.element_ty)

    b0 = _t18_lincomb_row(COEFF, 0, p0, a_s, a2, a3, a6)
    b1 = _t18_lincomb_row(COEFF, 1, p0, a_s, a2, a3, a6)
    b2 = _t18_lincomb_row(COEFF, 2, p0, a_s, a2, a3, a6)
    b3 = _t18_lincomb_row(COEFF, 3, p0, a_s, a2, a3, a6)
    b4 = _t18_lincomb_row(COEFF, 4, p0, a_s, a2, a3, a6)

    b3 += tl.dot(b0, b4, input_precision="ieee", out_dtype=A.dtype.element_ty)
    r = b1 + tl.dot(b2 + b3, b3, input_precision="ieee", out_dtype=A.dtype.element_ty)

    for _ in range(s_i):
        r = tl.dot(r, r, input_precision="ieee", out_dtype=A.dtype.element_ty)

    tl.store(OUT + offs, r, mask=mask)


@libentry()
@triton.jit
def _matrix_exp_norm_kernel(
    A,
    S,
    N,
    THETA,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    base = pid * N * N

    norm = tl.zeros((), dtype=A.dtype.element_ty)
    for j0 in range(0, N, BLOCK_N):
        cols = j0 + tl.arange(0, BLOCK_N)
        cmask = cols < N
        col_acc = tl.zeros([BLOCK_N], dtype=A.dtype.element_ty)
        for i0 in range(0, N, BLOCK_M):
            rows = i0 + tl.arange(0, BLOCK_M)
            mask = (rows[:, None] < N) & cmask[None, :]
            tile = tl.load(
                A + base + rows[:, None] * N + cols[None, :], mask=mask, other=0.0
            )
            col_acc += tl.sum(tl.abs(tile), axis=0)
        norm = tl.maximum(norm, tl.max(col_acc, axis=0))

    s = tl.maximum(tl.ceil(tl.log2(norm / THETA)), 0.0)
    s = tl.minimum(s, 4096.0)
    s = tl.where(norm != norm, 0.0, s)
    tl.store(S + pid, s.to(tl.int32))


@libentry()
@triton.jit
def _matrix_exp_bmm_kernel(
    A,
    B,
    C_IN,
    C_OUT,
    S,
    N,
    SCALE_A: tl.constexpr,
    SCALE_B: tl.constexpr,
    HAS_ACC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_mn = tl.cdiv(N, BLOCK_M) * num_pid_n
    pid_b = pid // num_pid_mn
    rem = pid % num_pid_mn
    pid_m = rem // num_pid_n
    pid_n = rem % num_pid_n
    base = pid_b * N * N

    scale = tl.full((), 1.0, dtype=A.dtype.element_ty)
    if SCALE_A or SCALE_B:
        s_i = tl.load(S + pid_b)
        scale = tl.exp2(-s_i.to(A.dtype.element_ty))

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (rm[:, None] < N) & (rn[None, :] < N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=A.dtype.element_ty)
    for k0 in range(0, N, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + base + rm[:, None] * N + rk[None, :],
            mask=(rm[:, None] < N) & (rk[None, :] < N),
            other=0.0,
        )
        b = tl.load(
            B + base + rk[:, None] * N + rn[None, :],
            mask=(rk[:, None] < N) & (rn[None, :] < N),
            other=0.0,
        )
        if SCALE_A:
            a = a * scale
        if SCALE_B:
            b = b * scale

        acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=A.dtype.element_ty)

    if HAS_ACC:
        acc += tl.load(
            C_IN + base + rm[:, None] * N + rn[None, :], mask=out_mask, other=0.0
        )
    tl.store(C_OUT + base + rm[:, None] * N + rn[None, :], acc, mask=out_mask)


@libentry()
@triton.jit
def _matrix_exp_lincomb_kernel(
    A,
    A2,
    A3,
    A6,
    S,
    COEFF,
    B_OUT,
    N,
    BATCH_MAT_NUMEL,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_mn = tl.cdiv(N, BLOCK_M) * num_pid_n
    pid_b = pid // num_pid_mn
    rem = pid % num_pid_mn
    pid_m = rem // num_pid_n
    pid_n = rem % num_pid_n
    base = pid_b * N * N

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < N) & (rn[None, :] < N)
    offs = base + rm[:, None] * N + rn[None, :]

    s_i = tl.load(S + pid_b)
    scale = tl.exp2(-s_i.to(A.dtype.element_ty))

    p0 = (rm[:, None] == rn[None, :]).to(A.dtype.element_ty)
    p1 = tl.load(A + offs, mask=mask, other=0.0) * scale
    p2 = tl.load(A2 + offs, mask=mask, other=0.0)
    p3 = tl.load(A3 + offs, mask=mask, other=0.0)
    p4 = tl.load(A6 + offs, mask=mask, other=0.0)

    for k in tl.static_range(5):
        c0 = tl.load(COEFF + k * 5 + 0)
        c1 = tl.load(COEFF + k * 5 + 1)
        c2 = tl.load(COEFF + k * 5 + 2)
        c3 = tl.load(COEFF + k * 5 + 3)
        c4 = tl.load(COEFF + k * 5 + 4)
        acc = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3 + c4 * p4
        tl.store(B_OUT + k * BATCH_MAT_NUMEL + offs, acc, mask=mask)


@libentry()
@triton.jit
def _matrix_exp_add_kernel(
    X,
    Y,
    Z,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_mn = tl.cdiv(N, BLOCK_M) * num_pid_n
    pid_b = pid // num_pid_mn
    rem = pid % num_pid_mn
    pid_m = rem // num_pid_n
    pid_n = rem % num_pid_n
    base = pid_b * N * N

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < N) & (rn[None, :] < N)
    offs = base + rm[:, None] * N + rn[None, :]

    x = tl.load(X + offs, mask=mask, other=0.0)
    y = tl.load(Y + offs, mask=mask, other=0.0)
    tl.store(Z + offs, x + y, mask=mask)


@libentry()
@triton.jit
def _matrix_exp_square_kernel(
    A,
    C_OUT,
    S,
    STEP,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_mn = tl.cdiv(N, BLOCK_M) * num_pid_n
    pid_b = pid // num_pid_mn
    rem = pid % num_pid_mn
    pid_m = rem // num_pid_n
    pid_n = rem % num_pid_n
    base = pid_b * N * N

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < N) & (rn[None, :] < N)
    offs = base + rm[:, None] * N + rn[None, :]

    s_i = tl.load(S + pid_b)
    if STEP >= s_i:
        tile = tl.load(A + offs, mask=mask, other=0.0)
        tl.store(C_OUT + offs, tile, mask=mask)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=A.dtype.element_ty)
        for k0 in range(0, N, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(
                A + base + rm[:, None] * N + rk[None, :],
                mask=(rm[:, None] < N) & (rk[None, :] < N),
                other=0.0,
            )
            b = tl.load(
                A + base + rk[:, None] * N + rn[None, :],
                mask=(rk[:, None] < N) & (rn[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(
                a, b, acc, input_precision="ieee", out_dtype=A.dtype.element_ty
            )
        tl.store(C_OUT + offs, acc, mask=mask)


def linalg_matrix_exp(A):
    logger.debug("GEMS LINALG_MATRIX_EXP")
    return _linalg_matrix_exp_impl(A)


def linalg_matrix_exp_out(A, *, out=None):
    logger.debug("GEMS LINALG_MATRIX_EXP_OUT")
    if out is None:
        raise TypeError("linalg_matrix_exp(): out must be provided for out variant")
    if out.dtype != A.dtype:
        raise RuntimeError(
            f"linalg_matrix_exp: dtype of out ({out.dtype}) does not match "
            f"dtype of input ({A.dtype})"
        )
    if out.device != A.device:
        raise RuntimeError(
            f"linalg_matrix_exp: device of out ({out.device}) does not match "
            f"device of input ({A.device})"
        )
    if out.shape != A.shape:
        raise RuntimeError(
            f"linalg_matrix_exp: shape of out {tuple(out.shape)} does not match "
            f"expected shape {tuple(A.shape)}"
        )
    out.copy_(_linalg_matrix_exp_impl(A))
    return out


def _linalg_matrix_exp_impl(A):
    if A.dim() < 2:
        raise RuntimeError(
            "linalg.matrix_exp: The input tensor A must have at least 2 dimensions."
        )
    m, n = A.shape[-2], A.shape[-1]
    if m != n:
        raise RuntimeError(
            "linalg.matrix_exp: A must be batches of square matrices, "
            f"but they are {m} by {n} matrices"
        )
    if A.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_matrix_exp currently supports float32 and float64 "
            f"only, got {A.dtype}"
        )

    if n == 0:
        return A.clone()
    if n == 1:
        return exp(A)

    batch_shape = A.shape[:-2]
    batch_count = math.prod(batch_shape)
    if batch_count == 0:
        return A.clone()

    A_work = A.contiguous().reshape(batch_count, n, n)
    theta = _THETA_18[A.dtype]
    dtype = A.dtype
    device = A.device
    coeff = _get_t18_coeff(dtype, device)

    if n <= _MATRIX_EXP_SMALL_N_MAX:
        out = torch.empty(batch_count, n, n, dtype=dtype, device=device)
        with torch_device_fn.device(device):
            _matrix_exp_small_kernel[(batch_count,)](
                A_work, out, coeff, n, theta, BLOCK_N=16, num_warps=1
            )
        return out.reshape(A.shape)

    S = torch.empty(batch_count, dtype=torch.int32, device=device)

    mat_numel = batch_count * n * n
    a2 = torch.empty(batch_count, n, n, dtype=dtype, device=device)
    a3 = torch.empty(batch_count, n, n, dtype=dtype, device=device)
    a6 = torch.empty(batch_count, n, n, dtype=dtype, device=device)
    b_mats = torch.empty(5, batch_count, n, n, dtype=dtype, device=device)
    t = torch.empty(batch_count, n, n, dtype=dtype, device=device)
    r = torch.empty(batch_count, n, n, dtype=dtype, device=device)

    block = 32 if (dtype == torch.float64 or n <= 32) else 64
    num_tiles = triton.cdiv(n, block) * triton.cdiv(n, block)
    grid_norm = (batch_count,)
    grid_mat = (batch_count * num_tiles,)

    with torch_device_fn.device(device):
        _matrix_exp_norm_kernel[grid_norm](
            A_work, S, n, theta, BLOCK_M=64, BLOCK_N=64, num_warps=4
        )
        _matrix_exp_bmm_kernel[grid_mat](
            A_work,
            A_work,
            A_work,
            a2,
            S,
            n,
            SCALE_A=True,
            SCALE_B=True,
            HAS_ACC=False,
            BLOCK_M=block,
            BLOCK_N=block,
            BLOCK_K=block,
            num_warps=4,
        )
        _matrix_exp_bmm_kernel[grid_mat](
            a2,
            A_work,
            A_work,
            a3,
            S,
            n,
            SCALE_A=False,
            SCALE_B=True,
            HAS_ACC=False,
            BLOCK_M=block,
            BLOCK_N=block,
            BLOCK_K=block,
            num_warps=4,
        )
        _matrix_exp_bmm_kernel[grid_mat](
            a3,
            a3,
            a3,
            a6,
            S,
            n,
            SCALE_A=False,
            SCALE_B=False,
            HAS_ACC=False,
            BLOCK_M=block,
            BLOCK_N=block,
            BLOCK_K=block,
            num_warps=4,
        )
        _matrix_exp_lincomb_kernel[grid_mat](
            A_work,
            a2,
            a3,
            a6,
            S,
            coeff,
            b_mats,
            n,
            mat_numel,
            BLOCK_M=block,
            BLOCK_N=block,
            num_warps=4,
        )
        _matrix_exp_bmm_kernel[grid_mat](
            b_mats[0],
            b_mats[4],
            b_mats[3],
            b_mats[3],
            S,
            n,
            SCALE_A=False,
            SCALE_B=False,
            HAS_ACC=True,
            BLOCK_M=block,
            BLOCK_N=block,
            BLOCK_K=block,
            num_warps=4,
        )
        _matrix_exp_add_kernel[grid_mat](
            b_mats[2],
            b_mats[3],
            t,
            n,
            BLOCK_M=block,
            BLOCK_N=block,
            num_warps=4,
        )
        _matrix_exp_bmm_kernel[grid_mat](
            t,
            b_mats[3],
            b_mats[1],
            r,
            S,
            n,
            SCALE_A=False,
            SCALE_B=False,
            HAS_ACC=True,
            BLOCK_M=block,
            BLOCK_N=block,
            BLOCK_K=block,
            num_warps=4,
        )

        s_max = max(S.tolist())
        if s_max > 0:
            tmp = torch.empty_like(r)
            for step in range(s_max):
                _matrix_exp_square_kernel[grid_mat](
                    r,
                    tmp,
                    S,
                    step,
                    n,
                    BLOCK_M=block,
                    BLOCK_N=block,
                    BLOCK_K=block,
                    num_warps=4,
                )
                r, tmp = tmp, r

    return r.reshape(A.shape)
