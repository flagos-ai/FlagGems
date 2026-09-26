import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.exp import exp
from flag_gems.ops.linalg_matrix_exp import _T18_B, _THETA_18
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)


@libentry()
@triton.jit
def _colsum_kernel(
    A,
    CS,
    NP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)
    num_ctile = NP // BLOCK
    pid_b = pid // num_ctile
    pid_c = pid % num_ctile
    base = pid_b * NP * NP

    cols = pid_c * BLOCK + tl.arange(0, BLOCK)
    rows = tl.arange(0, NP)
    offs_t = base + rows[None, :] * NP + cols[:, None]
    at = tl.load(A + offs_t)
    cs = tl.sum(tl.abs(at), axis=1)
    tl.store(CS + pid_b * NP + cols, cs)


@libentry()
@triton.jit
def _s_kernel(
    CS,
    S,
    NP: tl.constexpr,
    THETA,
):
    pid = tle.program_id(0)
    idx = tl.arange(0, NP)
    cs = tl.load(CS + pid * NP + idx)
    norm = tl.max(cs, axis=0)
    s = tl.maximum(tl.ceil(tl.log2(norm / THETA) * LOG2E), 0.0)
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
    NP,
    SCALE_A: tl.constexpr,
    SCALE_B: tl.constexpr,
    HAS_ACC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = NP // BLOCK_N
    num_pid_mn = (NP // BLOCK_M) * num_pid_n
    pid_b = pid // num_pid_mn
    rem = pid % num_pid_mn
    pid_m = rem // num_pid_n
    pid_n = rem % num_pid_n
    base = pid_b * NP * NP

    scale = tl.full((), 1.0, dtype=A.dtype.element_ty)
    if SCALE_A or SCALE_B:
        s_i = tl.load(S + pid_b)
        scale = tl.exp2(-s_i.to(A.dtype.element_ty) * LN2)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=A.dtype.element_ty)
    for k0 in range(0, NP, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A + base + rm[:, None] * NP + rk[None, :])
        b = tl.load(B + base + rk[:, None] * NP + rn[None, :])
        if SCALE_A:
            a = a * scale
        if SCALE_B:
            b = b * scale
        acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=A.dtype.element_ty)

    if HAS_ACC:
        acc += tl.load(C_IN + base + rm[:, None] * NP + rn[None, :])
    tl.store(C_OUT + base + rm[:, None] * NP + rn[None, :], acc)


@libentry()
@triton.jit
def _matrix_exp_lincomb_kernel(
    A1,
    A2,
    A3,
    A6,
    EYE,
    B_OUT,
    NP,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    C4: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = NP // BLOCK_N
    num_pid_mn = (NP // BLOCK_M) * num_pid_n
    pid_b = pid // num_pid_mn
    rem = pid % num_pid_mn
    pid_m = rem // num_pid_n
    pid_n = rem % num_pid_n
    base = pid_b * NP * NP

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = base + rm[:, None] * NP + rn[None, :]

    p0 = tl.load(EYE + offs)
    p1 = tl.load(A1 + offs)
    p2 = tl.load(A2 + offs)
    p3 = tl.load(A3 + offs)
    p4 = tl.load(A6 + offs)

    acc = C0 * p0 + C1 * p1 + C2 * p2 + C3 * p3 + C4 * p4
    tl.store(B_OUT + offs, acc)


@libentry()
@triton.jit
def _matrix_exp_add_kernel(
    X,
    Y,
    Z,
    NP,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = NP // BLOCK_N
    num_pid_mn = (NP // BLOCK_M) * num_pid_n
    pid_b = pid // num_pid_mn
    rem = pid % num_pid_mn
    pid_m = rem // num_pid_n
    pid_n = rem % num_pid_n
    base = pid_b * NP * NP

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = base + rm[:, None] * NP + rn[None, :]

    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    tl.store(Z + offs, x + y)


@libentry()
@triton.jit
def _matrix_exp_square_kernel(
    A,
    C_OUT,
    S,
    STEP,
    NP,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = NP // BLOCK_N
    num_pid_mn = (NP // BLOCK_M) * num_pid_n
    pid_b = pid // num_pid_mn
    rem = pid % num_pid_mn
    pid_m = rem // num_pid_n
    pid_n = rem % num_pid_n
    base = pid_b * NP * NP

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = base + rm[:, None] * NP + rn[None, :]

    s_i = tl.load(S + pid_b)
    tile = tl.load(A + offs)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=A.dtype.element_ty)
    for k0 in range(0, NP, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A + base + rm[:, None] * NP + rk[None, :])
        b = tl.load(A + base + rk[:, None] * NP + rn[None, :])
        acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=A.dtype.element_ty)
    still_squaring = STEP < s_i
    out = tl.where(still_squaring, acc, tile)
    tl.store(C_OUT + offs, out)


def _pick_block(n):
    if n <= 16:
        return 16
    if n <= 32:
        return 32
    return 64


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

    dtype = A.dtype
    device = A.device
    theta = _THETA_18[dtype]

    A_work = A.contiguous().reshape(batch_count, n, n)

    block = _pick_block(n)
    np = triton.cdiv(n, block) * block

    if np != n:
        A_pad = torch.zeros(batch_count, np, np, dtype=dtype, device=device)
        A_pad[:, :n, :n] = A_work
    else:
        A_pad = A_work

    S = torch.empty(batch_count, dtype=torch.int32, device=device)
    CS = torch.empty(batch_count, np, dtype=dtype, device=device)
    eye = (
        torch.eye(np, dtype=dtype, device=device)
        .unsqueeze(0)
        .expand(batch_count, np, np)
        .contiguous()
    )

    a2 = torch.empty(batch_count, np, np, dtype=dtype, device=device)
    a3 = torch.empty(batch_count, np, np, dtype=dtype, device=device)
    a6 = torch.empty(batch_count, np, np, dtype=dtype, device=device)
    a1 = torch.empty(batch_count, np, np, dtype=dtype, device=device)
    b_mats = torch.empty(5, batch_count, np, np, dtype=dtype, device=device)
    t = torch.empty(batch_count, np, np, dtype=dtype, device=device)
    r = torch.empty(batch_count, np, np, dtype=dtype, device=device)

    num_tiles = (np // block) * (np // block)
    grid_cs = (batch_count * (np // block),)
    grid_s = (batch_count,)
    grid_mat = (batch_count * num_tiles,)

    with torch_device_fn.device(device):
        _colsum_kernel[grid_cs](A_pad, CS, np, BLOCK=block, num_warps=4)
        _s_kernel[grid_s](CS, S, np, theta, num_warps=1)
        _matrix_exp_bmm_kernel[grid_mat](
            A_pad,
            A_pad,
            A_pad,
            a2,
            S,
            np,
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
            A_pad,
            A_pad,
            a3,
            S,
            np,
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
            np,
            SCALE_A=False,
            SCALE_B=False,
            HAS_ACC=False,
            BLOCK_M=block,
            BLOCK_N=block,
            BLOCK_K=block,
            num_warps=4,
        )
        _matrix_exp_bmm_kernel[grid_mat](
            A_pad,
            eye,
            A_pad,
            a1,
            S,
            np,
            SCALE_A=True,
            SCALE_B=False,
            HAS_ACC=False,
            BLOCK_M=block,
            BLOCK_N=block,
            BLOCK_K=block,
            num_warps=4,
        )
        for k in range(5):
            _matrix_exp_lincomb_kernel[grid_mat](
                a1,
                a2,
                a3,
                a6,
                eye,
                b_mats[k],
                np,
                _T18_B[k][0],
                _T18_B[k][1],
                _T18_B[k][2],
                _T18_B[k][3],
                _T18_B[k][4],
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
            np,
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
            np,
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
            np,
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
                    np,
                    BLOCK_M=block,
                    BLOCK_N=block,
                    BLOCK_K=block,
                    num_warps=4,
                )
                r, tmp = tmp, r

    if np != n:
        r = r[:, :n, :n].contiguous()
    return r.reshape(A.shape)


def linalg_matrix_exp(A):
    logger.debug("GEMS_KUNLUNXIN LINALG_MATRIX_EXP")
    return _linalg_matrix_exp_impl(A)


def linalg_matrix_exp_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_MATRIX_EXP_OUT")
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
