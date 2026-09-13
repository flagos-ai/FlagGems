import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_matrix_exp import (
    _THETA_18,
    _get_t18_coeff,
    _linalg_matrix_exp_impl,
    _matrix_exp_add_kernel,
    _matrix_exp_lincomb_kernel,
    _matrix_exp_norm_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _metax_matrix_exp_bmm_fp64_kernel(
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
):
    pid = tle.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_mn = tl.cdiv(N, BLOCK_M) * num_pid_n
    pid_b = pid // num_pid_mn
    rem = pid % num_pid_mn
    pid_m = rem // num_pid_n
    pid_n = rem % num_pid_n
    base = pid_b * N * N

    scale = tl.full((), 1.0, dtype=tl.float64)
    if SCALE_A or SCALE_B:
        s_i = tl.load(S + pid_b)
        scale = tl.exp2(-s_i.to(tl.float64))

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (rm[:, None] < N) & (rn[None, :] < N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for k in range(N):
        a_col = tl.load(A + base + rm * N + k, mask=rm < N, other=0.0)
        b_row = tl.load(B + base + k * N + rn, mask=rn < N, other=0.0)
        if SCALE_A:
            a_col = a_col * scale
        if SCALE_B:
            b_row = b_row * scale
        acc += a_col[:, None] * b_row[None, :]

    if HAS_ACC:
        acc += tl.load(
            C_IN + base + rm[:, None] * N + rn[None, :], mask=out_mask, other=0.0
        )
    tl.store(C_OUT + base + rm[:, None] * N + rn[None, :], acc, mask=out_mask)


@libentry()
@triton.jit
def _metax_matrix_exp_square_fp64_kernel(
    A,
    C_OUT,
    S,
    STEP,
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

    s_i = tl.load(S + pid_b)
    if STEP >= s_i:
        tile = tl.load(A + offs, mask=mask, other=0.0)
        tl.store(C_OUT + offs, tile, mask=mask)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
        for k in range(N):
            a_col = tl.load(A + base + rm * N + k, mask=rm < N, other=0.0)
            b_row = tl.load(A + base + k * N + rn, mask=rn < N, other=0.0)
            acc += a_col[:, None] * b_row[None, :]
        tl.store(C_OUT + offs, acc, mask=mask)


def linalg_matrix_exp(A):
    logger.debug("GEMS_METAX LINALG_MATRIX_EXP")
    if A.dtype == torch.float64 and A.dim() >= 2 and A.shape[-1] == A.shape[-2] > 1:
        return _linalg_matrix_exp_fp64_impl(A)
    return _linalg_matrix_exp_impl(A)


def linalg_matrix_exp_out(A, *, out=None):
    logger.debug("GEMS_METAX LINALG_MATRIX_EXP_OUT")
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
    if A.dtype == torch.float64 and A.dim() >= 2 and A.shape[-1] == A.shape[-2] > 1:
        out.copy_(_linalg_matrix_exp_fp64_impl(A))
    else:
        out.copy_(_linalg_matrix_exp_impl(A))
    return out


def _linalg_matrix_exp_fp64_impl(A):
    batch_shape = A.shape[:-2]
    batch_count = math.prod(batch_shape)
    if batch_count == 0:
        return A.clone()

    n = A.shape[-1]
    dtype = A.dtype
    device = A.device

    A_work = A.contiguous().reshape(batch_count, n, n)
    theta = _THETA_18[dtype]
    coeff = _get_t18_coeff(dtype, device)

    S = torch.empty(batch_count, dtype=torch.int32, device=device)

    mat_numel = batch_count * n * n
    a2 = torch.empty(batch_count, n, n, dtype=dtype, device=device)
    a3 = torch.empty(batch_count, n, n, dtype=dtype, device=device)
    a6 = torch.empty(batch_count, n, n, dtype=dtype, device=device)
    b_mats = torch.empty(5, batch_count, n, n, dtype=dtype, device=device)
    t = torch.empty(batch_count, n, n, dtype=dtype, device=device)
    r = torch.empty(batch_count, n, n, dtype=dtype, device=device)

    block = 32
    num_tiles = triton.cdiv(n, block) * triton.cdiv(n, block)
    grid_norm = (batch_count,)
    grid_mat = (batch_count * num_tiles,)

    with torch_device_fn.device(device):
        _matrix_exp_norm_kernel[grid_norm](
            A_work, S, n, theta, BLOCK_M=64, BLOCK_N=64, num_warps=4
        )
        _metax_matrix_exp_bmm_fp64_kernel[grid_mat](
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
            num_warps=4,
        )
        _metax_matrix_exp_bmm_fp64_kernel[grid_mat](
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
            num_warps=4,
        )
        _metax_matrix_exp_bmm_fp64_kernel[grid_mat](
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
        _metax_matrix_exp_bmm_fp64_kernel[grid_mat](
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
        _metax_matrix_exp_bmm_fp64_kernel[grid_mat](
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
            num_warps=4,
        )

        s_max = max(S.tolist())
        if s_max > 0:
            tmp = torch.empty_like(r)
            for step in range(s_max):
                _metax_matrix_exp_square_fp64_kernel[grid_mat](
                    r,
                    tmp,
                    S,
                    step,
                    n,
                    BLOCK_M=block,
                    BLOCK_N=block,
                    num_warps=4,
                )
                r, tmp = tmp, r

    return r.reshape(A.shape)
