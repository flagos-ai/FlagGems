import logging
import math

import torch
import triton

from flag_gems.ops.exp import exp
from flag_gems.ops.linalg_matrix_exp import (
    _THETA_18,
    _get_t18_coeff,
    _matrix_exp_add_kernel,
    _matrix_exp_bmm_kernel,
    _matrix_exp_lincomb_kernel,
    _matrix_exp_norm_kernel,
    _matrix_exp_square_kernel,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


_MATRIX_EXP_LINCOMB_BLOCK = 32

_ASCEND_DUMMY_BATCH = 40


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

    dtype = A.dtype
    device = A.device

    padded_count = batch_count + _ASCEND_DUMMY_BATCH
    A_work = torch.zeros(padded_count, n, n, dtype=dtype, device=device)
    A_work[_ASCEND_DUMMY_BATCH:] = A.contiguous().reshape(batch_count, n, n)

    theta = _THETA_18[dtype]
    coeff = _get_t18_coeff(dtype, device)

    S = torch.empty(padded_count, dtype=torch.int32, device=device)

    mat_numel = padded_count * n * n
    a2 = torch.empty(padded_count, n, n, dtype=dtype, device=device)
    a3 = torch.empty(padded_count, n, n, dtype=dtype, device=device)
    a6 = torch.empty(padded_count, n, n, dtype=dtype, device=device)
    b_mats = torch.empty(5, padded_count, n, n, dtype=dtype, device=device)
    t = torch.empty(padded_count, n, n, dtype=dtype, device=device)
    r = torch.empty(padded_count, n, n, dtype=dtype, device=device)

    block = 32 if (dtype == torch.float64 or n <= 32) else 64
    num_tiles = triton.cdiv(n, block) * triton.cdiv(n, block)
    num_tiles_lincomb = triton.cdiv(n, _MATRIX_EXP_LINCOMB_BLOCK) * triton.cdiv(
        n, _MATRIX_EXP_LINCOMB_BLOCK
    )
    grid_norm = (padded_count,)
    grid_mat = (padded_count * num_tiles,)
    grid_lincomb = (padded_count * num_tiles_lincomb,)

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
        _matrix_exp_lincomb_kernel[grid_lincomb](
            A_work,
            a2,
            a3,
            a6,
            S,
            coeff,
            b_mats,
            n,
            mat_numel,
            BLOCK_M=_MATRIX_EXP_LINCOMB_BLOCK,
            BLOCK_N=_MATRIX_EXP_LINCOMB_BLOCK,
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

    return r[_ASCEND_DUMMY_BATCH:].reshape(A.shape)
