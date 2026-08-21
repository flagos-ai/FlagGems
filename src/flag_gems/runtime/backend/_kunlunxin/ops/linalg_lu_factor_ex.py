import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .linalg_lu_factor import _check_linalg_lu_factor, _linalg_lu_factor

logger = logging.getLogger(__name__)

LinalgLUFactorExResult = namedtuple("LinalgLUFactorExResult", ["LU", "pivots", "info"])


@triton.jit
def _lu_factor_info_kernel(
    LU,
    INFO,
    M,
    N,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Scan the diagonal of the LU factors to find the first zero/NaN pivot.

    LAPACK-style info: 1-indexed position of the first zero (or NaN, which can
    arise from 0/0 division e.g. for an all-zero input) pivot, 0 if none.
    """
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    diag = tl.load(LU + pid * M * N + offsets * (N + 1), mask=mask, other=1.0)

    sentinel = K + 1
    is_zero = (diag == 0) | (diag != diag)
    candidates = tl.where(is_zero & mask, offsets + 1, sentinel)
    first_zero = tl.min(candidates, axis=0)
    info = tl.where(first_zero == sentinel, 0, first_zero).to(tl.int32)
    tl.store(INFO + pid, info)


def _lu_factor_info(lu):
    """Compute the LAPACK-style info tensor by scanning the LU diagonal."""
    m, n = lu.shape[-2], lu.shape[-1]
    k = min(m, n)
    batch_shape = lu.shape[:-2]
    batch = lu.numel() // (m * n)
    info = torch.empty(batch_shape, device=lu.device, dtype=torch.int32)

    with torch_device_fn.device(lu.device):
        _lu_factor_info_kernel[(batch,)](
            lu,
            info,
            m,
            n,
            k,
            triton.next_power_of_2(k),
            num_warps=4,
        )
    return info


def _check_linalg_lu_factor_ex_args(pivot, check_errors):
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")
    if check_errors not in (True, False):
        raise TypeError(f"check_errors must be a bool, got {type(check_errors)}")


def _check_lu_factor_errors(info):
    failed = info != 0
    if not torch.any(failed).item():
        return

    # Extract the first non-zero info on-device: the Kunlunxin copy/`to`
    # overrides reject cross-device .cpu() copies, so avoid them here.
    first_idx = torch.argmax(failed.to(torch.int32).flatten()).item()
    first_info = int(first_idx) + 1
    raise RuntimeError(
        "torch.linalg.lu_factor_ex: U[{},{}] is zero and using it on lu_solve "
        "would result in a division by zero. If you still want to perform the "
        "factorization, pass check_errors=False.".format(first_info, first_info)
    )


def linalg_lu_factor_ex(input, *, pivot=True, check_errors=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_FACTOR_EX")
    _check_linalg_lu_factor_ex_args(pivot, check_errors)
    _check_linalg_lu_factor(input, pivot)

    lu, pivots = _linalg_lu_factor(input, pivot)
    info = _lu_factor_info(lu)

    if check_errors:
        _check_lu_factor_errors(info)

    return LinalgLUFactorExResult(lu, pivots, info)


def _resolve_linalg_lu_factor_ex_out_args(LU, pivots, info, out):
    if out is not None:
        if LU is not None or pivots is not None or info is not None:
            raise TypeError(
                "linalg_lu_factor_ex(): out and LU/pivots/info cannot both be set"
            )
        if len(out) != 3:
            raise TypeError(
                "linalg_lu_factor_ex(): out must be a tuple of 3 tensors, "
                f"got {len(out)}"
            )
        return out
    if LU is None or pivots is None or info is None:
        raise TypeError(
            "linalg_lu_factor_ex(): LU, pivots and info must all be provided "
            "for out variant"
        )
    return LU, pivots, info


def linalg_lu_factor_ex_out(
    input,
    *,
    pivot=True,
    check_errors=False,
    LU=None,
    pivots=None,
    info=None,
    out=None,
):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_FACTOR_EX.OUT")
    _check_linalg_lu_factor_ex_args(pivot, check_errors)
    lu_out, pivots_out, info_out = _resolve_linalg_lu_factor_ex_out_args(
        LU, pivots, info, out
    )

    res = linalg_lu_factor_ex(input, pivot=pivot, check_errors=False)
    lu_out.resize_(res.LU.shape)
    pivots_out.resize_(res.pivots.shape)
    info_out.resize_(res.info.shape)
    lu_out.copy_(res.LU)
    pivots_out.copy_(res.pivots)
    info_out.copy_(res.info)

    if check_errors:
        _check_lu_factor_errors(info_out)

    return LinalgLUFactorExResult(lu_out, pivots_out, info_out)