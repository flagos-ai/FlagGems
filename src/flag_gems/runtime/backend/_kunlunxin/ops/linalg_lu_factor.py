import logging

import torch

logger = logging.getLogger(__name__)


def _check_linalg_lu_factor(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor: Expected input to have at least 2 dimensions, "
            f"got {input.dim()}"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently supports float32 and float64 only, "
            f"got {input.dtype}"
        )
    if input.shape[-2] == 0 or input.shape[-1] == 0:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently does not support empty matrices"
        )
    if not isinstance(pivot, bool):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")


def _linalg_lu_factor(input, pivot):
    _check_linalg_lu_factor(input, pivot)
    if not pivot:
        raise NotImplementedError(
            "Kunlunxin linalg_lu_factor does not support pivot=False: "
            "the vendor lu_factor_ex primitive rejects it and no XPU-safe "
            "no-pivot kernel is available"
        )

    lu, pivots, info = torch.linalg.lu_factor_ex(input, pivot=True)
    if info.numel() and bool(torch.any(info != 0)):
        raise RuntimeError("torch.linalg.lu_factor: LU factorization failed")
    if lu.dtype != input.dtype:
        lu = lu.to(input.dtype)
    return lu, pivots


def linalg_lu_factor(input, *, pivot=True):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_FACTOR")
    return _linalg_lu_factor(input, pivot)


def _resolve_linalg_lu_factor_out_args(input, LU, pivots):
    if LU is None or pivots is None:
        raise TypeError(
            "linalg_lu_factor(): LU and pivots must both be provided "
            "for out variant"
        )
    if LU.device != input.device or pivots.device != input.device:
        raise RuntimeError("linalg_lu_factor(): out tensors must be on input's device")
    if LU.dtype != input.dtype:
        raise RuntimeError("linalg_lu_factor(): LU out tensor must match input dtype")
    if pivots.dtype != torch.int32:
        raise RuntimeError("linalg_lu_factor(): pivots out tensor must have dtype int32")
    return LU, pivots


def linalg_lu_factor_out(input, *, pivot=True, LU=None, pivots=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_FACTOR_OUT")
    lu_out, pivots_out = _resolve_linalg_lu_factor_out_args(input, LU, pivots)
    lu, pivots_result = _linalg_lu_factor(input, pivot)
    lu_out.resize_(lu.shape)
    pivots_out.resize_(pivots_result.shape)
    lu_out.copy_(lu)
    pivots_out.copy_(pivots_result)
    return lu_out, pivots_out
