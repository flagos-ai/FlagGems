import logging

import torch

logger = logging.getLogger(__name__)


def _check_linalg_ldl_factor(A, hermitian, check_errors):
    if A.ndim < 2:
        raise ValueError("linalg_ldl_factor: A must be at least 2D")
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("linalg_ldl_factor: matrix must be square")
    if not isinstance(hermitian, bool):
        raise TypeError(f"hermitian must be a bool, got {type(hermitian)}")
    if not isinstance(check_errors, bool):
        raise TypeError(f"check_errors must be a bool, got {type(check_errors)}")
    if A.dtype == torch.float64:
        raise NotImplementedError(
            "Kunlunxin linalg_ldl_factor does not support float64: "
            "the XPU backend has fp64_enabled=False"
        )
    if A.dtype != torch.float32:
        raise TypeError("Kunlunxin linalg_ldl_factor supports float32 only")


def _linalg_ldl_factor_ex(A, hermitian, check_errors):
    _check_linalg_ldl_factor(A, hermitian, check_errors)
    LD = torch.empty_like(A)
    pivots = torch.empty(*A.shape[:-1], dtype=torch.int32, device=A.device)
    info = torch.empty(A.shape[:-2], dtype=torch.int32, device=A.device)
    return torch.ops.aten.linalg_ldl_factor_ex.out(
        A,
        hermitian=hermitian,
        check_errors=check_errors,
        LD=LD,
        pivots=pivots,
        info=info,
    )


def ldl_factor(A, *, hermitian=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LDL_FACTOR")
    LD, pivots, _info = _linalg_ldl_factor_ex(A, hermitian, False)
    return (LD, pivots)


def ldl_factor_ex(A, hermitian=False, check_errors=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LDL_FACTOR_EX")
    return _linalg_ldl_factor_ex(A, hermitian, check_errors)
