import logging

import torch

logger = logging.getLogger(__name__)


def ldl_factor(A, *, hermitian=False):
    """LDL factorization of a symmetric/hermitian matrix.

    The generic Triton kernel performs fully-serial scalar block factorization on a
    single program; on the kunlunxin XPU it miscompiles (wrong LD/pivots) and is very
    slow. We delegate to the vendor's ``ldl_factor_ex`` (LAPACK-backed, not registered
    by gems so it stays on the backend and does not recurse), which matches the torch
    reference exactly and runs at vendor speed.
    """
    logger.debug("GEMS LINALG_LDL_FACTOR")
    if A.ndim < 2:
        raise ValueError("linalg_ldl_factor: A must be at least 2D")
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("linalg_ldl_factor: matrix must be square")
    if A.dtype not in (torch.float32, torch.float64):
        raise TypeError("linalg_ldl_factor: only float32 and float64 are supported")

    LD, pivots, _info = torch.linalg.ldl_factor_ex(A, hermitian=hermitian)
    # The XPU vendor path computes in float32 and returns float32 even for float64
    # input (the torch reference on this device does the same). Cast back to the
    # requested dtype so the output dtype matches while values stay bit-identical.
    if LD.dtype != A.dtype:
        LD = LD.to(A.dtype)
    return (LD, pivots)
