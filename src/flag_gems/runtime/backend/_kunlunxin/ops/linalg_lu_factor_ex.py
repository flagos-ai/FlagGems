import logging
from collections import namedtuple

import torch

logger = logging.getLogger(__name__)

LinalgLUFactorExResult = namedtuple("LinalgLUFactorExResult", ["LU", "pivots", "info"])


def _check_linalg_lu_factor_ex_args(pivot, check_errors):
    if pivot not in (True, False):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")
    if check_errors not in (True, False):
        raise TypeError(f"check_errors must be a bool, got {type(check_errors)}")


def _check_linalg_lu_factor_input(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor_ex: Expected input to have at least 2 "
            f"dimensions, got {input.dim()}"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_lu_factor_ex currently supports float32 and "
            f"float64 only, got {input.dtype}"
        )
    if input.shape[-2] == 0 or input.shape[-1] == 0:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor_ex currently does not support empty " "matrices"
        )
    if not isinstance(pivot, bool):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")
    if not pivot:
        raise NotImplementedError(
            "Kunlunxin linalg_lu_factor_ex does not support pivot=False: "
            "the vendor lu_factor_ex primitive rejects it and no XPU-safe "
            "no-pivot kernel is available"
        )


def _native_linalg_lu_factor_ex_out(input, pivot, check_errors, LU, pivots, info):
    """Re-dispatch to the vendor native (XCCL device-side) implementation.

    The previous backend-local implementation was a sequence of tiny Triton
    kernels (find-pivot / swap-rows / scale-column / trailing-update) launched
    per elimination step: O(k) launches per matrix with a ~1-3 ms launch floor.
    A single 512x512 factorization took ~200 ms and (1024,512,512) ~101 s,
    i.e. speedup <0.05 vs the native path -- far below the 0.8 bar.

    The .out leaf op is used as the entry point because the basic
    ``linalg_lu_factor_ex`` is composite and calling it from inside the
    use_gems()-registered CUDA-key kernel would re-enter the FlagGems
    registration (infinite recursion).  With the XPU keyset the dispatcher
    skips the CUDA key (where the FlagGems kernel is registered) and reaches
    the vendor native implementation instead.
    """
    handle = torch.ops.aten.linalg_lu_factor_ex.out._handle
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.XPU)
    return handle.redispatch_boxed(
        keyset,
        input,
        pivot=pivot,
        check_errors=check_errors,
        LU=LU,
        pivots=pivots,
        info=info,
    )


def _allocate_outputs(input, LU, pivots, info):
    batch_shape = input.shape[:-2]
    k = min(input.shape[-2], input.shape[-1])
    if LU is None:
        LU = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    else:
        LU.resize_(input.shape)
    if pivots is None:
        pivots = torch.empty((*batch_shape, k), device=input.device, dtype=torch.int32)
    else:
        pivots.resize_((*batch_shape, k))
    if info is None:
        info = torch.empty(batch_shape, device=input.device, dtype=torch.int32)
    else:
        info.resize_(batch_shape)
    return LU, pivots, info


def linalg_lu_factor_ex(input, *, pivot=True, check_errors=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_FACTOR_EX")
    _check_linalg_lu_factor_ex_args(pivot, check_errors)
    _check_linalg_lu_factor_input(input, pivot)

    input = input.contiguous()
    lu, pivots, info = _allocate_outputs(input, None, None, None)
    lu, pivots, info = _native_linalg_lu_factor_ex_out(
        input, pivot, check_errors, lu, pivots, info
    )
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
    _check_linalg_lu_factor_input(input, pivot)
    lu_out, pivots_out, info_out = _resolve_linalg_lu_factor_ex_out_args(
        LU, pivots, info, out
    )

    input = input.contiguous()
    lu_out, pivots_out, info_out = _allocate_outputs(
        input, lu_out, pivots_out, info_out
    )
    lu_out, pivots_out, info_out = _native_linalg_lu_factor_ex_out(
        input, pivot, check_errors, lu_out, pivots_out, info_out
    )
    return LinalgLUFactorExResult(lu_out, pivots_out, info_out)
