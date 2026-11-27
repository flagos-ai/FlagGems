# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch

from .linalg_lu_solve import linalg_lu_solve

logger = logging.getLogger(__name__)


def _lu_solve_impl(B, LU_data, LU_pivots, out):
    if B.dim() < 2:
        raise ValueError(
            "lu_solve: The input tensor B must have at least 2 dimensions."
        )
    if LU_data.dim() < 2:
        raise ValueError("lu_solve: LU_data must have at least 2 dimensions.")

    m = LU_data.shape[-1]
    if LU_data.shape[-2] != m:
        raise ValueError("lu_solve: LU_data must be a batch of square matrices.")
    if B.shape[-2] != m:
        raise ValueError(
            "lu_solve: The number of rows of B must be equal to the order of "
            "the matrices in LU_data."
        )
    if LU_pivots.shape[-1] != m:
        raise ValueError(
            "lu_solve: Number of pivots per batch should be same as the "
            "dimension of the matrix."
        )
    if B.dtype != LU_data.dtype:
        raise ValueError("lu_solve: B and LU_data must have the same dtype.")
    if LU_pivots.dtype != torch.int32:
        raise ValueError("lu_solve: pivots should be a Tensor of scalar type int32.")

    batch_shape = torch.broadcast_shapes(B.shape[:-2], LU_data.shape[:-2])
    batch = 1
    for dim in batch_shape:
        batch *= dim
    n_rhs = B.shape[-1]
    org_shape = tuple(batch_shape) + (m, n_rhs)

    if n_rhs == 0 or m == 0 or batch == 0:
        if out is not None:
            return out
        return torch.empty(org_shape, dtype=B.dtype, device=B.device)

    if out is not None and (
        out.is_contiguous() and out.dtype == B.dtype and out.shape == org_shape
    ):
        # A contiguous, shape/dtype-matching ``out`` is handed to the kernels
        # as the solve target, so the solution is written in place and no
        # trailing full-tensor copy is needed. Non-contiguous or mismatched
        # buffers keep the copy-back path below.
        clean_out = None
        X = out
    elif out is not None:
        # Non-contiguous ``out`` cannot be written by the kernels directly.
        clean_out = out
        X = None
    else:
        clean_out = None
        X = None

    B_v = B.expand(batch_shape + B.shape[-2:]).reshape(batch, m, n_rhs)
    LU_v = LU_data.expand(batch_shape + LU_data.shape[-2:]).reshape(batch, m, m)
    PIV_v = LU_pivots.expand(batch_shape + (m,)).reshape(batch, m)

    # Delegate to the linalg_lu_solve kernels. When ``out`` is usable as the
    # solve target they write it directly (which aliases ``out``); otherwise
    # they allocate their own output and it is copied below.
    result = linalg_lu_solve(LU_v, PIV_v, B_v, out=X)

    if clean_out is not None:
        clean_out.copy_(result.reshape(org_shape))
        return clean_out
    if out is None:
        return result.reshape(org_shape)
    return result


def lu_solve(B, LU_data, LU_pivots, *, out=None):
    logger.debug("GEMS_ENFLAME LU_SOLVE")
    if out is not None:
        return _lu_solve_impl(B, LU_data, LU_pivots, out)
    return _lu_solve_impl(B, LU_data, LU_pivots, None)


def lu_solve_out(B, LU_data, LU_pivots, *, out):
    logger.debug("GEMS_ENFLAME LU_SOLVE_OUT")
    return _lu_solve_impl(B, LU_data, LU_pivots, out)
