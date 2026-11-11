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

"""Kunlunxin backend override for ``linalg_inv_ex``.

The generic ``_gj_inv_kernel`` keeps a ``BLOCK_N``-by``2*BLOCK_N`` augmented
register tile live across a serial pivot loop; the ``TritonXPULegalize`` MLIR
pipeline crashes on that shape/dependency combination on this backend.

The inverse is computed exactly as in the ``linalg_inv`` override (LU factor +
column substitution).  ``info`` is recovered from the U diagonal of the factors:
a zero diagonal element at index ``k`` means the ``k``-th pivot vanished, and
``info`` stores ``k + 1`` (LAPACK ordering) for the first such index, else 0.
"""

import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .linalg_lu_factor import _linalg_lu_factor
from .linalg_lu_solve import _xpu_lu_solve_impl

logger = logging.getLogger(__name__)

LinalgInvExResult = namedtuple("LinalgInvExResult", ["inverse", "info"])


@libentry()
@triton.jit
def _lu_diag_info_kernel(LU, INFO, n, BLOCK_N: tl.constexpr):
    """Report the first exactly-zero U(k, k) as ``k + 1`` (0 if none)."""
    b = tl.program_id(0).to(tl.int64)
    idx = tl.arange(0, BLOCK_N)
    valid = idx < n
    safe = tl.where(valid, idx, 0)
    base = LU + b * (n * n)
    vals = tl.load(base + safe * n + safe)
    diag = tl.where(valid, vals, 1.0)
    zero = valid & (diag == 0.0)
    pos = tl.min(tl.where(zero, idx, BLOCK_N), axis=0)
    info = tl.where(pos < BLOCK_N, pos + 1, 0)
    tl.store(INFO + b, info.to(tl.int32))


@libentry()
@triton.jit
def _lu_info_flag_kernel(INFO, FLAG, B, BLOCK_B: tl.constexpr):
    """Reduce the per-matrix info values to a single 0/1 flag on device."""
    offs = tl.arange(0, BLOCK_B)
    acc = 0
    for base in range(0, B, BLOCK_B):
        idx = base + offs
        valid = idx < B
        safe = tl.where(valid, idx, 0)
        vals = tl.load(INFO + safe)
        vals = tl.where(valid, vals, 0)
        acc = tl.maximum(acc, tl.max(tl.abs(vals), axis=0))
    tl.store(FLAG, (acc != 0).to(tl.int32))


def linalg_inv_ex(A, *, check_errors=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_INV_EX")

    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    nn = n * n

    out = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    info = torch.empty(batch_shape, dtype=torch.int32, device=A.device)
    if nn == 0:
        return LinalgInvExResult(out, info)
    batch = A.numel() // nn
    if batch == 0:
        return LinalgInvExResult(out, info)

    A_work = A.reshape(batch, n, n).contiguous()
    LU_work, piv_work = _linalg_lu_factor(A_work, True)

    eye = torch.eye(n, dtype=A_work.dtype, device=A_work.device)
    rhs = eye.unsqueeze(0).expand(batch, n, n)
    with torch_device_fn.device(A.device):
        inv = _xpu_lu_solve_impl(LU_work, piv_work, rhs, batch, n, n)
        _lu_diag_info_kernel[(batch,)](
            LU_work,
            info.reshape(batch),
            n,
            BLOCK_N=max(1, triton.next_power_of_2(n)),
            num_warps=1,
        )
    out.copy_(inv.reshape(A.shape))

    if check_errors:
        flag = torch.empty(1, dtype=torch.int32, device=A.device)
        block_b = max(1, triton.next_power_of_2(min(batch, 1024)))
        with torch_device_fn.device(A.device):
            _lu_info_flag_kernel[(1,)](info.reshape(batch), flag, batch, block_b)
        if flag.item() != 0:
            raise torch.linalg.LinAlgError(
                "torch.linalg.inv_ex: The diagonal element of the LU "
                "decomposition is zero."
            )

    return LinalgInvExResult(out, info)


__all__ = ["linalg_inv_ex"]
