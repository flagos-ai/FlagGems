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

"""Kunlunxin (XPU) linalg_lu / linalg_lu.out backend override.

Why this file exists
--------------------

The generic ``flag_gems/ops/linalg_lu.py`` implementation cannot be used on
this backend: both the small fused kernel (``_linalg_lu_fused_kernel``) and
the blocked panel path (``_lu_factor_panel_kernel``) embed 2D ``tl.sum`` /
``tl.max`` / ``tl.min`` reductions **inside** the elimination ``tl.range``
loop, and TritonXPU's ``CoreTiling`` pass rejects such kernels with

    Not All Reduce Op can be Optimized   (CoreTiling.cpp:203)

which surfaces as ``triton.runtime.errors.OutOfResources: out of resource:
uni_sram`` at compile time (measured on every test shape; see
``harness/solution/linalg_lu/`` for the baseline).

Implementation strategy
-----------------------

The backend-local implementation combines two primitives that already exist
and are known to compile/run on this backend (both pure Triton kernels, no
CPU/ATen/native/composite fallback):

1. ``_linalg_lu_factor`` (``_kunlunxin/ops/linalg_lu_factor.py``): the
   elimination itself.  Its pivot-search kernels use *only* 1-D reductions
   (``tl.argmax``/``tl.max`` on a 1-D vector), so they are accepted by the
   CoreTiling check; the row-swap / column-scale / trailing-update kernels
   are plain masked loads/stores.  The pivot step index ``J`` is a runtime
   scalar so the whole shape is compiled once instead of once per
   elimination step.

2. ``lu_unpack`` (``_kunlunxin/ops/lu_unpack.py``): P/L/U materialization.
   For ``m > 512`` the permutation matrix is built with an O(k) index-vector
   swap + scatter (instead of the generic per-row O(m*k) replay), with
   ``tl.debug_barrier()`` around the global-memory swaps; everything else
   delegates to the generic L/U kernels.

``linalg_lu`` therefore returns ``(P, L, U)`` with ``P @ L @ U == A`` and
torch's pivot semantics (``pivots`` are 1-based LAPACK row interchanges; the
permutation matrix is the product of the swap matrices in reverse order,
which is exactly what ``lu_unpack`` implements).

``linalg_lu_out`` computes the same factorization and writes the results
back through ``torch.ops.aten._copy_from`` (the raw native strided-copy
engine; the gems-registered ``copy_`` is deliberately avoided to prevent
nested dispatch through the overridden operator).

Limitations
-----------

- ``pivot=False`` is not supported (mirrors ``_linalg_lu_factor`` and the
  vendor ``lu_factor_ex`` primitive); the test/benchmark matrix only
  exercises ``pivot=True`` for this vendor.
- Only ``float32`` / ``float64`` (vendor ``_check_linalg_lu_factor``).
- Empty matrices (``m == 0`` or ``n == 0``) are not supported.
"""

import logging

import torch

from flag_gems.runtime import torch_device_fn

from .linalg_lu_factor import _check_linalg_lu_factor, _linalg_lu_factor
from .lu_unpack import lu_unpack

logger = logging.getLogger(__name__)


def _resolve_linalg_lu_out_args(P, L, U, out):
    if out is not None:
        if P is not None or L is not None or U is not None:
            raise TypeError("linalg_lu(): out and P/L/U cannot both be set")
        if len(out) != 3:
            raise TypeError(
                "linalg_lu(): out must be a tuple of 3 tensors, " f"got {len(out)}"
            )
        return out
    if P is None or L is None or U is None:
        raise TypeError("linalg_lu(): P, L and U must all be provided for out variant")
    return P, L, U


def linalg_lu(input, *, pivot=True):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU")
    _check_linalg_lu_factor(input, pivot)
    with torch_device_fn.device(input.device):
        lu, pivots = _linalg_lu_factor(input, pivot)
        P, L, U = lu_unpack(lu, pivots, unpack_data=True, unpack_pivots=True)
    return P, L, U


def linalg_lu_out(input, *, pivot=True, P=None, L=None, U=None, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU.OUT")
    _check_linalg_lu_factor(input, pivot)
    p_out, l_out, u_out = _resolve_linalg_lu_out_args(P, L, U, out)
    with torch_device_fn.device(input.device):
        lu, pivots = _linalg_lu_factor(input, pivot)
        P_res, L_res, U_res = lu_unpack(lu, pivots, unpack_data=True, unpack_pivots=True)
    # Write back through the raw native strided-copy engine
    # (``aten::_copy_from``) instead of the gems-registered ``copy_``
    # to avoid a nested dispatch through the overridden operator.
    if P_res.numel() > 0:
        if p_out.numel() != P_res.numel():
            p_out.resize_(P_res.shape)
        torch.ops.aten._copy_from(P_res, p_out, False)
    else:
        p_out.resize_((0,))
    if l_out.shape != L_res.shape:
        l_out.resize_(L_res.shape)
    torch.ops.aten._copy_from(L_res, l_out, False)
    if u_out.shape != U_res.shape:
        u_out.resize_(U_res.shape)
    torch.ops.aten._copy_from(U_res, u_out, False)
    return (p_out, l_out, u_out)