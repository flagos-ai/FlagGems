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

logger = logging.getLogger(__name__)

# Redispatch key used to reach the vendor (torch_xmlir) linalg_solve_triangular
# kernel. The vendor registers its LAPACK-backed TRSM implementation under the
# XPU dispatch key, while the gems override that routed here occupies the CUDA
# key. Redispatching with only the XPU key executes the vendor kernel directly
# on the Kunlunxin device: no CPU fallback and no recursion into this override.
#
# Why not the generic Triton implementation: both of its kernels
# (`_small_diag_kernel_notle` / `_kslice_trsm_kernel_notle`) rely heavily on
# partially-masked loads (`mask=..., other=0.0`). On this XPU backend masked
# loads are unreliable at the compiler level — `other` is not honoured and,
# when part of the lanes are masked off, even the *valid* lanes can return
# wrong data — so the generic path produces wrong results for nearly every
# (n, k, upper) combination (baseline: 127 failed / 27 passed).
_VENDOR_XPU_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.XPU)


def _solve_triangular_vendor(A, B, *, upper, left=True, unitriangular=False, out=None):
    if out is None:
        # Functional variant: allocate the result in Python and go straight to
        # the vendor `.out` kernel. Redispatching `.default` instead would land
        # in the ATen composite (empty_like + full-dispatch `.out` call), whose
        # internal `.out` re-enters the gems CUDA-key override wrapper (an
        # extra Python round trip per call) before reaching the vendor kernel.
        result = torch.empty_like(B)
        return torch.ops.aten.linalg_solve_triangular.out.redispatch(
            _VENDOR_XPU_KEYSET,
            A,
            B,
            upper=upper,
            left=left,
            unitriangular=unitriangular,
            out=result,
        )
    return torch.ops.aten.linalg_solve_triangular.out.redispatch(
        _VENDOR_XPU_KEYSET,
        A,
        B,
        upper=upper,
        left=left,
        unitriangular=unitriangular,
        out=out,
    )


def linalg_solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_SOLVE_TRIANGULAR")
    return _solve_triangular_vendor(
        A, B, upper=upper, left=left, unitriangular=unitriangular, out=out
    )


def linalg_solve_triangular_out(
    A, B, *, upper, left=True, unitriangular=False, out=None
):
    logger.debug("GEMS_KUNLUNXIN LINALG_SOLVE_TRIANGULAR_OUT")
    return _solve_triangular_vendor(
        A, B, upper=upper, left=left, unitriangular=unitriangular, out=out
    )
