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

"""linalg_matrix_power override for the hygon (HIP) backend.

Hygon runs the generic NV implementation unchanged except that its HIP
devices expose only 64 KB of shared memory per block, so the fp64-stored
TRSM update gemm must use a 2-stage software pipeline instead of the
3-stage default.  The shared hosts in flag_gems.ops.linalg_matrix_power are
hooked to the 2-stage solve below and the generic dispatch is re-exported.
(amd reuses this file verbatim.)
"""

import importlib

import torch

from flag_gems.ops.linalg_matrix_power import linalg_matrix_power  # noqa: E402, F401
from flag_gems.ops.linalg_matrix_power import _trsm_kernel

# NB: bind the *module* explicitly - ``import flag_gems.ops.linalg_matrix_power
# as _generic`` would resolve through the package attribute, which the
# re-exported ``linalg_matrix_power`` function shadows.
_generic = importlib.import_module("flag_gems.ops.linalg_matrix_power")


def _trsm_solve_2d(A_tri, B, upper: bool, unitriangular: bool):
    """Same blocked triangular solve as the generic one, at a 2-stage
    software-pipeline depth (64 KB shared memory, see the module docstring)."""
    n = A_tri.shape[0]
    k = B.shape[1]
    K_SLICE = 8
    BM = 128
    num_kslices = (k + K_SLICE - 1) // K_SLICE
    if unitriangular:
        inv = B  # INV_ptr is only dereferenced when UNIT is false
        unit_flag = True
    else:
        inv = torch.zeros(num_kslices * 32, dtype=A_tri.dtype, device=A_tri.device)
        unit_flag = False
    _trsm_kernel[(num_kslices,)](
        A_tri,
        B,
        inv,
        n,
        k,
        A_tri.stride(0),
        B.stride(0),
        32,
        K_SLICE,
        BM,
        upper,
        unit_flag,
        num_warps=4,
        # fp64 dot operands are staged through shared memory (8 bytes/element);
        # the 3-stage default needs ~73.7 KB, over the 64 KB HIP limit.
        num_stages=2,
    )
    return B


# Hook the shared hosts (linalg_lu_solve / _inverse) so every triangular solve
# on this backend runs at the 2-stage depth.  One backend per process, and the
# generic module is fully imported before this override is applied, so the
# rebind is safe.
_generic._trsm_solve_2d = _trsm_solve_2d
