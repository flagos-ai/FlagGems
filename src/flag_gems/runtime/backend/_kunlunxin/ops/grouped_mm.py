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

from .mm import mm_out as vendor_mm_out

logger = logging.getLogger(__name__)


def group_mm(A: torch.Tensor, B: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Kunlunxin XPU grouped_mm (`aten::_grouped_mm`).

    The generic implementation (`src/flag_gems/ops/group_gemm.py`) launches one
    persistent kernel with NUM_SMS programs that scans all groups per program;
    XPU probes showed grid scaling is not a lever on this backend, and the
    generic kernel measures at ~0.45-0.78x of native.

    This vendor override splits the variable-length groups into G fixed-shape
    2D GEMMs and reuses the already-tuned vendor mm kernel
    (`_kunlunxin/ops/mm.py:mm`), writing directly into ``C[s:e]`` row slices
    via `mm_out` (no extra copy).  The vendor mm kernel handles ragged tails
    by address wrap-around (`rm % M`) + store mask, so group sizes do not
    need to be multiples of the block size.
    """
    logger.debug("GEMS_KUNLUNXIN GROUP_MM")
    assert A.dim() == 2
    assert B.dim() == 3
    M, K = A.shape
    num_groups, BK, N = B.shape
    assert num_groups == offs.numel()
    C = A.new_empty(M, N)
    # one host sync for all group boundaries, then pure launch loop
    offs_cpu = offs.detach().cpu()
    s = 0
    for g in range(num_groups):
        e = int(offs_cpu[g])
        if e > s:
            vendor_mm_out(A[s:e], B[g], out=C[s:e])
        s = e
    return C