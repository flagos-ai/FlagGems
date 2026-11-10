# Copyright 2026, The FlagOS Contributors.
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

from flag_gems.runtime.backend._kunlunxin.ops.linalg_det import _linalg_det_impl

logger = logging.getLogger(__name__)


def det(A):
    logger.debug("GEMS_KUNLUNXIN DET")
    # aten::det is numerically equivalent to aten::linalg_det; both only
    # support float32/float64 (LU decomposition). The generic ops/det.py routes
    # through the generic linalg_det LU kernels (register/panel/blocked), which
    # hit OutOfResources on this backend; this override instead reuses the
    # Kunlunxin linalg_det elimination path that is already bit-exact and
    # launches once per step to avoid the in-kernel store->load reorder bug.
    if A.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"det only supports float32 and float64, got {A.dtype}")
    return _linalg_det_impl(A)
