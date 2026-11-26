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

from .linalg_det import _linalg_det_impl

logger = logging.getLogger(__name__)


def det(A):
    logger.debug("GEMS_KUNLUNXIN DET")
    # aten::det is numerically equivalent to aten::linalg_det; both only
    # support float32/float64 (LU decomposition). Reject other dtypes early.
    if A.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"det only supports float32 and float64, got {A.dtype}")
    return _linalg_det_impl(A)


def _install():
    """Wire the XPU implementation into the ops submodule entrypoint.

    ``tests/test_det.py`` calls the op through the top-level ``flag_gems``
    attribute. ``SpecOpRegistrar`` swaps the top-level name as long as this
    module is imported from ``_kunlunxin.ops``; the explicit setattr mirrors
    ``_ctc_loss._install`` for direct submodule callers.
    """
    import sys

    mod = sys.modules.get("flag_gems.ops")
    if mod is not None:
        mod.det = det


_install()
