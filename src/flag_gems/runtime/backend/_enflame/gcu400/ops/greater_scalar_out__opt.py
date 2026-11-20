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

"""greater_scalar_out, GCU400-optimized (sibling of ``greater_equal``).

Replaces the earlier ``max_grid_size=(1024, 1, 1)`` codegen override (bench
core baseline 0.595x arithmean) with the flat 1D scalar kernel from
``.greater`` writing straight into the caller's bool ``out``. Falls back to
the public ``gt_func_scalar`` pointwise kernel (``out0=out``) for non-bool /
non-contiguous / mismatched-shape / non-float inputs.
"""

import logging

import torch

from flag_gems.ops.gt import gt_func_scalar

from .greater import _gt_scalar_fast

logger = logging.getLogger(__name__)


def greater_scalar_out(A, B, *, out=None):
    logger.debug("GEMS_ENFLAME GREATER_SCALAR_OUT")
    if out is None:
        out = _gt_scalar_fast(A, B)
        if out is not None:
            return out
        return gt_func_scalar(A, B)
    if out.dtype == torch.bool and out.shape == A.shape and out.is_contiguous():
        result = _gt_scalar_fast(A, B, out)
        if result is not None:
            return out
    gt_func_scalar(A, B, out0=out)
    return out
