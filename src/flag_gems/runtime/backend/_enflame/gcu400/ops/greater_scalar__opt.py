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

"""greater_scalar, GCU400-optimized (sibling of ``greater_equal_scalar``).

Replaces the earlier ``max_grid_size=(1024, 1, 1)`` codegen override (still
grid-capped at 1024 CTAs and still paying the fp32 up-cast compare: bench core
baseline 0.604x arithmean, 0.30-0.53x beyond 2**24 elements) with the flat 1D
scalar kernel from ``.greater``: native-dtype compare with the Python scalar
cast to ``A.dtype`` (torch TensorIterator semantics), per-dtype BLOCK tiers
probe-validated on the ``ge`` sibling. Falls back to the public
``gt_func_scalar`` pointwise kernel for non-float / non-contiguous / huge
inputs.
"""

import logging

from flag_gems.ops.gt import gt_func_scalar

from .greater import _gt_scalar_fast

logger = logging.getLogger(__name__)


def greater_scalar(A, B):
    logger.debug("GEMS_ENFLAME GREATER_SCALAR")
    out = _gt_scalar_fast(A, B)
    if out is not None:
        return out
    return gt_func_scalar(A, B)
