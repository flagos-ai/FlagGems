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

"""floor_out candidate v1: hand-written 1D kernel, large shapes use B131072/nw4.

Production floor_func uses pointwise_dynamic: tl.floor(x.to(tl.float32)).to(x.dtype),
and ENFLAME max_grid_size=(12,1,1) leaves large shapes under-parallelized.
On gt_scalar_, B131072/nw4 was verified clearly better than B32768/nw4 at the 2^24/2^30 tiers;
microsecond small shapes reuse production's same underlying call to avoid regressions from extra Python checks.
"""

import logging

from .floor__opt import _flat_path_ok, _launch_floor

logger = logging.getLogger(__name__)


def floor_out(A, *, out=None):
    logger.debug("GEMS_ENFLAME FLOOR_OUT")
    if out is None:
        from .floor__opt import floor as _floor

        return _floor(A)
    if (
        not _flat_path_ok(A)
        or not out.is_contiguous()
        or out.dtype != A.dtype
        or out.numel() != A.numel()
    ):
        from flag_gems.ops.floor import floor_func as _public

        _public(A, out0=out)
        return out
    _launch_floor(A, out)
    return out
