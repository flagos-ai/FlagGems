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

"""floor candidate v5: override CodeGenConfig.max_grid_size per operator (zero Python branches).

The first 4 versions were "hand-written 1D kernel + small-shape tiering", but the pure production path always
had p10~=0.90 on microsecond small shapes: the override and production path sit at different dispatch levels,
so any extra Python check causes a 0.3~0.9us deviation.

This version directly reuses the pointwise_dynamic codegen path and only enlarges ENFLAME's default
`max_grid_size=(12,1,1)` to `(1024,1,1)`:
- codegen's internal `num_ctas = min(max_grid_size, num_tiles)` keeps small shapes using
  few CTAs, with a call chain fully identical to the common implementation and zero extra cost;
- large shapes go from 12 to 1024 CTAs, removing the grid parallelism bottleneck.
Probe measured 2^30 `16.38ms -> 5.90ms` (2.78x) and 2^28 `4.12 -> 1.51ms` (2.73x),
with `exact=True` for every shape and grid.

Global CODEGEN_COFIGS is not modified; only this operator's generated wrapper is affected, with no side effects on
others.
"""

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config

logger = logging.getLogger(__name__)


_BASE = get_codegen_config()

# 1024 is optimal or near-optimal at 2^24 / 2^28 / 2^30 and far below the grid.x limit of 65535
_OPT_CONFIG = CodeGenConfig(
    max_tile_size=_BASE.max_tile_size,
    max_grid_size=(1024, 1, 1),
    max_num_warps_per_cta=_BASE.max_num_warps_per_cta,
    prefer_block_pointer=_BASE.prefer_block_pointer,
    prefer_1d_tile=_BASE.prefer_1d_tile,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_OPT_CONFIG)
@triton.jit
def _floor_func(x):
    return tl.floor(x.to(tl.float32)).to(x.dtype)


def floor(A):
    logger.debug("GEMS_ENFLAME_GCU400 FLOOR")
    return _floor_func(A)
