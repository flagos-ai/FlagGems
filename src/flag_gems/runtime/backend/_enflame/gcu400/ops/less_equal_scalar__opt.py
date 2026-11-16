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

"""less_equal_scalar candidate v1: override CodeGenConfig.max_grid_size per operator.

Following the zero-branch template validated by item 11's floor: no hand-written kernel, only change ENFLAME's default.
`max_grid_size=(12,1,1)` to `(1024,1,1)` and regenerate the pointwise_dynamic
wrapper. Codegen's internal `num_ctas=min(grid,num_tiles)` keeps the small-shape call chain
identical to the common implementation with no microsecond regression, while large shapes gain grid parallelism.

Keep production's original `is_tensor=[True, False]` and `ALWAYS_BOOL` promotion semantics
(bool output, scalar B compared with x in fp32). Global CODEGEN_COFIGS is not modified.
"""

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config

_BASE = get_codegen_config()
_OPT_CONFIG = CodeGenConfig(
    max_tile_size=_BASE.max_tile_size,
    max_grid_size=(1024, 1, 1),
    max_num_warps_per_cta=_BASE.max_num_warps_per_cta,
    prefer_block_pointer=_BASE.prefer_block_pointer,
    prefer_1d_tile=_BASE.prefer_1d_tile,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=_OPT_CONFIG,
)
@triton.jit
def _less_equal_func_scalar(x, y):
    return x.to(tl.float32) <= y


def less_equal_scalar(A, B):
    return _less_equal_func_scalar(A, B)
