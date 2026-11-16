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

"""expm1 candidate v1: override CodeGenConfig.max_grid_size per operator.

Same semantics as item 15's expm1_out, only outplace (out0 omitted).
Following the zero-branch template: no hand-written kernel, only enlarge ENFLAME's default
`max_grid_size=(12,1,1)` to `(1024,1,1)` and regenerate the pointwise_dynamic
wrapper, keeping `INT_TO_FLOAT` promotion unchanged.

The baseline reuses `base_eoN`: collected in the same pytest process and time window as expm1_out,
the log contains expm1's 3 rounds of 15 cases, filterable by op_name.
Global CODEGEN_COFIGS is not modified, satisfying the skill's "pointwise_dynamic must be specialized" requirement.
"""

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config

logger = logging.getLogger(__name__)


_BASE = get_codegen_config()
_OPT_CONFIG = CodeGenConfig(
    max_tile_size=_BASE.max_tile_size,
    max_grid_size=(1024, 1, 1),
    max_num_warps_per_cta=_BASE.max_num_warps_per_cta,
    prefer_block_pointer=_BASE.prefer_block_pointer,
    prefer_1d_tile=_BASE.prefer_1d_tile,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=_OPT_CONFIG)
@triton.jit
def _expm1_func(x):
    return tl.exp(x.to(tl.float32)) - 1.0


def expm1(A):
    logger.debug("GEMS_ENFLAME_GCU400 EXPM1")
    return _expm1_func(A)
