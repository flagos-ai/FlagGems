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

"""clip candidate v1: override CodeGenConfig.max_grid_size uniformly for the three jit objects.

Production `clip()` dispatches by whether mini/maxi are None to three different
pointwise_dynamic objects (clip_func / clip_func_min / clip_func_max),
all limited by ENFLAME's default `max_grid_size=(12,1,1)`.
This version keeps the host branch structure word-for-word isomorphic to production and only swaps
the three jit objects' config to `(1024,1,1)`, adding no extra Python checks.

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


@pointwise_dynamic(
    is_tensor=[True, False, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=_OPT_CONFIG,
)
@triton.jit
def clip_func(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=_OPT_CONFIG,
)
@triton.jit
def clip_func_min(x, mini):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=_OPT_CONFIG,
)
@triton.jit
def clip_func_max(x, maxi):
    return tl.minimum(maxi, x.to(tl.float32))


def clip(A, mini=None, maxi=None):
    logger.debug("GEMS_ENFLAME_GCU400 CLIP")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clip_func_max(A, maxi)
    elif maxi is None:
        return clip_func_min(A, mini)
    else:
        return clip_func(A, mini, maxi)
