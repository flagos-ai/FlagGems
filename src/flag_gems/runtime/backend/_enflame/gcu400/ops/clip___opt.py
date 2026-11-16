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

"""clip_ candidate v1: reuse the three jit objects from clip__opt that already override config per operator.

Production `clip_(A, mini, maxi)` and `clip()` share the `clip_func` /
`clip_func_min` / `clip_func_max` pointwise_dynamic objects,
the only difference being `out0=A` for in-place write-back. Item 16 already overrode these three
objects' `max_grid_size=(12,1,1)->(1024,1,1)` per operator and validated it
(clip production-only 3 rounds PASS 3.178x); this item imports and reuses them directly instead of regenerating jit
objects.

The host branch is word-for-word isomorphic to production (zero extra Python checks), preserving ValueError semantics;
`out0=A` guarantees the in-place contract (the return value is A itself).
"""

import logging

from flag_gems.runtime.backend._enflame.gcu400.ops.clip__opt import (
    clip_func,
    clip_func_max,
    clip_func_min,
)

logger = logging.getLogger(__name__)


__all__ = ["clip_"]


def clip_(A, mini=None, maxi=None):
    logger.debug("GEMS_ENFLAME_GCU400 CLIP_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clip_func_max(A, maxi, out0=A)
    elif maxi is None:
        return clip_func_min(A, mini, out0=A)
    else:
        return clip_func(A, mini, maxi, out0=A)
