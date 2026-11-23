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

"""expm1_ candidate v1: reuse the expm1__opt jit object that already overrides config per operator.

Production `expm1_(A)` and `expm1(A)` share `expm1_func`, the only difference being
`out0=A` for in-place write-back. `expm1__opt.py` delivered in item 17 already changed
`max_grid_size=(12,1,1)->(1024,1,1)` and kept `INT_TO_FLOAT`
promotion (expm1 production-only 3 rounds PASS 3.220x), so this item imports and reuses it directly,
with no duplicate jit objects and zero Python branches.

`out0=A` guarantees the in-place contract; integral inputs are promoted to floating-point output,
matching the common implementation.
"""

import logging

from flag_gems.runtime.backend._enflame.gcu400.ops.expm1__opt import _expm1_func

logger = logging.getLogger(__name__)


__all__ = ["expm1_"]


def expm1_(A):
    logger.debug("GEMS_ENFLAME_GCU400 EXPM1_")
    return _expm1_func(A, out0=A)
