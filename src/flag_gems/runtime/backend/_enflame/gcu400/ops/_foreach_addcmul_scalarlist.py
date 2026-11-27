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
"""GCU400-specific ``_foreach_addcmul_.ScalarList`` entry point.

History: this file once carried its own single-launch fused kernel to avoid the shared
executor's 2-D ``grid=(block, tensor)`` crossing the GCU400 ``grid.y <= 255`` cap. After
``_foreach_binary`` was flattened, all six addcmul overloads share ``_apply_ternary``'s flat
single-launch path (int64 metadata content cache, 1-D ``grid = NT * parts`` always an exact
multiple, 16-bit via fp32 opmath), so this file collapses into a thin delegation: two
isomorphic kernels inevitably drift -- the "grid is not an exact NT multiple so tiles are
dropped" defect that flat-ification introduced in ``_flat_grid`` (11-16) went unnoticed
precisely because two implementations diverged without cross-checking.

dtype policy, error semantics and the fallback for non-contiguous/mixed shapes all live in
``_apply_ternary``: the fast path serves same-dtype contiguous float lists, and everything
else goes to the shared executor.
"""

import logging

from ._foreach_binary import _apply_ternary

logger = logging.getLogger(__name__)


def _foreach_addcmul__ScalarList(self, tensor1, tensor2, value=1):
    """In-place ``self[i] += value[i] * tensor1[i] * tensor2[i]``."""
    logger.debug("GEMS_GCU400 _FOREACH_ADDCMUL__SCALARLIST")
    return _apply_ternary("addcmul", self, tensor1, tensor2, value, True)
