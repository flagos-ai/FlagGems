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

"""GCU400-specific ``_foreach_add_.List`` (in-place).

The per-tensor flat kernel this file used to ship -- one launch per list
entry, sixteen ``@libentry`` dispatches per call -- measured a constant
~200us host tax on every shape and dtype, leaving the official benchmark
at 0.30-0.75x against ATen while every other ``.List`` member of the
family already ran on the single-launch paired-list kernel in
``_foreach_binary.py``.  This file now delegates wholesale to that table
module, which keeps its own fallback chain: non-float / non-contiguous /
mixed dtypes / ragged sizes / ``alpha != 1`` all reach the common
executor with ATen's promotion and error wording.

The import is relative on purpose.  The same-named generic module
``flag_gems.ops._foreach_binary`` must not capture it: an absolute import
binds to the generic table and silently bypasses every GCU400 flat path
(stale-binding trap, same reasoning as the sign consolidation).
"""

import logging

from ._foreach_binary import _apply_list as _table_apply_list

logger = logging.getLogger(__name__)


def _foreach_add__List(self, other, *, alpha=1):
    logger.debug("GEMS_ENFLAME_GCU400 _FOREACH_ADD__LIST")
    # inplace wrappers answer None, matching the void result of
    # ``aten::_foreach_add_.List``.
    return _table_apply_list("add", self, other, True, alpha)
