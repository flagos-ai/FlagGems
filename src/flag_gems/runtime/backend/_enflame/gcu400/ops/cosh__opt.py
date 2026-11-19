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

"""cosh candidate v1: the outplace variant reuses the optimized jit object already delivered in item 29.

Common `cosh = cosh_func(A)` shares the same
`pointwise_dynamic` wrapper object (`INT_TO_FLOAT` promotion) with `cosh_out` / `cosh_`.
`_cosh_func` in `cosh_out__opt.py` delivered in item 29 already overrides
`max_grid_size` per operator from the ENFLAME default `(12,1,1)` to `(1024,1,1)`,
so this item **imports and reuses it directly with zero duplicate jit objects**, and the host is
word-for-word isomorphic to production `cosh`.

No TRACKED file is modified, nor global CODEGEN_COFIGS.
"""

import logging

from flag_gems.runtime.backend._enflame.gcu400.ops.cosh_out__opt import _cosh_func

logger = logging.getLogger(__name__)


__all__ = ["cosh"]


def cosh(A):
    logger.debug("GEMS_ENFLAME_GCU400 COSH")
    return _cosh_func(A)
