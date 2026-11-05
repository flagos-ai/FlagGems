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

import logging

from .nonzero import nonzero

logger = logging.getLogger(__name__)


def argwhere(input):
    """XPU override of ``flag_gems.argwhere``.

    ``torch.argwhere(x)`` returns exactly ``torch.nonzero(x)`` for ``x.ndim >=
    1`` (an ``(nnz, ndim)`` int64 index tensor), so the proven Kunlunxin
    ``nonzero`` implementation (``tle.raw`` payload / verified Triton kernels)
    is reused as-is.  The generic ``ops/argwhere.py`` three-kernel pipeline
    (count / scan / scatter) miscompiles on this backend: the in-block cumsum
    rank disagrees with the count pass (off-by-one output rows) and the
    unmasked stores overrun the output buffer (illegal memory access).
    """
    logger.debug("GEMS_KUNLUNXIN ARGWHERE")
    return nonzero(input)
