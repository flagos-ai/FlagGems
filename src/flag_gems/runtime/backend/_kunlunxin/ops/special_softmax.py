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

from .softmax import softmax as core_softmax

logger = logging.getLogger(__name__)


def special_softmax(self, dim, dtype=None):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SOFTMAX")

    if dtype is not None:
        # Cast input to the specified dtype for computation
        self = self.to(dtype)

    # The tuned backend `softmax` only handles the reduced dim as the innermost
    # axis (its K > 1 transpose path is broken: the rank-reducing device copy
    # it needs is not available). For a non-last `dim`, move it last with a
    # metadata-only view, run the tuned last-axis path, and move the result
    # back. ``movedim`` is a view (no compute) and the contiguous copies it
    # forces are device-side data movement, never an ATen/CPU fallback.
    if self.ndim == 0:
        return core_softmax(self, dim)

    dim = dim % self.ndim
    if dim == self.ndim - 1:
        return core_softmax(self, dim)

    result = core_softmax(self.movedim(dim, -1), -1)
    return result.movedim(-1, dim).contiguous()
