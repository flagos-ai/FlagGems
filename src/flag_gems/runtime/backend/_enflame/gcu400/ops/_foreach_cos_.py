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
"""GCU400-specific ``_foreach_cos_`` (in-place): delegates to the shared unary executor.

Older versions used a per-tensor 1-D flat kernel to bypass the shared executor's grid.y cap
(no 256^2/512^2/1024^2 x 16 tier exceeds grid.y, but each tensor costs ~10us of host
``@libentry`` dispatch), leaving candidate latency flat at ~172-224us regardless of shape
(256^2 1.27-1.37x, 512^2 2.3-2.5x, 1024^2 5.5-6.8x, reproduced 2026-11-16). The lean
single-launch route in ``_foreach_unary.py`` here scales BLOCK by powers of two to bring
grid.y back under 255 and finishes all tensors in one launch (~8us host, see the 2026-11-16
24-variant campaign; the same table measured 11.93x for out-of-place ``cos`` and 13.05x for
in-place ``sin_``), so this row delegates to that executor. The fp32 promotion semantics of
``cos_func`` in the table match the historical kernel's ``.to(tl.float32)`` branch, so there
is no new math risk.

Non-float, non-contiguous, mixed dtype/device and complex inputs fall back to ``apply_unary``'s
shared grouped path; dtype promotion and error semantics (int64 in-place raising
RuntimeError, etc.) are unchanged.
"""

import logging

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)


def _foreach_cos_(self):
    logger.debug("GEMS_GCU400 _FOREACH_COS_")
    return apply_unary("cos", self, inplace=True)
