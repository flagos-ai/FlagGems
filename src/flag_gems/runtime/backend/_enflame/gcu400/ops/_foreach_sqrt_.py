# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400-specific ``_foreach_sqrt_`` (in-place): delegates to the shared unary executor.

Older versions used a per-tensor 1-D kernel to bypass the shared executor's grid.y cap
(no 256^2/512^2/1024^2 x 16 tier exceeds grid.y, but each tensor costs ~10us of host
``@libentry`` dispatch), leaving candidate latency flat at ~155us regardless of shape
(256^2 0.36x, 512^2 0.43x, 1024^2 0.64x, reproduced 2026-11-16). The lean single-launch
route in ``_foreach_unary.py`` here scales BLOCK by powers of two to bring grid.y back under
255 and finishes all tensors in one launch (~8us host, see the 2026-11-16 24-variant
campaign), so the sqrt row delegates to that executor; the fp32 roundtrip semantics of
``sqrt_func`` in the table are bit-identical to the historical kernel's ``FP32_ROUNDTRIP`` path.

Non-float, non-contiguous, mixed dtype/device and complex inputs fall back to ``apply_unary``'s
shared grouped path; dtype promotion and error semantics (int64 in-place raising
RuntimeError, etc.) are unchanged.
"""

import logging

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)


def _foreach_sqrt_(self):
    logger.debug("GEMS_GCU400 FOREACH_SQRT_")
    return apply_unary("sqrt", self, inplace=True)
