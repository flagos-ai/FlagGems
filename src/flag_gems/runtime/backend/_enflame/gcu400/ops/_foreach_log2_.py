# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400-specific ``_foreach_log2_`` (in-place): delegates to the shared unary executor.

Older versions bypassed the shared executor's grid.y hardware cap with a per-tensor 1-D kernel
(neither 256^2/512^2/1024^2 x 16 tiers exceed grid.y, but each tensor costs ~11us of host
``@libentry`` dispatch), which regressed small and mid shapes across the official bench core
(256^2 0.39x, 512^2 0.53x, candidate latency flat at ~175us regardless of shape).
The lean single-launch route in ``_foreach_unary.py`` in this directory now scales BLOCK by
powers of two to bring grid.y back under 255, finishing all tensors in one launch (~8us host,
see the 2026-11-16 24-variant campaign notes), and the log2 lines delegate to that executor.

Non-float, non-contiguous, mixed dtype/device and complex inputs fall back to ``apply_unary``'s
shared grouped path; dtype promotion and error semantics (int64 in-place raising
RuntimeError, etc.) are unchanged.
"""

import logging

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)


def _foreach_log2_(self):
    logger.debug("GEMS_GCU400 FOREACH_LOG2_")
    return apply_unary("log2", self, inplace=True)
