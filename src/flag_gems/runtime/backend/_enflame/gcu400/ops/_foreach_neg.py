# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400-specific ``_foreach_neg`` (out-of-place): delegates to the shared unary executor.

Older versions used a standalone int64-metadata single-launch kernel to bypass the shared
executor's grid.y hardware cap, but the wrapper rebuilt metadata on the host every call (3xNT
CPU tensor-scalar writes plus one pageable H2D, family-measured ~136us, 177us per item in the
local probe) and then went through an uncached Triton dispatch; the pageable H2D serializing
with the stream leaks host tax into the official bench kernel timing window: all of
256^2/512^2/1024^2 x 16 ran ~115-128us, weakly shape-correlated (2026-11-16 local baseline
0.42-0.70x, baseline archive 0.47-0.74x).

The lean single-launch route in ``_foreach_unary.py`` here (host ~8us for same-dtype contiguous
float lists, ``_device_meta`` memoized by pointer list plus ``_launch_executor_kernel`` launch
replay, see the 2026-11-16 24-variant campaign) already scales BLOCK by powers of two to bring
grid.y back under 255; the neg row fn reuses the single-tensor ``neg_func`` (``-x``, memory
floor, no extern), with the in-place twin ``_foreach_neg_`` as precedent (log2/lgamma
out-of-place delegate to the same family), so out-of-place delegates to that executor.

Non-float, non-contiguous, mixed dtype/device and complex inputs fall back to ``apply_unary``'s
shared grouped path; dtype policy (neg keeps the input dtype, rejects bool) and errors are unchanged.
"""

import logging

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)


def _foreach_neg(self):
    logger.debug("GEMS_GCU400 FOREACH_NEG")
    return apply_unary("neg", self, inplace=False)
