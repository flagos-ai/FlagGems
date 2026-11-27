# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400-specific ``_foreach_log2`` (out-of-place): delegates to the shared unary executor.

Older versions used a standalone int64-metadata single-launch kernel to bypass the shared
executor's grid.y hardware cap, but the wrapper rebuilt metadata on the host every call (3xNT
CPU tensor-scalar writes plus one pageable H2D, measured ~136us) and then went through an
uncached Triton dispatch (~15us): in official bench kernel mode every 256^2/512^2/1024^2 x 16
tier ran ~105-213us with weak shape correlation (2026-11-16 local baseline 0.64-0.92x). The
lean single-launch route in ``_foreach_unary.py`` here (host ~8us for same-dtype contiguous
float lists, ``_device_meta`` memoized by pointer list plus launch replay, see the 2026-11-16
24-variant campaign) already scales BLOCK by powers of two to bring grid.y under 255, and the
log2 row fn uses the mathematically equivalent ``tl.log(x)*log2(e)`` (log costs 8-17% less per
call, same precedent as ``_foreach_log2_``), so out-of-place also delegates to that executor.

Non-float, non-contiguous, mixed dtype/device and complex inputs fall back to ``apply_unary``'s
shared grouped path; dtype promotion (integer/bool output float32) and error semantics are unchanged.
"""

import logging

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)


def _foreach_log2(self):
    logger.debug("GEMS_GCU400 FOREACH_LOG2")
    return apply_unary("log2", self, inplace=False)
