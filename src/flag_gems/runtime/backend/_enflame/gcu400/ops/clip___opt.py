"""clip_ candidate v2: reuses clip__opt's flat kernel and ladder (in-place write back).

Production `clip_(A, mini, maxi)` shares the same dispatch object with `clip()`, differing only
in `out0=A` for the in-place write back. This version keeps the flat grid-stride kernel of
clip__opt v2 (2-stream scalar bound, probe: 2^30 fp16 in-place 1345us vs pointwise 3362us vs
native 1409us) and writes straight into A when eligible; non-contiguous / non-float / tensor
bound / launch floor / i32-unsafe cases fall back to pointwise (the (1024,1,1) config override
of candidate v1). The host branch keeps the ValueError semantics.
"""

import logging

from flag_gems.runtime.backend._enflame.gcu400.ops.clip__opt import (
    _flat_eligible,
    _flat_plan,
    _launch_flat,
    clip_func,
    clip_func_max,
    clip_func_min,
)

logger = logging.getLogger(__name__)

__all__ = ["clip_"]


def clip_(A, mini=None, maxi=None):
    logger.debug("GEMS_ENFLAME CLIP_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    if _flat_eligible(A, mini, maxi):
        plan = _flat_plan(A)
        if plan is not None:
            block, num_warps, body = plan
            _launch_flat(A, A, mini, maxi, A.numel(), block, num_warps, body)
            return A
    if mini is None:
        return clip_func_max(A, maxi, out0=A)
    if maxi is None:
        return clip_func_min(A, mini, out0=A)
    return clip_func(A, mini, maxi, out0=A)
