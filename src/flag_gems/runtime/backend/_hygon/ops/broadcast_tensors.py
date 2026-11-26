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

# This vendor wrapper only exists to avoid the data-copy kernel the general
# layer launches; shape resolution is not vendor-specific. Reuse the general
# layer's implementation instead of keeping a forked copy -- the one previously
# inlined here was taken from the pre-#6024 version and still used ``max()``,
# so it resolved (0, 8) + (1, 8) to (1, 8) and ``expand`` raised on the
# zero-size dim.
from flag_gems.ops.broadcast_tensors import (
    _compute_broadcast_shape as _general_compute_broadcast_shape,
)

logger = logging.getLogger(__name__)


def _compute_broadcast_shape(tensors):
    """Compute the broadcast shape (delegates to the general layer)."""
    return _general_compute_broadcast_shape(tensors)


def broadcast_tensors(*tensors):
    """Broadcasts the given tensors according to broadcasting semantics.

    ``torch.broadcast_tensors`` is a pure view/meta operation: it resolves to
    stride-0 broadcasted dims and copies no data, so torch eager returns in
    near-zero time. The general layer (``src/flag_gems/ops/broadcast_tensors.py``)
    calls ``.contiguous()`` on every result, which materializes a full data-copy
    kernel for each broadcasted tensor and regresses badly (mean speedup ~0.001
    -- 0.028, worst on large shapes). The only specialization that avoids the
    regression is one that launches no kernel at all and returns strided views
    via ``expand``.
    """
    logger.debug("GEMS_HYGON BROADCAST_TENSORS")

    # Handle case where tensors are passed as a single list/tuple.
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])

    if not tensors:
        return []

    broadcast_shape = _compute_broadcast_shape(tensors)

    # ``expand`` returns a view (stride-0 on broadcasted dims) without copying
    # data or launching a kernel, matching torch's own meta implementation.
    return [tensor.expand(broadcast_shape) for tensor in tensors]
