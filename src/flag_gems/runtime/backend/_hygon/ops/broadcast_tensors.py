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

logger = logging.getLogger(__name__)


def _compute_broadcast_shape(tensors):
    """Compute the broadcast shape from a list of tensors."""
    max_ndim = max(t.ndim for t in tensors)

    padded_shapes = []
    for t in tensors:
        shape = [1] * (max_ndim - t.ndim) + list(t.shape)
        padded_shapes.append(shape)

    broadcast_shape = []
    for i in range(max_ndim):
        dim_size = 1
        for shape in padded_shapes:
            dim_size = max(dim_size, shape[i])
        broadcast_shape.append(dim_size)

    return tuple(broadcast_shape)


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
