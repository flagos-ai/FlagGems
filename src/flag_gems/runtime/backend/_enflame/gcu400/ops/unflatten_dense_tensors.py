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

"""GCU400 specialization of aten::unflatten_dense_tensors.

The operator is pure metadata: every non-empty output is a view of ``flat``
carrying the template's shape and ``flat``'s strides, while a zero-numel
template yields a fresh empty tensor.  No elementwise computation happens, so
the whole cost is host-side view construction, and the GCU400 variant is a
tight loop over the cheapest view constructor measured on this backend:

    torch.as_strided (bound)   2.09 us   <- used
    flat.as_strided            2.34 us
    flat.narrow                3.29 us
    aten::as_strided.default   6.50 us
    aten::view + narrow        8.66 us
    empty().set_(storage)      6.34 us

(The per-call costs are for a 16M-element flat buffer; creating a view through
the PrivateUse1 dispatch is two orders of magnitude more expensive than reading
``numel()`` at 0.18 us, so the loop avoids every avoidable call.)
"""

import logging
from typing import List

import torch
from packaging import version

logger = logging.getLogger(__name__)

# torch < 2.10 collapses a zero-numel output to a 1-D ``(0,)`` tensor while
# torch >= 2.10 keeps the template's shape.  Hoisted to module scope: the
# comparison would otherwise run for every zero-numel template.
_KEEPS_EMPTY_SHAPE = version.parse(torch.__version__.split("+")[0]).release[:2] >= (
    2,
    10,
)


def unflatten_dense_tensors(
    flat: torch.Tensor, tensors: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Return one view of ``flat`` per template in ``tensors``.

    Non-empty outputs alias ``flat``'s storage at a running offset, keep
    ``flat``'s dtype/device and inherit its strides; a zero-numel template
    returns a fresh empty tensor with no base, matching the native op.
    """
    logger.debug("GEMS_ENFLAME UNFLATTEN_DENSE_TENSORS")

    # One attribute read per output would still dominate, so the two pieces of
    # per-call state that never change for ``flat`` are read once and the view
    # constructor plus ``empty`` are bound locally.
    as_strided = torch.as_strided
    one_d = flat.dim() == 1
    stride0 = flat.stride(0) if one_d else 0
    # Cached 1-D stride tuple: ``(stride0,)`` is rebuilt once instead of once
    # per template.
    stride_tuple = (stride0,)
    dim0 = flat.size(0)
    base = flat.storage_offset()

    outputs = []
    append = outputs.append
    offset = 0
    for tensor in tensors:
        shape = tensor.shape
        numel = tensor.numel()
        if numel == 0:
            # dtype/device are only needed on the rare zero-numel branch, so
            # they are read lazily instead of adding two dispatch calls per
            # template.
            empty_shape = shape if _KEEPS_EMPTY_SHAPE else (0,)
            append(torch.empty(empty_shape, dtype=flat.dtype, device=flat.device))
        elif not one_d:
            # Multi-dimensional flat buffers keep the narrow+view path so the
            # raised errors and stride inheritance stay identical to the
            # common implementation.
            append(flat.narrow(0, offset, numel).view(shape))
        else:
            if offset + numel > dim0:
                # Delegate to narrow() so the raised RuntimeError keeps the
                # exact native message instead of a hard-coded copy of it.
                flat.narrow(0, offset, numel)
            if len(shape) == 1:
                append(as_strided(flat, shape, stride_tuple, base + offset * stride0))
            else:
                # Row-major strides for ``shape`` laid out over the flat
                # buffer's element stride.
                strides = []
                acc = stride0
                for dim_size in reversed(shape):
                    strides.append(acc)
                    acc *= dim_size
                strides.reverse()
                append(as_strided(flat, shape, tuple(strides), base + offset * stride0))
        offset += numel
    return outputs
