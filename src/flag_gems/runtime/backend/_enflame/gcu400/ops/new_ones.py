# Copyright 2026, The FlagOS Contributors.
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
"""GCU400-specific ``new_ones`` host-side dispatch.

The previous local copy emitted the generic ``flag_gems.ops.ones.ones_kernel`` directly
(1-D grid = cdiv(N, 1024)), and for N >= 2^30 grid.x = 2^20 exceeded the backend 65535 cap,
so large benchmark shapes failed with OutOfResources.

``new_ones(self, size)`` is semantically identical to ``ones(size)`` (fill with 1 using self's
dtype/device), so this reuses the already tuned ``ones`` in the same directory
(``ones_flat_kernel``: grid-stride loop, grid capped at 24 schedulable SIPs, EXACT divisibility
specialization, int64/uint64 via an int32 word view plus storage-model probing), which never
crosses the grid.x cap for any N; 64-bit element dtypes such as fp64/complex cannot do reliable
flat element stores on this backend, so following ``new_full`` they use its pointwise scalar
fill fallback. Operator semantics and the function signature are unchanged.
"""

import logging
import math

import torch

from .new_full import new_full
from .ones import ones

logger = logging.getLogger(__name__)

_INT32_MAX = torch.iinfo(torch.int32).max

# Element-wise Triton stores of 64-bit element dtypes lower unreliably on
# this backend (see new_full.py), so fp64/complex stay on new_full's
# pointwise scalar-fill fallback instead of the flat fill kernel.
_FLAT_UNSAFE_DTYPES = (
    torch.float64,
    torch.complex64,
    torch.complex128,
)

# int64/uint64 outputs are filled through an int32 word view: under the
# full-width storage model every element occupies two words, halving the
# int32-safe element domain.
_WORD_FILL_DTYPES = (
    torch.int64,
    torch.uint64,
)


def new_ones(self, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    """Returns a tensor filled with ones with the same dtype and device as self.

    Args:
        self: The tensor to get dtype and device from
        size: Shape of the output tensor
        dtype: The desired dtype of the output tensor. If None, uses self.dtype
        layout: The desired layout of the output tensor
        device: The desired device of the output tensor. If None, uses self.device
        pin_memory: Whether to use pinned memory

    Returns:
        A tensor filled with ones
    """
    logger.debug("GEMS_ENFLAME NEW_ONES")
    if device is None:
        device = self.device
    if dtype is None:
        dtype = self.dtype
    if dtype in _FLAT_UNSAFE_DTYPES:
        return new_full(
            self,
            size,
            1.0,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )
    if isinstance(size, int):
        size = (size,)
    # The flat fill kernel addresses int32 element (int64/uint64: int32
    # word) offsets, so fills past that domain route through new_full's
    # guarded fallback -- the same _INT32_MAX boundary as new_full.py.
    limit = _INT32_MAX >> 1 if dtype in _WORD_FILL_DTYPES else _INT32_MAX
    if math.prod(size) > limit:
        return new_full(
            self,
            size,
            1.0,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )
    return ones(size, dtype=dtype, device=device)
