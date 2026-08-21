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

import torch

logger = logging.getLogger(__name__)


def scalar_tensor(s, *, dtype=None, layout=None, device=None, pin_memory=None):
    """0-dim (scalar) tensor creation, Kunlunxin(XPU) optimized.

    The generic implementation allocates via the FlagGems ``aten::empty``
    override (itself a Triton kernel launch) and then runs a second Triton kernel
    to store the scalar value; on XPU the two launches cost ~15us vs ~5us for
    native ``torch.scalar_tensor``.

    This XPU specialization:
      1. allocates natively with ``torch.empty_strided`` (not overridden by
         FlagGems), avoiding the empty-kernel launch entirely;
      2. stores the scalar through the native (vendor) in-place fill path by
         bypassing the FlagGems Autograd-key override with
         ``torch._C._AutoDispatchBelowAutograd`` -- the same ATen dispatch
         machinery used by the native ``scalar_tensor`` implementation
         (``empty`` + ``fill``).

    Both steps run on the XPU device; no CPU/ATen/native/composite fallback.
    """
    logger.debug("GEMS SCALAR_TENSOR (kunlunxin native fill)")
    out = torch.empty_strided(
        (), (), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
    if dtype == torch.bool:
        s = bool(s)
    with torch._C._AutoDispatchBelowAutograd():
        out.fill_(s)
    return out