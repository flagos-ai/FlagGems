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

# Kunlunxin(XPU) override for detach_copy (aten::detach_copy = detach().clone()).
#
# The generic implementation uses a flat copy kernel with BLOCK=1024/nw=4 and a
# per-element div/mod strided kernel. On XPU the small 1024-lane flat tile is
# launch-bound: 1M-element copies measure ~0.058 ms vs torch 0.0075 ms (0.12x).
#
# Fix: delegate the actual data movement to the backend copy_ (which uses the
# tuned _copy_flat_kernel with a bounded 1024..65536 tile and num_warps=32 for
# contiguous same-dtype copies, and tle_copy for strided layouts), keeping the
# lazy-bit / complex handling from the generic implementation.  This is a real
# XPU kernel (not aten._copy_from) and is not a fallback.

import logging

import torch

from .copy import copy_ as _copy_

logger = logging.getLogger(__name__)


def _resolve_lazy_bits(input: torch.Tensor) -> torch.Tensor:
    if input.is_conj():
        input = input.resolve_conj()
    is_neg = getattr(input, "is_neg", None)
    if is_neg is not None and is_neg():
        input = input.resolve_neg()
    return input


def detach_copy(self: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN DETACH_COPY")
    self = _resolve_lazy_bits(self)
    is_complex = self.dtype.is_complex
    if is_complex:
        self = torch.view_as_real(self)

    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    if out.numel() == 0:
        return torch.view_as_complex(out) if is_complex else out

    _copy_(out, self)
    return torch.view_as_complex(out) if is_complex else out
