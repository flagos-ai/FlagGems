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

from ..utils.tle_copy import tle_copy

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def unfold_copy(input: torch.Tensor, dimension: int, size: int, step: int):
    """aten::unfold_copy: materialize the sliding-window view of `input`.

    `input.unfold(...)` is exactly the layout unfold_copy has to produce, so the
    operator is just a copy of that strided view: a TMA tile when the view
    collapses to at most 2D (1D input, or `input.shape[d] == L * step`), and an
    SDNN row transfer otherwise -- the windows are rows of `size` contiguous
    elements, `step` apart. Overlapping windows are fine either way, the shared
    elements are simply read more than once.

    The fallback here is a last resort rather than a faster equivalent: the
    generic kernel gets overlapping windows wrong (`(4, 8)` unfolded by
    `size=3, step=1` misses 12 of 72 elements).
    """
    logger.debug("GEMS_KUNLUNXIN UNFOLD_COPY")

    if step <= 0:
        raise ValueError("step must be > 0")

    if input.ndim > 0:
        d = dimension % input.ndim
        if size <= input.shape[d]:
            view = input.unfold(d, size, step)
            if view.numel() > 0:
                out = torch.empty(view.shape, dtype=input.dtype, device=input.device)
                if tle_copy(view, out):
                    return out

    from flag_gems.ops.unfold_copy import unfold_copy as generic_unfold_copy

    return generic_unfold_copy(input, dimension, size, step)
