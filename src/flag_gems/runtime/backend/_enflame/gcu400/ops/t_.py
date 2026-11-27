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


def t_(inp: torch.Tensor) -> torch.Tensor:
    """In-place transpose of a tensor with at most 2 dimensions.

    ``t_`` is a pure view operation: it only swaps the shape and stride
    metadata of ``inp`` without moving any data, matching the semantics of
    ``torch.Tensor.t_``. For 0D/1D tensors it is a no-op; for 2D tensors it
    swaps the two dimensions (and their strides) in place, so the result is a
    transposed view over the original storage.
    """
    logger.debug("GEMS_ENFLAME T_")
    ndim = inp.ndim
    if ndim == 2:
        # 2D tensor - swap dims 0/1 metadata in place (no data movement);
        # equivalent to as_strided_ with swapped sizes/strides, exactly
        # like the native aten::t_ operator (which itself is transpose_(0, 1)).
        return inp.transpose_(0, 1)
    if ndim < 2:
        # 0D scalar / 1D tensor - no-op, same as torch.Tensor.t_
        return inp
    raise RuntimeError("t_ expects a tensor with <= 2 dimensions")
