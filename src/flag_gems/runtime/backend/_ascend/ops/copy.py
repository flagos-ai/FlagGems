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

import torch

from flag_gems.ops.copy import copy_ as _generic_copy_


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    if dst.numel() == 0 and dst.layout == torch.strided:
        if dst._is_zerotensor():
            raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
        if isinstance(src, torch.Tensor) and src._is_zerotensor():
            return dst.zero_()
        src_shape = src.shape if isinstance(src, torch.Tensor) else ()
        broadcast_shape = torch.broadcast_shapes(dst.shape, src_shape)
        if torch.Size(broadcast_shape) != dst.shape:
            raise RuntimeError(
                f"The broadcast shape {broadcast_shape} does not match "
                f"destination shape {tuple(dst.shape)}"
            )
        return dst
    return _generic_copy_(dst, src, non_blocking=non_blocking)


def copy(template: torch.Tensor, src: torch.Tensor, *, non_blocking=False):
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    return copy_(out, src, non_blocking=bool(non_blocking))
