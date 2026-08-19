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
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _tile_copy_func(src):
    return src


def tile(inp: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS_HYGON TILE")

    if isinstance(dims, int):
        dims = (dims,)
    dims = list(dims)

    in_rank = inp.dim()
    dims_rank = len(dims)
    in_shape = list(inp.shape)

    # Broadcast the smaller of the two shapes by left-padding with ones, matching
    # torch.tile semantics.
    if dims_rank < in_rank:
        dims = [1] * (in_rank - dims_rank) + dims
    elif dims_rank > in_rank:
        in_shape = [1] * (dims_rank - in_rank) + in_shape

    rank = len(dims)

    out_shape = []
    is_empty = False
    for i in range(rank):
        assert (
            dims[i] >= 0
        ), "the number of repetitions per dimension out of range (expected to >= 0) but got {}".format(
            dims[i]
        )
        if dims[i] == 0:
            is_empty = True
        out_shape.append(in_shape[i] * dims[i])

    inp = inp.reshape(in_shape)

    if is_empty:
        return torch.empty(out_shape, device=inp.device, dtype=inp.dtype)

    if rank == 0:
        return inp.clone()

    # Realize tile as an interleave -> expand -> contiguous copy -> reshape. Each
    # dim i is repeated dims[i] times by inserting a broadcast axis in front of it:
    #   [d0, d1, ...] -> [1, d0, 1, d1, ...] -> expand [r0, d0, r1, d1, ...]
    # then a single contiguous copy followed by a reshape yields the tiled tensor.
    # The copy is driven by the optimized pointwise_dynamic kernel, which handles
    # the stride-0 broadcast reads efficiently instead of reconstructing a
    # multi-index per element.
    interleaved_shape = []
    expand_shape = []
    for i in range(rank):
        interleaved_shape.append(1)
        interleaved_shape.append(in_shape[i])
        expand_shape.append(dims[i])
        expand_shape.append(in_shape[i])

    view = inp.reshape(interleaved_shape).expand(expand_shape)
    out = _tile_copy_func(view)
    return out.reshape(out_shape)
