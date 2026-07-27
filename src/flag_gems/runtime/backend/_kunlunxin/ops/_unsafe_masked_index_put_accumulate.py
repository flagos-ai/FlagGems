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

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["mask_numel"])
def _unsafe_masked_index_put_accumulate_kernel(
    input,
    mask,
    index0,
    index1,
    index2,
    values,
    mask_numel,
    SHAPE0: tl.constexpr,
    SHAPE1: tl.constexpr,
    SHAPE2: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_offset = ext.program_id(0)

    if RANK == 1:
        coordinate0 = output_offset
        coordinate1 = 0
        coordinate2 = 0
        input_offset = coordinate0 * STRIDE0
    elif RANK == 2:
        coordinate0 = output_offset // SHAPE1
        coordinate1 = output_offset % SHAPE1
        coordinate2 = 0
        input_offset = coordinate0 * STRIDE0 + coordinate1 * STRIDE1
    else:
        coordinate0 = output_offset // (SHAPE1 * SHAPE2)
        remainder = output_offset % (SHAPE1 * SHAPE2)
        coordinate1 = remainder // SHAPE2
        coordinate2 = remainder % SHAPE2
        input_offset = (
            coordinate0 * STRIDE0
            + coordinate1 * STRIDE1
            + coordinate2 * STRIDE2
        )

    offsets = tl.arange(0, BLOCK_SIZE)
    active = offsets < mask_numel
    selected = tl.load(mask + offsets, mask=active, other=0) != 0
    selected &= tl.load(index0 + offsets, mask=active, other=0).to(
        tl.int32
    ) == coordinate0
    if RANK >= 2:
        selected &= tl.load(index1 + offsets, mask=active, other=0).to(
            tl.int32
        ) == coordinate1
    if RANK == 3:
        selected &= tl.load(index2 + offsets, mask=active, other=0).to(
            tl.int32
        ) == coordinate2

    update_values = tl.load(values + offsets, mask=active, other=0.0).to(tl.float32)
    update = tl.sum(tl.where(selected, update_values, 0.0), axis=0)
    original = tl.load(input + input_offset).to(tl.float32)
    tl.store(input + input_offset, original + update)


def _unsafe_masked_index_put_accumulate(input, mask, indices, values):
    logger.debug("GEMS_KUNLUNXIN _UNSAFE_MASKED_INDEX_PUT_ACCUMULATE")
    rank = input.ndim
    if rank < 1 or rank > 3 or len(indices) != rank:
        raise RuntimeError(
            "Kunlunxin _unsafe_masked_index_put_accumulate supports ranks 1 to 3"
        )
    if input.numel() == 0 or mask.numel() == 0:
        return input

    mask_contiguous = mask.contiguous()
    values_contiguous = values.contiguous()
    contiguous_indices = [index.contiguous() for index in indices]
    while len(contiguous_indices) < 3:
        contiguous_indices.append(contiguous_indices[0])

    shape = list(input.shape) + [1] * (3 - rank)
    strides = list(input.stride()) + [0] * (3 - rank)
    block_size = triton.next_power_of_2(mask.numel())

    with torch_device_fn.device(input.device):
        _unsafe_masked_index_put_accumulate_kernel[(input.numel(),)](
            input,
            mask_contiguous,
            contiguous_indices[0],
            contiguous_indices[1],
            contiguous_indices[2],
            values_contiguous,
            mask.numel(),
            SHAPE0=shape[0],
            SHAPE1=shape[1],
            SHAPE2=shape[2],
            STRIDE0=strides[0],
            STRIDE1=strides[1],
            STRIDE2=strides[2],
            RANK=rank,
            BLOCK_SIZE=block_size,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    return input
