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
import operator

import torch

logger = logging.getLogger(__name__)


def tensor_split(input, indices_or_sections, dim=0):
    """Return storage-sharing splits using only tensor metadata.

    A copy kernel is incorrect here: distributed code writes into the splits
    and expects those writes to update the original communication buffer.
    """
    logger.debug("GEMS ASCEND TENSOR_SPLIT")
    if input.ndim == 0:
        raise RuntimeError(
            "tensor_split expected at least a 1-dimensional tensor, "
            "but got a tensor with 0 dims"
        )
    dim = operator.index(dim)
    if dim < -input.ndim or dim >= input.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-input.ndim}, {input.ndim - 1}], but got {dim})"
        )
    dim %= input.ndim
    dim_size = input.shape[dim]

    if isinstance(indices_or_sections, torch.Tensor):
        # These are host metadata, never move an accelerator tensor to CPU.
        if indices_or_sections.device.type != "cpu":
            raise RuntimeError(
                "tensor_split expected tensor_indices_or_sections to be on cpu"
            )
        if indices_or_sections.dtype != torch.int64:
            raise RuntimeError(
                "tensor_split expected tensor_indices_or_sections "
                "to have dtype of long"
            )
        if indices_or_sections.ndim > 1:
            raise RuntimeError(
                "tensor_split expected tensor_indices_or_sections "
                "to be a zero-dimensional or one-dimensional tensor"
            )
        indices_or_sections = indices_or_sections.tolist()

    if isinstance(indices_or_sections, (tuple, list)):
        # Normalize each boundary independently. Native slicing permits
        # nonmonotonic boundaries, producing empty and overlapping views.
        ends = []
        for index in indices_or_sections:
            index = operator.index(index)
            if index < 0:
                index += dim_size
            ends.append(min(max(index, 0), dim_size))
        ends.append(dim_size)
    else:
        sections = operator.index(indices_or_sections)
        if sections <= 0:
            raise RuntimeError(
                f"number of sections must be larger than 0, got {sections}"
            )
        quotient, remainder = divmod(dim_size, sections)
        ends = [(i + 1) * quotient + min(i + 1, remainder) for i in range(sections)]

    outputs = []
    start = 0
    strides = input.stride()
    base_offset = input.storage_offset()
    for end in ends:
        shape = list(input.shape)
        shape[dim] = max(end - start, 0)
        outputs.append(
            torch.as_strided(input, shape, strides, base_offset + start * strides[dim])
        )
        start = end
    return outputs
