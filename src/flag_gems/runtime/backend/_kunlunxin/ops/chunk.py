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

import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def chunk(A: torch.Tensor, chunks: int, dim: int = 0) -> List[torch.Tensor]:
    r"""Split a tensor into a specific number of chunks.

    The last chunk will be smaller if the tensor size along the given
    dimension is not divisible by :attr:`chunks`. If :attr:`chunks` exceeds
    the size along :attr:`dim`, fewer than :attr:`chunks` tensors are
    returned (matching ``torch.chunk``).

    Args:
        A (torch.Tensor): Input tensor.
        chunks (int): Number of chunks to produce.
        dim (int): Dimension along which to split the tensor.

    Returns:
        List[torch.Tensor]: List of tensor chunks (views of the original).
    """
    logger.debug("GEMS_KUNLUNXIN CHUNK")

    if chunks <= 0:
        raise RuntimeError(
            f"chunk expects `chunks` to be greater than 0, got: {chunks}"
        )

    # Handle negative dim
    if dim < 0:
        dim = dim + A.ndim

    shape = A.shape
    dim_size = shape[dim]
    chunk_size = (dim_size + chunks - 1) // chunks
    stride = A.stride()
    storage_offset = A.storage_offset()
    dim_stride = stride[dim]

    # Create list to hold chunks
    result = []

    for i in range(chunks):
        start = i * chunk_size

        # Stop if this chunk would be empty
        if start >= dim_size:
            break

        end = min(start + chunk_size, dim_size)
        size = list(shape)
        size[dim] = end - start
        result.append(
            torch.as_strided(A, size, stride, storage_offset + start * dim_stride)
        )

    return result
