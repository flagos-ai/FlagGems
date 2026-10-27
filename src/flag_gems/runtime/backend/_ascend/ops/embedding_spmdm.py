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

"""Fixed-length embedding lookup followed by an optional weighted sum."""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)


def _validate_embedding_spmdm_inputs(
    embedding_table: torch.Tensor,
    indices: torch.Tensor,
    per_sample_weights: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if not isinstance(embedding_table, torch.Tensor):
        raise TypeError("embedding_table must be a torch.Tensor")
    if not isinstance(indices, torch.Tensor):
        raise TypeError("indices must be a torch.Tensor")
    if embedding_table.ndim != 2:
        raise ValueError("embedding_table must be 2-dimensional with shape [V, D]")
    if indices.ndim != 2:
        raise ValueError("indices must be 2-dimensional with shape [B, bag_size]")
    if embedding_table.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "embedding_table supports only torch.float16, torch.bfloat16, "
            f"and torch.float32, got {embedding_table.dtype}"
        )
    if indices.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise TypeError(
            f"indices must have dtype torch.int32 or torch.int64, got {indices.dtype}"
        )
    if embedding_table.device != indices.device:
        raise ValueError("embedding_table and indices must be on the same device")

    num_embeddings, embedding_dim = embedding_table.shape
    batch_size, bag_size = indices.shape
    if min(num_embeddings, embedding_dim, batch_size, bag_size) <= 0:
        raise ValueError("V, D, B, and bag_size must all be positive")

    if per_sample_weights is not None:
        if not isinstance(per_sample_weights, torch.Tensor):
            raise TypeError("per_sample_weights must be a torch.Tensor or None")
        if per_sample_weights.device != embedding_table.device:
            raise ValueError("per_sample_weights must be on the same device")
        if per_sample_weights.dtype != embedding_table.dtype:
            raise TypeError(
                "per_sample_weights must have the same dtype as embedding_table"
            )
        if tuple(per_sample_weights.shape) != tuple(indices.shape):
            raise ValueError(
                "per_sample_weights must have exactly the same shape as indices"
            )

    return num_embeddings, embedding_dim, batch_size, bag_size


@libentry()
@triton.jit
def embedding_spmdm_fwd_kernel(
    embedding_table,
    indices,
    per_sample_weights,
    output,
    num_embeddings,
    embedding_dim,
    bag_size,
    stride_table_row,
    stride_table_dim,
    stride_indices_batch,
    stride_indices_bag,
    stride_weights_batch,
    stride_weights_bag,
    stride_output_batch,
    stride_output_dim,
    HAS_WEIGHTS: tl.constexpr,
    NUM_BAG_BLOCKS: tl.constexpr,
    BLOCK_BAG: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bag = tl.program_id(0)
    block_d = tl.program_id(1)
    offsets_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offsets_d < embedding_dim
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    positions_in_block = tl.arange(0, BLOCK_BAG)
    for bag_block in range(NUM_BAG_BLOCKS):
        positions = bag_block * BLOCK_BAG + positions_in_block
        mask_positions = positions < bag_size
        index = tl.load(
            indices + bag * stride_indices_batch + positions * stride_indices_bag,
            mask=mask_positions,
            other=0,
        ).to(tl.int32)
        valid_index = mask_positions & (index >= 0) & (index < num_embeddings)
        values = tl.load(
            embedding_table
            + index[:, None] * stride_table_row
            + offsets_d[None, :] * stride_table_dim,
            mask=valid_index[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        if HAS_WEIGHTS:
            scale = tl.load(
                per_sample_weights
                + bag * stride_weights_batch
                + positions * stride_weights_bag,
                mask=mask_positions,
                other=0.0,
            ).to(tl.float32)
            values *= scale[:, None]
        accumulator += tl.sum(values, axis=0)

    tl.store(
        output + bag * stride_output_batch + offsets_d * stride_output_dim,
        accumulator,
        mask=mask_d,
    )


def embedding_spmdm(
    embedding_table: torch.Tensor,
    indices: torch.Tensor,
    per_sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Look up fixed-length bags and reduce each bag with a weighted sum.

    ``per_sample_weights=None`` has the same meaning as an all-ones weight
    tensor, without allocating or reading such a tensor.
    """
    logger.debug("GEMS_ASCEND EMBEDDING_SPMDM")
    num_embeddings, embedding_dim, batch_size, bag_size = (
        _validate_embedding_spmdm_inputs(embedding_table, indices, per_sample_weights)
    )
    output = torch.empty(
        (batch_size, embedding_dim),
        dtype=embedding_table.dtype,
        device=embedding_table.device,
    )
    block_d = min(256, triton.next_power_of_2(embedding_dim))
    block_bag = 16
    num_bag_blocks = triton.cdiv(bag_size, block_bag)
    grid = (batch_size, triton.cdiv(embedding_dim, block_d))

    weights_arg = (
        per_sample_weights if per_sample_weights is not None else embedding_table
    )
    weights_strides = (
        per_sample_weights.stride() if per_sample_weights is not None else (0, 0)
    )
    with torch_device_fn.device(embedding_table.device):
        embedding_spmdm_fwd_kernel[grid](
            embedding_table,
            indices,
            weights_arg,
            output,
            num_embeddings,
            embedding_dim,
            bag_size,
            *embedding_table.stride(),
            *indices.stride(),
            *weights_strides,
            *output.stride(),
            HAS_WEIGHTS=per_sample_weights is not None,
            NUM_BAG_BLOCKS=num_bag_blocks,
            BLOCK_BAG=block_bag,
            BLOCK_D=block_d,
        )
    return output


__all__ = ["embedding_spmdm"]
