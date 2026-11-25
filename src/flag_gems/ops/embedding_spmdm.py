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

"""Fixed-length embedding lookup followed by an optional weighted sum.

This module implements ``embedding_spmdm`` for generic GPU backends.  Given
a dense embedding table ``[V, D]`` and a 2D ``[B, S]`` integer index matrix,
each row of the index matrix looks up ``S`` rows of the table which are
(optionally) weighted and summed into a single output row of shape ``[B, D]``.

The semantics are identical to::

    torch.nn.functional.embedding_bag(
        indices, weight, offsets=None, mode="sum",
        per_sample_weights=per_sample_weights,
    )

in the 2D-index case.  Passing ``per_sample_weights=None`` has the same
meaning as an all-ones weight tensor, without allocating or reading such a
tensor.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)

_MAX_BLOCK_D = 64
_MIN_BLOCK_D = 16
_MAX_BLOCK_BAG = 64
_MIN_BLOCK_BAG = 16
_NUM_WARPS = 1


def _next_power_of_2(value: int) -> int:
    """``triton.next_power_of_2`` without the runtime dispatch overhead."""
    return 1 << (value - 1).bit_length()


def _spmdm_launch_config(embedding_dim: int, bag_size: int) -> tuple[int, int]:
    """Return ``(BLOCK_D, BLOCK_BAG)`` for the given problem sizes."""
    block_d = min(max(_next_power_of_2(embedding_dim), _MIN_BLOCK_D), _MAX_BLOCK_D)
    block_bag = min(max(_next_power_of_2(bag_size), _MIN_BLOCK_BAG), _MAX_BLOCK_BAG)
    return block_d, block_bag


def _validate_embedding_spmdm_inputs(
    embedding_table: torch.Tensor,
    indices: torch.Tensor,
    per_sample_weights: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    """Validate arguments and return ``(V, D, B, S)``.

    The same validation is shared between the forward kernel and the
    backward kernel so that the two agree on what counts as a valid input.
    """
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
    HAS_WEIGHTS: tl.constexpr,
    BLOCK_BAG: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Single-kernel implementation of the forward operator.

    Each program handles one ``(bag, d_block)`` pair: it streams over all
    ``S`` indices of that bag in chunks of ``BLOCK_BAG``, multiplies the
    gathered table rows by their per-sample weights (if present), and
    accumulates them into an fp32 register of width ``BLOCK_D``.

    ``output`` is always a freshly allocated contiguous ``[B, D]`` tensor, so
    its layout is ``(embedding_dim, 1)`` and does not need to be passed in.
    """
    bag = tl.program_id(0)
    block_d = tl.program_id(1)

    offsets_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offsets_d < embedding_dim

    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    positions_in_block = tl.arange(0, BLOCK_BAG)

    for bag_block in range(0, tl.cdiv(bag_size, BLOCK_BAG)):
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
        output + bag * embedding_dim + offsets_d,
        accumulator,
        mask=mask_d,
    )


def embedding_spmdm(
    embedding_table: torch.Tensor,
    indices: torch.Tensor,
    per_sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Look up fixed-length bags and reduce each bag with a weighted sum.

    Args:
        embedding_table: dense weight matrix with shape ``[V, D]``.
        indices: int32/int64 tensor with shape ``[B, S]`` whose entries
            index into the first dimension of ``embedding_table``.
        per_sample_weights: optional tensor with the same shape and dtype
            as ``indices``.  When provided, the gathered row for position
            ``s`` of bag ``b`` is multiplied by ``per_sample_weights[b, s]``
            before being summed.

    Returns:
        Tensor with shape ``[B, D]`` and the same dtype as ``embedding_table``.
    """
    logger.debug("GEMS EMBEDDING_SPMDM")
    num_embeddings, embedding_dim, batch_size, bag_size = (
        _validate_embedding_spmdm_inputs(embedding_table, indices, per_sample_weights)
    )
    output = torch.empty(
        (batch_size, embedding_dim),
        dtype=embedding_table.dtype,
        device=embedding_table.device,
    )
    block_d, block_bag = _spmdm_launch_config(embedding_dim, bag_size)
    grid = (batch_size, -(-embedding_dim // block_d))

    # When no weights are provided, the kernel never reads the weights
    # pointer thanks to ``HAS_WEIGHTS=False``.  Pass any valid device pointer
    # so the call signature stays uniform across the two paths.
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
            HAS_WEIGHTS=per_sample_weights is not None,
            BLOCK_BAG=block_bag,
            BLOCK_D=block_d,
            num_warps=_NUM_WARPS,
        )
    return output


__all__ = ["embedding_spmdm"]
