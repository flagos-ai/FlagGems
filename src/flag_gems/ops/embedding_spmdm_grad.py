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

"""Standalone backward operator for :func:`flag_gems.ops.embedding_spmdm`.

The backward is split into two independent Triton kernels because the
table gradient is a conflicting scatter-add (different bags may collide on
the same embedding row), whereas the per-sample-weight gradient is a
conflict-free dot product over the embedding dimension.
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


def _validate_embedding_spmdm_grad_inputs(
    embedding_table: torch.Tensor,
    indices: torch.Tensor,
    per_sample_weights: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    """Validate backward inputs and return ``(V, D, B, S)``."""
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
def embedding_spmdm_grad_table_kernel(
    grad_output,
    indices,
    per_sample_weights,
    grad_embedding_table,
    num_embeddings,
    embedding_dim,
    stride_grad_batch,
    stride_grad_dim,
    stride_indices_batch,
    stride_indices_bag,
    stride_weights_batch,
    stride_weights_bag,
    stride_grad_table_row,
    stride_grad_table_dim,
    HAS_WEIGHTS: tl.constexpr,
    BAG_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Accumulate the embedding-table gradient via ``tl.atomic_add``.

    Each program handles one ``(bag, d_block)`` pair: for every position
    ``s`` in the bag it loads ``grad_output[bag, d_block]``, optionally
    multiplies it by the per-sample weight, and adds the result into
    ``grad_embedding_table[indices[bag, s], d_block]``.
    """
    bag = tl.program_id(0)
    block_d = tl.program_id(1)

    offsets_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offsets_d < embedding_dim

    values = tl.load(
        grad_output + bag * stride_grad_batch + offsets_d * stride_grad_dim,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)

    for position in tl.range(0, BAG_SIZE):
        index = tl.load(
            indices + bag * stride_indices_batch + position * stride_indices_bag
        ).to(tl.int32)
        valid_index = (index >= 0) & (index < num_embeddings)
        contribution = values
        if HAS_WEIGHTS:
            scale = tl.load(
                per_sample_weights
                + bag * stride_weights_batch
                + position * stride_weights_bag
            ).to(tl.float32)
            contribution = values * scale

        tl.atomic_add(
            grad_embedding_table
            + index * stride_grad_table_row
            + offsets_d * stride_grad_table_dim,
            contribution,
            mask=mask_d & valid_index,
        )


@libentry()
@triton.jit
def embedding_spmdm_grad_weights_kernel(
    grad_output,
    embedding_table,
    indices,
    grad_per_sample_weights,
    num_embeddings,
    embedding_dim,
    bag_size,
    num_samples,
    stride_grad_batch,
    stride_grad_dim,
    stride_table_row,
    stride_table_dim,
    stride_indices_batch,
    stride_indices_bag,
    stride_grad_weights_batch,
    stride_grad_weights_bag,
    NUM_D_BLOCKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute the per-sample-weight gradient.

    Each program handles one sample ``i`` and computes::

            grad_per_sample_weights[i] = <grad_output[bag(i)], table[idx(i)]>

    using a streaming reduction over the embedding dimension.  There are
    no conflicting writes, so this kernel does not need atomics.
    """
    sample = tl.program_id(0)
    bag = sample // bag_size
    position = sample - bag * bag_size
    index = tl.load(
        indices + bag * stride_indices_batch + position * stride_indices_bag
    ).to(tl.int32)
    valid = (sample < num_samples) & (index >= 0) & (index < num_embeddings)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for block in range(NUM_D_BLOCKS):
        offsets_d = block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = (offsets_d < embedding_dim) & valid
        grad = tl.load(
            grad_output + bag * stride_grad_batch + offsets_d * stride_grad_dim,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        table = tl.load(
            embedding_table + index * stride_table_row + offsets_d * stride_table_dim,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        accumulator += grad * table

    result = tl.sum(accumulator, axis=0)
    tl.store(
        grad_per_sample_weights
        + bag * stride_grad_weights_batch
        + position * stride_grad_weights_bag,
        result,
        mask=sample < num_samples,
    )


def _validate_grad_output(
    grad_output: torch.Tensor,
    embedding_table: torch.Tensor,
    batch_size: int,
    embedding_dim: int,
) -> None:
    if not isinstance(grad_output, torch.Tensor):
        raise TypeError("grad_output must be a torch.Tensor")
    if grad_output.ndim != 2 or tuple(grad_output.shape) != (
        batch_size,
        embedding_dim,
    ):
        raise ValueError(f"grad_output must have shape [{batch_size}, {embedding_dim}]")
    if grad_output.device != embedding_table.device:
        raise ValueError("grad_output must be on the same device as embedding_table")
    if grad_output.dtype != embedding_table.dtype:
        raise TypeError("grad_output must have the same dtype as embedding_table")


def embedding_spmdm_grad(
    grad_output: torch.Tensor,
    embedding_table: torch.Tensor,
    indices: torch.Tensor,
    per_sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return gradients for the embedding table and optional sample weights.

    The table gradient is always produced.  The sample-weight gradient is
    produced only when ``per_sample_weights`` was present in the forward
    call.  Internally these are two independent Triton kernels because the
    former is a conflicting scatter-add and the latter is a conflict-free
    dot product.

    Args:
        grad_output: upstream gradient of shape ``[B, D]`` and same dtype
            as ``embedding_table``.
        embedding_table: the same table passed to ``embedding_spmdm``.
        indices: the same ``[B, S]`` index matrix passed to the forward.
        per_sample_weights: the same weight tensor passed to the forward,
            or ``None`` if the forward was unweighted.

    Returns:
        Tuple ``(grad_embedding_table, grad_per_sample_weights)`` where
        ``grad_per_sample_weights`` is ``None`` when ``per_sample_weights``
        itself was ``None``.
    """
    logger.debug("GEMS EMBEDDING_SPMDM_GRAD")
    num_embeddings, embedding_dim, batch_size, bag_size = (
        _validate_embedding_spmdm_grad_inputs(
            embedding_table, indices, per_sample_weights
        )
    )
    _validate_grad_output(grad_output, embedding_table, batch_size, embedding_dim)

    # Triton 3.x does not support low-precision atomic_add on every backend
    # and the ordering of competing writes through atomic_add is not
    # guaranteed to match the reference.  Accumulate all low-precision
    # inputs (fp16, bf16) in fp32 and cast back at the end so that
    # reorderings of the writes do not leak into the final result.
    grad_table_dtype = (
        torch.float32
        if embedding_table.dtype in (torch.float16, torch.bfloat16)
        else embedding_table.dtype
    )
    grad_embedding_table_acc = torch.zeros(
        (num_embeddings, embedding_dim),
        dtype=grad_table_dtype,
        device=embedding_table.device,
    )
    num_samples = batch_size * bag_size
    table_block_d = min(256, triton.next_power_of_2(embedding_dim))
    table_grid = (
        batch_size,
        triton.cdiv(embedding_dim, table_block_d),
    )
    weights_arg = (
        per_sample_weights if per_sample_weights is not None else embedding_table
    )
    weights_strides = (
        per_sample_weights.stride() if per_sample_weights is not None else (0, 0)
    )

    with torch_device_fn.device(embedding_table.device):
        embedding_spmdm_grad_table_kernel[table_grid](
            grad_output,
            indices,
            weights_arg,
            grad_embedding_table_acc,
            num_embeddings,
            embedding_dim,
            *grad_output.stride(),
            *indices.stride(),
            *weights_strides,
            *grad_embedding_table_acc.stride(),
            HAS_WEIGHTS=per_sample_weights is not None,
            BAG_SIZE=bag_size,
            BLOCK_D=table_block_d,
        )

    grad_embedding_table = (
        grad_embedding_table_acc.to(embedding_table.dtype)
        if grad_embedding_table_acc.dtype != embedding_table.dtype
        else grad_embedding_table_acc
    )
    if per_sample_weights is None:
        return grad_embedding_table, None

    grad_per_sample_weights = torch.empty(
        per_sample_weights.shape,
        dtype=per_sample_weights.dtype,
        device=per_sample_weights.device,
    )
    weights_block_d = min(256, triton.next_power_of_2(embedding_dim))
    num_d_blocks = triton.cdiv(embedding_dim, weights_block_d)
    with torch_device_fn.device(embedding_table.device):
        embedding_spmdm_grad_weights_kernel[(num_samples,)](
            grad_output,
            embedding_table,
            indices,
            grad_per_sample_weights,
            num_embeddings,
            embedding_dim,
            bag_size,
            num_samples,
            *grad_output.stride(),
            *embedding_table.stride(),
            *indices.stride(),
            *grad_per_sample_weights.stride(),
            NUM_D_BLOCKS=num_d_blocks,
            BLOCK_D=weights_block_d,
        )
    return grad_embedding_table, grad_per_sample_weights


__all__ = ["embedding_spmdm_grad"]
