import logging
from typing import List

import torch
import triton
import triton.language as tl

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@triton.jit
def index_kernel_func(
    input_ptr,
    stride: tl.constexpr,
    index_len,
    index_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    MAX_DATA_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)

    for i in range(0, BLOCK_SIZE):
        offset = pid0 * BLOCK_SIZE + i

        if offset < index_len:
            in_start_index = tl.load(index_ptr + offset) * stride
            out_start_offset = offset * stride
            loop_num = (stride - 1) // MAX_DATA_SIZE + 1

            for loop_idx in range(0, loop_num):
                inner_offset = loop_idx * MAX_DATA_SIZE + tl.arange(0, MAX_DATA_SIZE)
                mask = inner_offset < stride
                cur_value = tl.load(
                    input_ptr + in_start_index + inner_offset, mask=mask
                )
                tl.store(
                    out_ptr + out_start_offset + inner_offset, cur_value, mask=mask
                )


def index_wrapper(input, indices, out):
    input_shape = input.shape
    input_dim = len(input_shape)
    indices_dim = len(indices)

    stride = 1
    for i in range(0, input_dim - indices_dim):
        stride *= input_shape[input_dim - i - 1]

    index_len = indices[0].numel()

    actual_index = indices[0]
    for idx in range(0, indices_dim - 1):
        actual_index = actual_index * input_shape[idx + 1] + indices[idx + 1]

    BLOCK_SIZE = 32
    MAX_DATA_SIZE = 8 * 1024

    grid = lambda meta: (triton.cdiv(index_len, meta["BLOCK_SIZE"]),)

    index_kernel_func[grid](
        input,
        stride,
        index_len,
        actual_index,
        out,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_DATA_SIZE=MAX_DATA_SIZE,
    )


def get_max_rank_shape(indices: List[torch.Tensor]) -> List[int]:
    max_rank = max([len(index.shape) for index in indices])
    shape = [0 for _ in range(max_rank)]
    for i in range(max_rank):
        max_num = 0
        for index in indices:
            axis = len(index.shape) - 1 - i
            if axis >= 0:
                max_num = max(max_num, index.shape[axis])  #
        shape[max_rank - 1 - i] = max_num
    return shape


def broadcast_indices(indices, target_shape):
    for i, index in enumerate(indices):
        if tuple(index.shape) != tuple(target_shape):
            indices[i] = torch.broadcast_to(index, target_shape)


def index(inp, indices):
    logger.debug("GEMS_ASCEND INDEX")
    original_indices = list(indices)  # Save original indices for later checks
    indices = list(indices)

    if not indices:
        raise ValueError("at least one index must be provided")

    indices = [
        index.to(inp.device)
        if index is not None and index.device != inp.device
        else index
        for index in indices
    ]

    # Step 1: Process indices (convert bool/int8 to long, handle None)
    # Following PyTorch meta implementation
    processed_indices = []
    for i, index in enumerate(indices):
        if index is not None:
            # Check dtype
            if index.dtype in [torch.int8, torch.bool]:
                # Convert boolean/int8 mask to long indices
                nonzero = index.nonzero()
                k = len(processed_indices)
                if k + index.ndim > inp.ndim:
                    raise IndexError(
                        f"too many indices for tensor of dimension {inp.ndim}"
                    )
                # Check shape matches
                for j in range(index.ndim):
                    if index.shape[j] != inp.shape[k + j]:
                        raise IndexError(
                            f"The shape of the mask {index.shape} at index {i} "
                            f"does not match the shape of the indexed tensor {inp.shape} at index {k + j}"
                        )
                # Extract indices from nonzero
                for j in range(index.ndim):
                    processed_indices.append(nonzero.select(1, j))
            elif index.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                processed_indices.append(index)
            else:
                raise TypeError(
                    "tensors used as indices must be long, int, byte or bool tensors"
                )
        else:
            processed_indices.append(None)

    indices = processed_indices

    # Check indices count
    if len(indices) > inp.ndim:
        raise IndexError(
            f"too many indices for tensor of dimension {inp.ndim} (got {len(indices)})"
        )

    # Save for later use
    has_any_tensor = any(idx is not None for idx in indices)
    starts_with_none = indices[0] is None if indices else False

    # Step 2: Broadcast indices (only tensor indices, not None)
    tensor_indices = [idx for idx in indices if idx is not None]
    if tensor_indices:
        # Broadcast all tensor indices together
        if len(tensor_indices) > 1:
            tensor_indices = list(torch.broadcast_tensors(*tensor_indices))
        # Update indices list with broadcasted tensors
        tensor_idx = 0
        for i in range(len(indices)):
            if indices[i] is not None:
                indices[i] = tensor_indices[tensor_idx]
                tensor_idx += 1

    # Step 3: Add missing None indices (pad to input.ndim)
    while len(indices) < inp.ndim:
        indices.append(None)

    # Step 4: Check if has contiguous subspace
    # (all non-None tensors are adjacent)
    state = 0
    has_contiguous_subspace = False
    for index in indices:
        if state == 0:
            if index is not None:
                state = 1
        elif state == 1:
            if index is None:
                state = 2
        else:
            if index is not None:
                break
    else:
        has_contiguous_subspace = True

    # Transpose if not contiguous OR starts with None (and has tensor indices)
    need_post_process = False
    first_tensor_dim = None
    if not has_contiguous_subspace or (starts_with_none and has_any_tensor):
        dims = []
        transposed_indices = []
        # First add all non-None index positions
        for i, index in enumerate(indices):
            if index is not None:
                dims.append(i)
                transposed_indices.append(index)
        # Then add all None positions
        for i, index in enumerate(indices):
            if index is None:
                dims.append(i)
                transposed_indices.append(index)
        # Permute input
        inp = inp.permute(dims)
        indices = transposed_indices

        # Check if we need post-processing
        # (only when originally started with None and was contiguous)
        if starts_with_none and has_any_tensor and has_contiguous_subspace:
            need_post_process = True
            # Find first tensor dimension in original indices
            for i, idx in enumerate(original_indices):
                if idx is not None:
                    first_tensor_dim = i
                    break

    # Step 5: Now indices have contiguous subspace (after potential transpose)
    # Calculate output shape: before_shape + replacement_shape + after_shape
    before_shape = []
    after_shape = []
    replacement_shape = []

    for dim, index in enumerate(indices):
        if index is None:
            if replacement_shape:
                # None after tensor indices -> goes to after_shape
                after_shape.append(inp.shape[dim])
            else:
                # None before tensor indices -> goes to before_shape
                before_shape.append(inp.shape[dim])
        else:
            # First tensor index determines replacement_shape
            if not replacement_shape:
                replacement_shape = list(index.shape)

    # Step 6: Build output shape and create output tensor
    out_shape = before_shape + replacement_shape + after_shape
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # Step 7: Handle empty tensor case
    if inp.numel() == 0:
        return out

    # Step 8: Extract only tensor indices for kernel
    tensor_indices = [idx for idx in indices if idx is not None]
    if not tensor_indices:
        # All None, just reshape
        return inp.view(*out_shape)

    # Step 9: Call kernel with tensor indices
    index_wrapper(inp, tensor_indices, out)

    # Step 10: Post-process if needed (for originally contiguous tensor indices starting with None)
    if need_post_process:
        # Calculate index_rank from the first tensor index
        index_rank = tensor_indices[0].ndim
        # Create permutation order to move broadcast dimensions to correct position
        pre_dims = list(range(index_rank, index_rank + first_tensor_dim))
        broadcast_dims = list(range(index_rank))
        post_dims = list(range(index_rank + first_tensor_dim, out.ndim))
        new_order = pre_dims + broadcast_dims + post_dims
        out = out.permute(new_order)

    return out
