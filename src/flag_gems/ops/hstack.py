import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def hstack_copy_func_kernel_4(
    out_ptr,
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    in_ptr_d,
    dim_size_in_a,
    dim_size_in_b,
    dim_size_in_c,
    dim_size_in_d,
    dim_size_out,
    dim_prod_post,
    dim_offset_a,
    dim_offset_b,
    dim_offset_c,
    dim_offset_d,
    total_elements_a,
    total_elements_b,
    total_elements_c,
    total_elements_d,
    BLOCK_X: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    if pid_y == 0:
        in_ptr = in_ptr_a
        dim_size_in = dim_size_in_a
        dim_offset = dim_offset_a
        total_elements = total_elements_a
    elif pid_y == 1:
        in_ptr = in_ptr_b
        dim_size_in = dim_size_in_b
        dim_offset = dim_offset_b
        total_elements = total_elements_b
    elif pid_y == 2:
        in_ptr = in_ptr_c
        dim_size_in = dim_size_in_c
        dim_offset = dim_offset_c
        total_elements = total_elements_c
    else:
        in_ptr = in_ptr_d
        dim_size_in = dim_size_in_d
        dim_offset = dim_offset_d
        total_elements = total_elements_d

    block_start = pid_x * BLOCK_X
    offsets = tl.arange(0, BLOCK_X)
    mask = block_start + offsets < total_elements

    idx = block_start + offsets

    pre_idx = idx // (dim_size_in * dim_prod_post)
    dim_idx = (idx // dim_prod_post) % dim_size_in
    post_idx = idx % dim_prod_post

    out_idx = (
        pre_idx * dim_size_out * dim_prod_post
        + (dim_idx + dim_offset) * dim_prod_post
        + post_idx
    )

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)


def hstack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
) -> torch.Tensor:
    logger.debug("GEMS HSTACK")

    if len(tensors) == 0:
        raise RuntimeError("hstack expected a non-empty TensorList")

    if tensors[0].ndim == 0:
        tensors[0] = tensors[0].view(1)
    inp0_shape = tensors[0].shape
    out_shape = list(inp0_shape)
    inp_shapes = [inp0_shape]

    if len(inp0_shape) == 1:
        dim = 0
    else:
        dim = 1

    for tensor_num, tensor in enumerate(tensors[1:]):
        if tensor.ndim == 0:
            tensor = tensor.view(1)
        if tensor.ndim != tensors[0].ndim:
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {tensors[0].ndim} and {tensor.ndim}"
            )

        inp_shape = tensor.shape
        inp_shapes.append(inp_shape)

        for i in range(len(inp_shape)):
            if i != dim and inp_shape[i] != inp0_shape[i]:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. \
                        Expected size {inp0_shape[i]} but got size {inp_shape[i]} \
                        for tensor number {tensor_num + 1} in the list."
                )

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]

    # Type promotion: find the common dtype for all tensors
    dtypes = [t.dtype for t in tensors]
    dtype = dtypes[0]
    for dt in dtypes[1:]:
        dtype = torch.promote_types(dtype, dt)
    # Convert all tensors to the common dtype if needed
    tensors = [t.to(dtype) if t.dtype != dtype else t for t in tensors]
    device = tensors[0].device
    out_shape[dim] = sum(s[dim] for s in inp_shapes)
    out = torch.empty(out_shape, dtype=dtype, device=device)

    dim_prod_post = 1
    for s in inp0_shape[dim:]:
        dim_prod_post *= s
    BLOCK = 1024
    dim_offset = 0
    i = 0
    while i < len(tensors):
        tensors_in_batch = tensors[i : i + 4]
        num_tensors_in_batch = len(tensors_in_batch)

        args = []
        total_elements_list = []
        current_dim_offset = dim_offset

        for j in range(4):
            if j < num_tensors_in_batch:
                tensor = tensors_in_batch[j].contiguous()
                shape = tensor.shape
                total_elements = tensor.numel()
                dim_size_in = shape[dim]

                args.extend([tensor, dim_size_in, current_dim_offset, total_elements])
                total_elements_list.append(total_elements)
                current_dim_offset += dim_size_in
            else:
                # Add placeholders for unused tensor slots
                args.extend([tensors_in_batch[0], 0, 0, 0])
                total_elements_list.append(0)

        dim_size_out = out_shape[dim]
        dim_prod_post = 1
        for d in range(dim + 1, tensors[0].ndim):
            dim_prod_post *= tensors[0].shape[d]

        grid_y = num_tensors_in_batch
        max_elements_in_batch = max(total_elements_list) if total_elements_list else 0
        grid = (triton.cdiv(max_elements_in_batch, BLOCK), grid_y)

        (
            tensor_a,
            dim_size_in_a,
            dim_offset_a,
            total_elements_a,
            tensor_b,
            dim_size_in_b,
            dim_offset_b,
            total_elements_b,
            tensor_c,
            dim_size_in_c,
            dim_offset_c,
            total_elements_c,
            tensor_d,
            dim_size_in_d,
            dim_offset_d,
            total_elements_d,
        ) = args

        hstack_copy_func_kernel_4[grid](
            out,
            tensor_a,
            tensor_b,
            tensor_c,
            tensor_d,
            dim_size_in_a,
            dim_size_in_b,
            dim_size_in_c,
            dim_size_in_d,
            dim_size_out,
            dim_prod_post,
            dim_offset_a,
            dim_offset_b,
            dim_offset_c,
            dim_offset_d,
            total_elements_a,
            total_elements_b,
            total_elements_c,
            total_elements_d,
            BLOCK_X=BLOCK,
        )

        dim_offset = current_dim_offset
        i += num_tensors_in_batch

    return out
