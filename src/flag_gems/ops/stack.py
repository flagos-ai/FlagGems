import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def stack_copy_func_kernel_4(
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
    post_idx = idx % dim_prod_post
    pre_idx = idx // dim_prod_post

    out_idx = (
        pre_idx * dim_size_out * dim_prod_post + dim_offset * dim_prod_post + post_idx
    )

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)


def stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS STACK")

    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    for i, s in enumerate(inp_shapes[1:]):
        if (dim < -tensors[i + 1].dim() - 1) or (dim > tensors[i + 1].dim()):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -tensors[i + 1].dim() - 1, tensors[i + 1].dim(), dim
                )
            )
        if s != inp0_shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {inp0_shape} at entry 0 and {s} at entry {i + 1}"
            )

    if dim < 0:
        dim = dim + len(inp0_shape) + 1

    # Type promotion: find the common dtype for all tensors
    dtypes = [t.dtype for t in tensors]
    dtype = dtypes[0]
    for dt in dtypes[1:]:
        dtype = torch.promote_types(dtype, dt)
    # Convert all tensors to the result dtype if needed
    tensors = [t.to(dtype) if t.dtype != dtype else t for t in tensors]
    device = tensors[0].device
    out_shape = inp0_shape[:dim] + [len(tensors)] + inp0_shape[dim:]
    out = torch.empty(out_shape, dtype=dtype, device=device)

    dim_prod_post = 1
    for s in inp0_shape[dim:]:
        dim_prod_post *= s

    BLOCK = 1024
    i = 0
    while i < len(tensors):
        tensors_in_batch = tensors[i : i + 4]
        num_tensors_in_batch = len(tensors_in_batch)

        args = []
        total_elements_list = []

        for j in range(4):
            if j < num_tensors_in_batch:
                tensor = tensors_in_batch[j].contiguous()
                total_elements = tensor.numel()
                args.extend([tensor, 1, i + j, total_elements])
                total_elements_list.append(total_elements)
            else:
                args.extend([tensors_in_batch[0], 0, 0, 0])
                total_elements_list.append(0)

        dim_size_out = len(tensors)

        grid_y = num_tensors_in_batch
        max_elements_in_batch = tensors[0].numel() if total_elements_list else 0
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

        stack_copy_func_kernel_4[grid](
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
        i += num_tensors_in_batch

    return out
