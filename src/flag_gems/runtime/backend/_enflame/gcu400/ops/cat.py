import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def cat_flat_kernel_4(
    out_ptr,
    in_ptr_a, in_ptr_b, in_ptr_c, in_ptr_d,
    out_offset_a, out_offset_b, out_offset_c, out_offset_d,
    size_a, size_b, size_c, size_d,
    num_programs_x,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    if pid_y == 0:
        in_ptr = in_ptr_a
        out_off = out_offset_a
        total = size_a
    elif pid_y == 1:
        in_ptr = in_ptr_b
        out_off = out_offset_b
        total = size_b
    elif pid_y == 2:
        in_ptr = in_ptr_c
        out_off = out_offset_c
        total = size_c
    else:
        in_ptr = in_ptr_d
        out_off = out_offset_d
        total = size_d

    block_start = pid_x * BLOCK_SIZE
    if block_start >= total:
        return

    in_block = tl.make_block_ptr(
        base=in_ptr,
        shape=(total,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    out_block = tl.make_block_ptr(
        base=out_ptr + out_off,
        shape=(total,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    stride = num_programs_x * BLOCK_SIZE
    for _offset in tl.range(block_start, total, stride, num_stages=3):
        data = tl.load(in_block, boundary_check=(0,))
        tl.store(out_block, data, boundary_check=(0,))
        in_block = tl.advance(in_block, (stride,))
        out_block = tl.advance(out_block, (stride,))


@triton.jit
def cat_general_kernel_4(
    out_ptr,
    in_ptr_a, in_ptr_b, in_ptr_c, in_ptr_d,
    dim_size_in_a, dim_size_in_b, dim_size_in_c, dim_size_in_d,
    dim_size_out,
    dim_prod_post,
    dim_offset_a, dim_offset_b, dim_offset_c, dim_offset_d,
    total_elements_a, total_elements_b, total_elements_c, total_elements_d,
    BLOCK_SIZE: tl.constexpr,
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

    block_start = pid_x * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    pre_idx = offsets // (dim_size_in * dim_prod_post)
    dim_idx = (offsets // dim_prod_post) % dim_size_in
    post_idx = offsets % dim_prod_post

    out_idx = (
        pre_idx * dim_size_out * dim_prod_post
        + (dim_idx + dim_offset) * dim_prod_post
        + post_idx
    )

    data = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS CAT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    device = A[0].device
    dtype = A[0].dtype
    A = list(A)
    for i in range(len(A) - 1, -1, -1):
        if A[i].shape == torch.Size([0]):
            A.pop(i)
    if len(A) == 0:
        return torch.tensor([], device=device, dtype=dtype)
    elif len(A) == 1:
        return A[0]

    assert dim >= -A[0].ndim and dim < A[0].ndim, f"Invalid dim: {dim}"
    dim = dim % A[0].ndim

    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx == dim:
                continue
            elif length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    out_shape = list(inp0_shape)
    out_shape[dim] = sum(s[dim] for s in inp_shapes)
    out = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)

    all_contiguous = all(a.is_contiguous() for a in A)
    BLOCK_SIZE = 8192

    if dim == 0 and all_contiguous:
        max_numel = max(a.numel() for a in A)
        if max_numel <= 16384:
            offset = 0
            for a in A:
                out.narrow(0, offset, a.shape[0]).copy_(a)
                offset += a.shape[0]
        else:
            bpe = A[0].element_size()
            if bpe == 2:
                total_bytes = out.numel() * 2
                if total_bytes % 4 == 0:
                    work_out = out.view(torch.int32)
                    work_A = [t.contiguous().view(torch.int32) for t in A]
                else:
                    work_out = out
                    work_A = A
            else:
                work_out = out
                work_A = A

            nw = 4

            offset = 0
            i = 0
            while i < len(work_A):
                batch = work_A[i : i + 4]
                num = len(batch)

                sizes = []
                offsets_list = []
                ptrs = []
                for t in batch:
                    n = t.numel()
                    sizes.append(n)
                    offsets_list.append(offset)
                    ptrs.append(t)
                    offset += n

                while len(sizes) < 4:
                    sizes.append(0)
                    offsets_list.append(0)
                    ptrs.append(batch[0])

                max_size = max(sizes)
                num_programs = min(triton.cdiv(max_size, BLOCK_SIZE), 8)
                grid = (num_programs, num)

                cat_flat_kernel_4[grid](
                    work_out,
                    ptrs[0], ptrs[1], ptrs[2], ptrs[3],
                    offsets_list[0], offsets_list[1], offsets_list[2], offsets_list[3],
                    sizes[0], sizes[1], sizes[2], sizes[3],
                    num_programs,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=nw,
                )

                i += num
    else:
        max_numel = max(a.numel() for a in A)
        if max_numel > 2097152:
            dim_offset = 0
            for a in A:
                ct = a.contiguous()
                out.narrow(dim, dim_offset, ct.shape[dim]).copy_(ct)
                dim_offset += ct.shape[dim]
        else:
            dim_prod_post = 1
            for d in range(dim + 1, A[0].ndim):
                dim_prod_post *= A[0].shape[d]

            dim_offset = 0
            i = 0
            while i < len(A):
                batch = A[i : i + 4]
                num = len(batch)

                args_tensors = []
                args_dim_sizes = []
                args_dim_offsets = []
                args_total_elements = []
                current_dim_offset = dim_offset

                for t in batch:
                    ct = t.contiguous()
                    args_tensors.append(ct)
                    args_dim_sizes.append(ct.shape[dim])
                    args_dim_offsets.append(current_dim_offset)
                    args_total_elements.append(ct.numel())
                    current_dim_offset += ct.shape[dim]

                while len(args_tensors) < 4:
                    args_tensors.append(batch[0])
                    args_dim_sizes.append(0)
                    args_dim_offsets.append(0)
                    args_total_elements.append(0)

                max_elements = max(args_total_elements)
                grid = (triton.cdiv(max_elements, BLOCK_SIZE), num)

                cat_general_kernel_4[grid](
                    out,
                    args_tensors[0], args_tensors[1], args_tensors[2], args_tensors[3],
                    args_dim_sizes[0], args_dim_sizes[1], args_dim_sizes[2], args_dim_sizes[3],
                    out_shape[dim],
                    dim_prod_post,
                    args_dim_offsets[0], args_dim_offsets[1], args_dim_offsets[2], args_dim_offsets[3],
                    args_total_elements[0], args_total_elements[1], args_total_elements[2], args_total_elements[3],
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=4,
                )

                dim_offset = current_dim_offset
                i += num

    return out
