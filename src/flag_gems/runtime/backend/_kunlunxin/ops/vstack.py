import logging

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def heur_block_size(args):
    return triton.next_power_of_2(
        triton.cdiv(args["max_tile_elems"], 12)
    )  # cluster_num


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("vstack"),
#     key=[
#         "max_tile_elems",
#     ],
# )
@triton.heuristics(
    values={
        "BLOCK_SIZE": heur_block_size,
    },
)
@triton.jit
def vstack_kernel(
    itensor_ptr0,
    itensor_ptr1,
    itensor_ptr2,
    itensor_ptr3,
    output_ptr,
    local_row0,
    local_row1,
    local_row2,
    local_row3,
    exc_row_offset0,
    exc_row_offset1,
    exc_row_offset2,
    exc_row_offset3,
    total_row_offset,
    row_stride,
    max_tile_elems,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tle.program_id(axis=0)
    tensor_idx = tle.program_id(axis=1)
    col_idx = tl.arange(0, BLOCK_SIZE)

    intensor_ptr = tl.where(tensor_idx == 0, itensor_ptr0, itensor_ptr1)
    intensor_ptr = tl.where(tensor_idx == 2, itensor_ptr2, intensor_ptr)
    intensor_ptr = tl.where(tensor_idx == 3, itensor_ptr3, intensor_ptr)
    base_exc_row_idx = tl.where(tensor_idx == 0, exc_row_offset0, exc_row_offset1)
    base_exc_row_idx = tl.where(tensor_idx == 2, exc_row_offset2, base_exc_row_idx)
    base_exc_row_idx = tl.where(tensor_idx == 3, exc_row_offset3, base_exc_row_idx)
    local_row = tl.where(tensor_idx == 0, local_row0, local_row1)
    local_row = tl.where(tensor_idx == 2, local_row2, local_row)
    local_row = tl.where(tensor_idx == 3, local_row3, local_row)

    end_idx = local_row * row_stride.to(tl.int64)
    idx = (pid_x * BLOCK_SIZE + col_idx).to(tl.int64)
    offset_mask = idx < end_idx
    in_offset = intensor_ptr + idx
    row_stride_offset = (total_row_offset + base_exc_row_idx) * row_stride.to(tl.int64)
    out_offset = output_ptr + row_stride_offset + idx
    out = tl.load(in_offset, mask=offset_mask)
    tl.store(out_offset, out, mask=offset_mask)


def vstack(tensors: list):
    logger.debug("GEMS VSTACK")

    tensors = torch.atleast_2d(tensors)
    num_tensors = len(tensors)
    assert num_tensors > 0

    # Ensure all tensors are on the same device and have the same dtype
    device = tensors[0].device
    dtype = tensors[0].dtype
    for tensor in tensors:
        assert (
            tensor.device == device
            and tensor.dtype == dtype
            and tensors[0].shape[1:] == tensor.shape[1:]
        )

    c_tensors = [t.contiguous() for t in tensors]
    # Calculate the output shape
    total_rows = sum(tensor.shape[0] for tensor in c_tensors)
    output_shape = list(c_tensors[0].shape)
    output_shape[0] = total_rows
    output = torch.empty(output_shape, device=device, dtype=dtype)
    row_stride = c_tensors[0].stride(0)

    outer_iters = triton.cdiv(num_tensors, 4)
    total_row_offset = 0
    for i in range(outer_iters):
        max_rows = 1
        itensors = []
        exclusive_row = []
        local_row = []
        array_row_offset = 0
        scheduled_num_tensors = 0
        for j in range(4):
            tensor_idx = i * 4 + j
            if tensor_idx < num_tensors:
                scheduled_num_tensors += 1
                itensors.append(c_tensors[tensor_idx])
                local_row.append(c_tensors[tensor_idx].shape[0])
                exclusive_row.append(array_row_offset)
                array_row_offset += c_tensors[tensor_idx].shape[0]
                max_rows = max(max_rows, c_tensors[tensor_idx].shape[0])
            else:
                empty_tensor = torch.empty(
                    0, dtype=c_tensors[0].dtype, device=c_tensors[0].device
                )
                itensors.append(empty_tensor)
                local_row.append(local_row[-1])
                exclusive_row.append(exclusive_row[-1])
        max_tile_elems = max_rows * row_stride
        grid = lambda META: (
            triton.cdiv(max_tile_elems, META["BLOCK_SIZE"]),
            scheduled_num_tensors,
        )
        # Launch the kernel
        with torch_device_fn.device(c_tensors[0].device):
            vstack_kernel[grid](
                itensors[0],
                itensors[1],
                itensors[2],
                itensors[3],
                output,
                local_row[0],
                local_row[1],
                local_row[2],
                local_row[3],
                exclusive_row[0],
                exclusive_row[1],
                exclusive_row[2],
                exclusive_row[3],
                total_row_offset,
                row_stride,
                max_tile_elems,
            )
            total_row_offset += array_row_offset
    return output
