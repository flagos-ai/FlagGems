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
    MAX_DATA_SIZE = 16 * 1024

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
    indices = list(indices)

    target_shape = get_max_rank_shape(indices)
    broadcast_indices(indices, target_shape)
    target_shape += inp.shape[len(indices) :]
    out = torch.empty(target_shape, dtype=inp.dtype, device=inp.device)

    index_wrapper(inp, indices, out)
    return out
