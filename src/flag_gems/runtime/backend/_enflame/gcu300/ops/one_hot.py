import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def one_hot_kernel(
    index_ptr,
    out_ptr,
    num_classes,
    numel,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    row_start = pid * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < numel

    target_classes = tl.load(index_ptr + row_offsets, mask=row_mask, other=0)

    for col_st in range(0, num_classes, BLOCK_N):
        col_offsets = col_st + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < num_classes
        result = target_classes[:, None] == col_offsets[None, :]
        # GCU300 does not support 64-bit data types; accumulate the one-hot
        # result in int32 and cast back to int64 on the host after the kernel.
        result = result.to(tl.int32)
        offs_2d = row_offsets[:, None] * num_classes + col_offsets[None, :]
        tl.store(out_ptr + offs_2d, result, mask=row_mask[:, None] & col_mask[None, :])


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME ONE_HOT")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    numel = tensor.numel()
    if num_classes == -1:
        num_classes = int(tensor.max().item()) + 1

    # GCU300 does not support 64-bit data types. torch.one_hot returns int64,
    # so allocate an int32 output for the kernel and cast back to int64 after.
    out_i32 = torch.empty(
        (*tensor.shape, num_classes), device=tensor.device, dtype=torch.int32
    )
    BLOCK_N = triton.next_power_of_2(num_classes)
    BLOCK_N = min(BLOCK_N, 128)
    BLOCK_M = 32

    grid = (triton.cdiv(numel, BLOCK_M),)

    with torch_device_fn.device(tensor.device):
        one_hot_kernel[grid](
            tensor,
            out_i32,
            num_classes,
            numel,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    return out_i32.to(torch.int64)
