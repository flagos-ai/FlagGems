import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(
    f"flag_gems.runtime.backend._mthreads.ops.{__name__.split('.')[-1]}"
)


@libentry()
@triton.jit
def one_hot_comparison_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Single-pass comparison kernel — writes every output element (both zeros
    and ones) in one kernel launch.  Best for small num_classes (≤ 64) where
    the extra torch.zeros launch would cost more than the coalesced writes.

    Each block processes BLOCK_M input elements, comparing them against a
    tile of BLOCK_N classes at a time.
    """
    pid = ext.program_id(axis=0)
    row_start = pid * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < num_elements

    indices = tl.load(input_ptr + row_offsets, mask=row_mask, other=0)
    tl.device_assert(indices >= 0, "Class values must be non-negative.")
    tl.device_assert(
        indices < num_classes, "Class values must be smaller than num_classes."
    )
    out_base = row_offsets * num_classes

    for col_st in range(0, num_classes, BLOCK_N):
        col_offsets = col_st + tl.arange(0, BLOCK_N)
        valid_classes = col_offsets < num_classes
        out_offsets = out_base[:, None] + col_offsets[None, :]
        values = tl.where(indices[:, None] == col_offsets[None, :], 1, 0)
        combined_mask = row_mask[:, None] & valid_classes[None, :]
        tl.store(output_ptr + out_offsets, values, mask=combined_mask)


@libentry()
@triton.jit
def one_hot_scatter_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter-based kernel: writes only the 1s.  The output buffer must be
    pre-initialized to zero (done via torch.zeros).  Best for large
    num_classes (> 64) where the O(n*k) write volume of the comparison
    kernel would dominate.
    """
    pid = ext.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    tl.device_assert(indices >= 0, "Class values must be non-negative.")
    tl.device_assert(
        indices < num_classes, "Class values must be smaller than num_classes."
    )
    out_offsets = offsets * num_classes + indices
    tl.store(output_ptr + out_offsets, 1, mask=mask)


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    logger.debug("GEMS_MTHREADS ONE_HOT")

    if tensor.dtype != torch.int64:
        raise RuntimeError(
            "one_hot is only applicable to index tensor of type LongTensor."
        )

    if tensor.numel() == 0:
        if num_classes <= 0:
            raise RuntimeError(
                "Can not infer total number of classes from empty tensor."
            )
        shape = (*tensor.shape, num_classes)
        return torch.empty(shape, device=tensor.device, dtype=torch.int64)

    if num_classes == -1:
        num_classes = int(tensor.max().item()) + 1

    invalid = (tensor < 0) | (tensor >= num_classes)
    if invalid.any():
        if (tensor < 0).any():
            raise RuntimeError("Class values must be non-negative.")
        raise RuntimeError("Class values must be smaller than num_classes.")

    if num_classes < 1:
        raise RuntimeError("num_classes should be positive")

    if tensor.device.type == "cpu":
        out = torch.zeros((*tensor.shape, num_classes), device="cpu", dtype=torch.int64)
        out.scatter_(-1, tensor.unsqueeze(-1), 1)
        return out

    flat_input = tensor.contiguous().view(-1)
    num_elements = flat_input.numel()

    with torch_device_fn.device(tensor.device):
        if num_classes <= 64:
            # Small num_classes: single-pass comparison kernel.
            # Coalesced writes and single launch beat torch.zeros overhead.
            out = torch.empty(
                num_elements * num_classes, device=tensor.device, dtype=torch.int64
            )
            # Dynamic BLOCK_M: scale with num_elements for better GPU occupancy.
            # Larger blocks reduce grid launch count on big tensors.
            if num_elements <= 512:
                BLOCK_M = max(32, triton.next_power_of_2(num_elements))
            elif num_elements <= 65536:
                BLOCK_M = 128
            elif num_elements <= 262144:
                BLOCK_M = 256
            else:
                BLOCK_M = 512
            BLOCK_N = min(triton.next_power_of_2(num_classes), 64)
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_M"]),)
            one_hot_comparison_kernel[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
        else:
            # Large num_classes: scatter kernel.  O(n) writes beat O(n*k).
            out = torch.zeros(
                num_elements * num_classes, device=tensor.device, dtype=torch.int64
            )
            # Dynamic BLOCK_SIZE: more threads for large tensors.
            if num_elements <= 4096:
                BLOCK_SIZE = 256
            elif num_elements <= 65536:
                BLOCK_SIZE = 1024
            else:
                BLOCK_SIZE = 2048
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            one_hot_scatter_kernel[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return out.view(*tensor.shape, num_classes)
