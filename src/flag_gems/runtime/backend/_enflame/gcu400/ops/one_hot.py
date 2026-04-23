import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

device = device.name

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["total_elements", "num_classes"])
def one_hot_kernel(
    output_ptr,
    input_ptr,
    total_elements,
    num_classes,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for elem_id in tl.range(pid, total_elements, num_programs, num_stages=3):
        val = tl.load(input_ptr + elem_id)

        for c_start in tl.range(0, num_classes, BLOCK_C):
            c_offsets = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offsets < num_classes
            result = tl.where(c_offsets == val, 1, 0)
            tl.store(
                output_ptr + elem_id * num_classes + c_offsets,
                result,
                mask=c_mask,
            )


SMALL_THRESHOLD = 1048576


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    logger.debug("GEMS ONE_HOT")
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

    output = torch.zeros(
        (*tensor.shape, num_classes), device=tensor.device, dtype=torch.int64
    )
    torch.ops.aten.scatter_.value(output, -1, tensor.unsqueeze(-1), 1)
    return output
