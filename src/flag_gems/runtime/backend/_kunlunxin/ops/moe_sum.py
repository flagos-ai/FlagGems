import logging

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _moe_sum_kernel(input, output, hidden_size, TOPK: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    token = tl.program_id(0)
    hidden = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden < hidden_size
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for expert in tl.static_range(0, TOPK):
        input_offset = (token * TOPK + expert) * hidden_size + hidden
        accumulator += tl.load(input + input_offset, mask=hidden_mask, other=0.0).to(tl.float32)
    tl.store(output + token * hidden_size + hidden, accumulator, mask=hidden_mask)


def moe_sum(input, output):
    logger.debug("GEMS_KUNLUNXIN MOE_SUM")
    input_work = input.contiguous()
    output_work = output if output.is_contiguous() else output.new_empty(output.shape)
    topk = input_work.shape[1]
    hidden_size = input_work.shape[2]
    block_size = 128
    with torch_device_fn.device(input.device):
        _moe_sum_kernel[(input_work.shape[0], triton.cdiv(hidden_size, block_size))](
            input_work,
            output_work,
            hidden_size,
            TOPK=topk,
            BLOCK_SIZE=block_size,
            buffer_size_limit=2048,
        )
    if output_work is not output:
        output.copy_(output_work)
