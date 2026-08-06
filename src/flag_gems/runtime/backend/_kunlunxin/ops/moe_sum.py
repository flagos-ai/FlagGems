import logging

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _moe_sum_kernel(input, output, hidden_size, TOPK: tl.constexpr):
    output_offset = tl.program_id(0)
    token = output_offset // hidden_size
    hidden = output_offset % hidden_size
    accumulator = 0.0
    for expert in tl.static_range(0, TOPK):
        input_offset = (token * TOPK + expert) * hidden_size + hidden
        accumulator += tl.load(input + input_offset).to(tl.float32)
    tl.store(output + output_offset, accumulator)


def moe_sum(input, output):
    logger.debug("GEMS_KUNLUNXIN MOE_SUM")
    input_work = input.contiguous()
    output_work = output if output.is_contiguous() else output.new_empty(output.shape)
    topk = input_work.shape[1]
    hidden_size = input_work.shape[2]
    with torch_device_fn.device(input.device):
        _moe_sum_kernel[(output_work.numel(),)](
            input_work,
            output_work,
            hidden_size,
            TOPK=topk,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    if output_work is not output:
        output.copy_(output_work)
