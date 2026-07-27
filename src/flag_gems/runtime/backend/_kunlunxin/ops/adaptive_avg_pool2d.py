import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _adaptive_avg_pool2d_kernel(
    input,
    output,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    MAX_KH: tl.constexpr,
    MAX_KW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = ext.program_id(0)
    ow = tl.arange(0, BLOCK_SIZE)
    valid = ow < OW
    oh = program_id % OH
    nc = program_id // OH

    ih_start = (oh * IH) // OH
    ih_end = ((oh + 1) * IH + OH - 1) // OH
    iw_start = (ow * IW) // OW
    iw_end = ((ow + 1) * IW + OW - 1) // OW

    value = 0.0
    for kh in tl.static_range(MAX_KH):
        ih = ih_start + kh
        for kw in tl.static_range(MAX_KW):
            iw = iw_start + kw
            active = valid & (ih < ih_end) & (iw < iw_end)
            safe_ih = tl.minimum(ih, ih_end - 1)
            safe_iw = tl.minimum(iw, tl.minimum(iw_end - 1, IW - 1))
            input_offset = (nc * IH + safe_ih) * IW + safe_iw
            loaded = tl.load(input + input_offset).to(tl.float32)
            value += tl.where(active, loaded, 0.0)

    area = (ih_end - ih_start) * (iw_end - iw_start)
    output_offset = (nc * OH + oh) * OW + ow
    tl.store(output + output_offset, value / area, mask=valid)


def adaptive_avg_pool2d(input, output_size):
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_AVG_POOL2D")
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    output_height, output_width = output_size
    input_contiguous = input.contiguous()
    input_height, input_width = input.shape[-2:]
    output_shape = (*input.shape[:-2], output_height, output_width)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    if output.numel() == 0:
        return output

    max_kernel_height = triton.cdiv(input_height, output_height) + 1
    max_kernel_width = triton.cdiv(input_width, output_width) + 1
    output_rows = output.numel() // output_width
    block_size = triton.next_power_of_2(output_width)
    with torch_device_fn.device(input.device):
        _adaptive_avg_pool2d_kernel[(output_rows,)](
            input_contiguous,
            output,
            IH=input_height,
            IW=input_width,
            OH=output_height,
            OW=output_width,
            MAX_KH=max_kernel_height,
            MAX_KW=max_kernel_width,
            BLOCK_SIZE=block_size,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    return output
