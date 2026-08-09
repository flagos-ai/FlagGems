import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Windows with an exact integer input/output ratio are handled by a dedicated
# row-reduction kernel (contiguous 2D tile loads + in-kernel axis-1 sum, no
# per-lane integer division and no masked accumulate chain). The cap keeps the
# fully-unrolled tile count bounded; beyond it the XPU backend's buffer-size
# tuning fails on huge static unrolls (observed with a 57x57 window / 3249
# unrolled iterations -> "Failed to tune buffer size").
INT_KERNEL_MAX_UNROLL = 65536


@libentry()
@triton.jit
def _adaptive_avg_pool2d_general_kernel(
    input,
    output,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    MAX_KH,
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

    value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for kh in range(0, MAX_KH):
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


@libentry()
@triton.jit
def _adaptive_avg_pool2d_int_kernel(
    input,
    output,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    G: tl.constexpr,
    BG: tl.constexpr,
    BKW: tl.constexpr,
    AREA: tl.constexpr,
):
    program_id = ext.program_id(0)
    oh = program_id % OH
    nc = program_id // OH
    r = tl.arange(0, BG)
    c = tl.arange(0, BKW)
    base = ((nc * OH + oh) * KH) * IW

    value = tl.zeros((BG,), dtype=tl.float32)
    for kh in tl.static_range(KH):
        tile = tl.load(
            input + base + kh * IW + r[:, None] * KW + c[None, :],
            mask=(r[:, None] < G) & (c[None, :] < KW),
            other=0.0,
        ).to(tl.float32)
        value += tl.sum(tile, axis=1)

    ow = tl.arange(0, BG)
    tl.store(output + (nc * OH + oh) * OW + ow, value / AREA, mask=ow < OW)


def adaptive_avg_pool2d(input, output_size):
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_AVG_POOL2D")
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    output_height, output_width = output_size
    input_contiguous = input.contiguous()
    input_height, input_width = input_contiguous.shape[-2:]
    output_shape = (*input_contiguous.shape[:-2], output_height, output_width)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    if output.numel() == 0:
        return output

    output_rows = output.numel() // output_width  # N * C * OH programs
    with torch_device_fn.device(input.device):
        if (
            output_height > 0
            and output_width > 0
            and input_height % output_height == 0
            and input_width % output_width == 0
        ):
            kernel_height = input_height // output_height
            kernel_width = input_width // output_width
            if kernel_height * kernel_width <= INT_KERNEL_MAX_UNROLL:
                groups = input_width // kernel_width
                _adaptive_avg_pool2d_int_kernel[(output_rows,)](
                    input_contiguous,
                    output,
                    IW=input_width,
                    OH=output_height,
                    OW=output_width,
                    KH=kernel_height,
                    KW=kernel_width,
                    G=groups,
                    BG=triton.next_power_of_2(groups),
                    BKW=triton.next_power_of_2(kernel_width),
                    AREA=kernel_height * kernel_width,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
                return output

        max_kernel_height = triton.cdiv(input_height, output_height) + 1
        max_kernel_width = triton.cdiv(input_width, output_width) + 1
        block_size = triton.next_power_of_2(output_width)
        _adaptive_avg_pool2d_general_kernel[(output_rows,)](
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