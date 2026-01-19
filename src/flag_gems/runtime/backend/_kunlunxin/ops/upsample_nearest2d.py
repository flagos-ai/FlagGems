import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
device = device.name


def heur_block_size(args):
    return triton.next_power_of_2(
        triton.cdiv(args["N"] * args["C"] * args["OH"] * args["OW"], 12)
    )  # cluster_num


# @triton.autotune(
#     configs=runtime.get_tuned_config("upsample_nearest2d"), key=["N", "C", "OH", "OW"]
# )
@triton.heuristics(
    {
        "SAME_H": lambda args: args["OH"] == args["IH"],
        "SAME_W": lambda args: args["OW"] == args["IW"],
        "BLOCK_SIZE": heur_block_size,
    }
)
@triton.jit
def upsample_nearest2d_kernel(
    ptr_o,
    ptr_i,
    N: tl.constexpr,
    C: tl.constexpr,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ow = idx % OW
    oh = idx // OW % OH
    c = idx // OW // OH % C
    n = idx // OW // OH // C % N
    if SAME_H:
        ih = oh
    else:
        # tl.floor() cannot be found in 2.3.1, using int trunc
        ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)
    offset_o = ((n * C + c) * OH + oh) * OW + ow
    offset_i = ((n * C + c) * IH + ih) * IW + iw
    data = tl.load(ptr_i + offset_i)
    tl.store(ptr_o + offset_o, data)


def upsample_nearest2d(
    input: torch.Tensor,
    output_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D")
    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    OH, OW = output_size
    N, C, IH, IW = input.shape
    if scales_h is not None:
        reciprocal_scale_h = 1 / scales_h
    else:
        reciprocal_scale_h = IH / OH
    if scales_w is not None:
        reciprocal_scale_w = 1 / scales_w
    else:
        reciprocal_scale_w = IW / OW
    # allocate output
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    total_threads = N * C * OH * OW
    grid = lambda META: (triton.cdiv(total_threads, META["BLOCK_SIZE"]),)
    with torch_device_fn.device(input.device):
        upsample_nearest2d_kernel[grid](
            output, input, N, C, OH, OW, IH, IW, reciprocal_scale_h, reciprocal_scale_w
        )
    return output
