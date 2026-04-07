import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# grid_sample: Samples input using grid coordinates.
# Supports bilinear and nearest interpolation modes.
# Supports zeros, border, and reflection padding modes.
# Input: (N, C, IH, IW), Grid: (N, OH, OW, 2) -> Output: (N, C, OH, OW)
@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["n_elements"],
)
@triton.jit
def grid_sample_2d_kernel(
    input_ptr,
    grid_ptr,
    output_ptr,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    n_elements,
    MODE: tl.constexpr,  # 0=bilinear, 1=nearest
    PADDING: tl.constexpr,  # 0=zeros, 1=border, 2=reflection
    ALIGN_CORNERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode to (n, c, oh, ow)
    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C

    # Load grid coordinates
    grid_idx = n * OH * OW * 2 + oh * OW * 2 + ow * 2
    gx = tl.load(grid_ptr + grid_idx, mask=mask, other=0.0).to(tl.float32)
    gy = tl.load(grid_ptr + grid_idx + 1, mask=mask, other=0.0).to(tl.float32)

    # Unnormalize grid coordinates
    if ALIGN_CORNERS:
        ix = (gx + 1.0) * 0.5 * (IW - 1)
        iy = (gy + 1.0) * 0.5 * (IH - 1)
    else:
        ix = ((gx + 1.0) * IW - 1.0) * 0.5
        iy = ((gy + 1.0) * IH - 1.0) * 0.5

    # Apply padding mode
    if PADDING == 1:  # border
        ix = tl.clamp(ix, 0.0, IW - 1.0)
        iy = tl.clamp(iy, 0.0, IH - 1.0)
    elif PADDING == 2:  # reflection
        # Simplified reflection: clamp after reflect
        ix = tl.abs(ix)
        iy = tl.abs(iy)
        ix = tl.where(ix > IW - 1.0, 2.0 * (IW - 1.0) - ix, ix)
        iy = tl.where(iy > IH - 1.0, 2.0 * (IH - 1.0) - iy, iy)
        ix = tl.clamp(ix, 0.0, IW - 1.0)
        iy = tl.clamp(iy, 0.0, IH - 1.0)

    input_base = n * C * IH * IW + c * IH * IW

    if MODE == 1:  # nearest
        ix_nearest = tl.math.nearbyint(ix).to(tl.int32)
        iy_nearest = tl.math.nearbyint(iy).to(tl.int32)
        valid = (
            (ix_nearest >= 0)
            & (ix_nearest < IW)
            & (iy_nearest >= 0)
            & (iy_nearest < IH)
        )
        in_idx = input_base + iy_nearest * IW + ix_nearest
        val = tl.load(input_ptr + in_idx, mask=mask & valid, other=0.0)
    else:  # bilinear
        ix0 = tl.math.floor(ix).to(tl.int32)
        iy0 = tl.math.floor(iy).to(tl.int32)
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        wa = (ix1.to(tl.float32) - ix) * (iy1.to(tl.float32) - iy)
        wb = (ix - ix0.to(tl.float32)) * (iy1.to(tl.float32) - iy)
        wc = (ix1.to(tl.float32) - ix) * (iy - iy0.to(tl.float32))
        wd = (ix - ix0.to(tl.float32)) * (iy - iy0.to(tl.float32))

        va_mask = (ix0 >= 0) & (ix0 < IW) & (iy0 >= 0) & (iy0 < IH)
        vb_mask = (ix1 >= 0) & (ix1 < IW) & (iy0 >= 0) & (iy0 < IH)
        vc_mask = (ix0 >= 0) & (ix0 < IW) & (iy1 >= 0) & (iy1 < IH)
        vd_mask = (ix1 >= 0) & (ix1 < IW) & (iy1 >= 0) & (iy1 < IH)

        va = tl.load(
            input_ptr + input_base + iy0 * IW + ix0, mask=mask & va_mask, other=0.0
        )
        vb = tl.load(
            input_ptr + input_base + iy0 * IW + ix1, mask=mask & vb_mask, other=0.0
        )
        vc = tl.load(
            input_ptr + input_base + iy1 * IW + ix0, mask=mask & vc_mask, other=0.0
        )
        vd = tl.load(
            input_ptr + input_base + iy1 * IW + ix1, mask=mask & vd_mask, other=0.0
        )

        val = (
            wa * va.to(tl.float32)
            + wb * vb.to(tl.float32)
            + wc * vc.to(tl.float32)
            + wd * vd.to(tl.float32)
        )

    tl.store(output_ptr + offsets, val, mask=mask)


def grid_sample(
    input, grid, mode="bilinear", padding_mode="zeros", align_corners=False
):
    logger.debug("GEMS GRID_SAMPLE")
    assert input.ndim == 4, "Input must be 4D (N, C, H, W)"
    assert grid.ndim == 4, "Grid must be 4D (N, OH, OW, 2)"

    N, C, IH, IW = input.shape
    _, OH, OW, _ = grid.shape

    input = input.contiguous()
    grid = grid.contiguous()
    output = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)

    mode_int = 0 if mode == "bilinear" else 1
    padding_int = {"zeros": 0, "border": 1, "reflection": 2}[padding_mode]

    n_elements = output.numel()
    if n_elements == 0:
        return output

    grid_launch = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        grid_sample_2d_kernel[grid_launch](
            input,
            grid,
            output,
            N,
            C,
            IH,
            IW,
            OH,
            OW,
            n_elements,
            MODE=mode_int,
            PADDING=padding_int,
            ALIGN_CORNERS=align_corners,
        )
    return output
