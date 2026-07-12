import logging
import math
from typing import Sequence

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def cubic_weight(d, a: tl.constexpr):
    ad = tl.abs(d)
    ad2 = ad * ad
    ad3 = ad2 * ad
    w1 = (a + 2.0) * ad3 - (a + 3.0) * ad2 + 1.0
    w2 = a * ad3 - 5.0 * a * ad2 + 8.0 * a * ad - 4.0 * a
    return tl.where(ad <= 1.0, w1, tl.where(ad < 2.0, w2, 0.0))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 128}, num_warps=4),
        triton.Config({"BLOCK_W": 256}, num_warps=4),
        triton.Config({"BLOCK_W": 256}, num_warps=8),
        triton.Config({"BLOCK_W": 512}, num_warps=8),
        triton.Config({"BLOCK_W": 1024}, num_warps=8),
    ],
    key=["W_out"],
)
@triton.jit
def _upsample_bicubic2d_row_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    strideN,
    strideC,
    strideH,
    strideW,
    out_strideN,
    out_strideC,
    out_strideH,
    out_strideW,
    scale_h,
    scale_w,
    align_corners: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)

    pid_w = pid % num_w_blocks
    row_id = pid // num_w_blocks

    y_out = row_id % H_out
    nc = row_id // H_out
    c = nc % C
    n = nc // C

    fy = y_out * 1.0
    if align_corners:
        in_y = fy * scale_h
    else:
        in_y = (fy + 0.5) * scale_h - 0.5

    y0f = tl.floor(in_y)
    y0 = y0f.to(tl.int32)
    ty = in_y - y0f

    y_m1 = tl.maximum(0, tl.minimum(H_in - 1, y0 - 1))
    y_0 = tl.maximum(0, tl.minimum(H_in - 1, y0 + 0))
    y_p1 = tl.maximum(0, tl.minimum(H_in - 1, y0 + 1))
    y_p2 = tl.maximum(0, tl.minimum(H_in - 1, y0 + 2))

    a = -0.75
    wy0 = cubic_weight(1.0 + ty, a)
    wy1 = cubic_weight(ty, a)
    wy2 = cubic_weight(1.0 - ty, a)
    wy3 = cubic_weight(2.0 - ty, a)

    n_64 = n.to(tl.int64)
    c_64 = c.to(tl.int64)
    base_ptr = in_ptr + n_64 * strideN + c_64 * strideC

    row_m1_ptr = base_ptr + y_m1.to(tl.int64) * strideH
    row_0_ptr = base_ptr + y_0.to(tl.int64) * strideH
    row_p1_ptr = base_ptr + y_p1.to(tl.int64) * strideH
    row_p2_ptr = base_ptr + y_p2.to(tl.int64) * strideH

    x_out = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = x_out < W_out

    fx = x_out.to(tl.float32)
    if align_corners:
        in_x = fx * scale_w
    else:
        in_x = (fx + 0.5) * scale_w - 0.5

    x0f = tl.floor(in_x)
    x0 = x0f.to(tl.int32)
    tx = in_x - x0f

    x_m1 = tl.maximum(0, tl.minimum(W_in - 1, x0 - 1))
    x_0 = tl.maximum(0, tl.minimum(W_in - 1, x0 + 0))
    x_p1 = tl.maximum(0, tl.minimum(W_in - 1, x0 + 1))
    x_p2 = tl.maximum(0, tl.minimum(W_in - 1, x0 + 2))

    wx0 = cubic_weight(1.0 + tx, a)
    wx1 = cubic_weight(tx, a)
    wx2 = cubic_weight(1.0 - tx, a)
    wx3 = cubic_weight(2.0 - tx, a)

    off_x_m1 = x_m1 * strideW
    off_x_0 = x_0 * strideW
    off_x_p1 = x_p1 * strideW
    off_x_p2 = x_p2 * strideW

    v0 = tl.load(row_m1_ptr + off_x_m1).to(tl.float32)
    v1 = tl.load(row_m1_ptr + off_x_0).to(tl.float32)
    v2 = tl.load(row_m1_ptr + off_x_p1).to(tl.float32)
    v3 = tl.load(row_m1_ptr + off_x_p2).to(tl.float32)
    acc = (v0 * wx0 + v1 * wx1 + v2 * wx2 + v3 * wx3) * wy0

    v0 = tl.load(row_0_ptr + off_x_m1).to(tl.float32)
    v1 = tl.load(row_0_ptr + off_x_0).to(tl.float32)
    v2 = tl.load(row_0_ptr + off_x_p1).to(tl.float32)
    v3 = tl.load(row_0_ptr + off_x_p2).to(tl.float32)
    acc += (v0 * wx0 + v1 * wx1 + v2 * wx2 + v3 * wx3) * wy1

    v0 = tl.load(row_p1_ptr + off_x_m1).to(tl.float32)
    v1 = tl.load(row_p1_ptr + off_x_0).to(tl.float32)
    v2 = tl.load(row_p1_ptr + off_x_p1).to(tl.float32)
    v3 = tl.load(row_p1_ptr + off_x_p2).to(tl.float32)
    acc += (v0 * wx0 + v1 * wx1 + v2 * wx2 + v3 * wx3) * wy2

    v0 = tl.load(row_p2_ptr + off_x_m1).to(tl.float32)
    v1 = tl.load(row_p2_ptr + off_x_0).to(tl.float32)
    v2 = tl.load(row_p2_ptr + off_x_p1).to(tl.float32)
    v3 = tl.load(row_p2_ptr + off_x_p2).to(tl.float32)
    acc += (v0 * wx0 + v1 * wx1 + v2 * wx2 + v3 * wx3) * wy3

    out_offset = (
        n_64 * out_strideN
        + c_64 * out_strideC
        + y_out.to(tl.int64) * out_strideH
        + x_out.to(tl.int64) * out_strideW
    )
    tl.store(out_ptr + out_offset, acc.to(out_ptr.dtype.element_ty), mask=mask)


def upsample_bicubic2d(
    input: torch.Tensor,
    output_size: Sequence[int] | None = None,
    align_corners: bool = False,
    scales_h: float | None = None,
    scales_w: float | None = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE BICUBIC2D")
    scale_factors = (scales_h, scales_w)

    if input.dim() != 4:
        raise ValueError("input must be a 4D tensor (N, C, H, W)")
    if output_size is None and scale_factors is None:
        raise ValueError("Either output_size or scale_factors must be provided")

    N, C, H_in, W_in = input.shape

    if output_size is not None:
        if len(output_size) != 2:
            raise ValueError(
                "output_size must be a sequence of two ints (H_out, W_out)"
            )
        H_out, W_out = int(output_size[0]), int(output_size[1])
    else:
        if len(scale_factors) == 2:
            sh, sw = float(scale_factors[0]), float(scale_factors[1])
        elif len(scale_factors) == 1:
            sh = sw = float(scale_factors[0])
        else:
            raise ValueError("scale_factors must have length 1 or 2 for 2D upsampling")
        H_out = max(int(math.floor(H_in * sh)), 1)
        W_out = max(int(math.floor(W_in * sw)), 1)

    if H_out <= 0 or W_out <= 0:
        raise ValueError("Output size must be positive")

    device = input.device
    if not input.is_cuda:
        raise ValueError("This Triton kernel requires CUDA tensors")

    if align_corners:
        scale_h = 0.0 if H_out <= 1 else (H_in - 1.0) / (H_out - 1.0)
        scale_w = 0.0 if W_out <= 1 else (W_in - 1.0) / (W_out - 1.0)
    else:
        scale_h = float(H_in) / float(H_out)
        scale_w = float(W_in) / float(W_out)

    out = torch.empty((N, C, H_out, W_out), dtype=input.dtype, device=device)

    sN, sC, sH, sW = input.stride()
    oN, oC, oH, oW = out.stride()

    grid = lambda meta: (triton.cdiv(W_out, meta["BLOCK_W"]) * N * C * H_out,)

    _upsample_bicubic2d_row_kernel[grid](
        input,
        out,
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        sN,
        sC,
        sH,
        sW,
        oN,
        oC,
        oH,
        oW,
        float(scale_h),
        float(scale_w),
        align_corners,
    )

    return out
