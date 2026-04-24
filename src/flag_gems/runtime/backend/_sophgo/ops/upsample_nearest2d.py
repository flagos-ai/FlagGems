import logging
from math import gcd
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

EXPECTED_DEVICE_TYPE = runtime_device.name


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest2d"),
    key=[
        "N",
        "C",
        "OH",
        "OW",
        "SCALE_H_NUM",
        "SCALE_H_DEN",
        "SCALE_W_NUM",
        "SCALE_W_DEN",
    ],
)
@triton.jit
def upsample_nearest2d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    SCALE_H_NUM: tl.constexpr,
    SCALE_H_DEN: tl.constexpr,
    SCALE_W_NUM: tl.constexpr,
    SCALE_W_DEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    ow = idx % OW
    oh = idx // OW % OH
    c = idx // OW // OH % C
    n = idx // OW // OH // C % N

    ih_calc = (oh * SCALE_H_NUM) // SCALE_H_DEN
    iw_calc = (ow * SCALE_W_NUM) // SCALE_W_DEN

    ih_temp = tl.where(ih_calc > (IH - 1), IH - 1, ih_calc)
    ih = tl.where(ih_temp < 0, 0, ih_temp)

    iw_temp = tl.where(iw_calc > (IW - 1), IW - 1, iw_calc)
    iw = tl.where(iw_temp < 0, 0, iw_temp)

    offset_o = ((n * C + c) * OH + oh) * OW + ow
    offset_i = ((n * C + c) * IH + ih) * IW + iw

    mask = idx < N * C * OH * OW

    data = tl.load(ptr_i + offset_i, mask=mask, other=0.0)
    tl.store(ptr_o + offset_o, data, mask=mask)


def upsample_nearest2d(
    input: torch.Tensor,
    output_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    """
    Upsample Nearest 2D - Pure TPU implementation

    Implementation strategy (consistent with PyTorch):
        - User passes size argument → use input_size / output_size (integer ratio)
        - User passes scale_factor argument → use Fraction for high-precision conversion

    Args:
        input: Input tensor (N, C, IH, IW)
        output_size: Output dimensions (OH, OW)
        scales_h: Height scale factor (from scale_factor argument)
        scales_w: Width scale factor (from scale_factor argument)

    Returns:
        output: Output tensor (N, C, OH, OW)
    """
    logging.debug("GEMS UPSAMPLE NEAREST2D (Pure TPU)")

    # ========== Input validation ==========
    if input.device.type not in ["tpu", "sophgo"]:
        raise RuntimeError(
            f"upsample_nearest2d expects TPU device, got {input.device.type}"
        )

    if input.ndim != 4:
        raise ValueError(f"Expected 4D input, got {input.ndim}D")

    if len(output_size) != 2:
        raise ValueError("Expected output_size length 2")

    OH, OW = output_size
    N, C, IH, IW = input.shape

    if OH <= 0 or OW <= 0:
        raise ValueError(f"Invalid output_size: {output_size}")

    if IH <= 0 or IW <= 0:
        raise ValueError(f"Invalid input shape: {input.shape}")

    # ========== Compute scale ratios (consistent with PyTorch logic) ==========
    if scales_h is not None:
        # User passed scale_factor → use Fraction for high-precision conversion
        from fractions import Fraction

        frac_h = Fraction(1 / scales_h).limit_denominator(10000)
        scale_h_num = frac_h.numerator
        scale_h_den = frac_h.denominator
    else:
        # User passed size → use input_size / output_size
        scale_h_num = IH
        scale_h_den = OH

    if scales_w is not None:
        from fractions import Fraction

        frac_w = Fraction(1 / scales_w).limit_denominator(10000)
        scale_w_num = frac_w.numerator
        scale_w_den = frac_w.denominator
    else:
        scale_w_num = IW
        scale_w_den = OW

    # Simplify fractions (reduce overflow risk)
    g_h = gcd(scale_h_num, scale_h_den)
    scale_h_num //= g_h
    scale_h_den //= g_h

    g_w = gcd(scale_w_num, scale_w_den)
    scale_w_num //= g_w
    scale_w_den //= g_w

    # ========== Check for overflow ==========
    MAX_SAFE_PRODUCT = 2**30

    if scale_h_num * OH >= MAX_SAFE_PRODUCT or scale_w_num * OW >= MAX_SAFE_PRODUCT:
        raise RuntimeError(
            f"upsample_nearest2d: Input/output size too large, may cause overflow. "
            f"Input: {IH}x{IW}, Output: {OH}x{OW}, "
            f"Scale (simplified): H={scale_h_num}/{scale_h_den}, W={scale_w_num}/{scale_w_den}"
        )

    logging.debug(
        f"Input: {IH}x{IW} -> Output: {OH}x{OW}, "
        f"Scale: H={scale_h_num}/{scale_h_den}, W={scale_w_num}/{scale_w_den}, "
        f"Source: {'scale_factor' if scales_h is not None else 'size'}"
    )

    # ========== Allocate output and launch kernel ==========
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)

    total_threads = N * C * OH * OW
    grid = lambda META: (triton.cdiv(total_threads, META["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        upsample_nearest2d_kernel[grid](
            output,
            input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            SCALE_H_NUM=scale_h_num,
            SCALE_H_DEN=scale_h_den,
            SCALE_W_NUM=scale_w_num,
            SCALE_W_DEN=scale_w_den,
        )

    return output
