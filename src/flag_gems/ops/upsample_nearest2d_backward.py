import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


def _build_range_lut(
    output_size: int,
    input_size: int,
    scale: Optional[float],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fwd_scale = scale if scale is not None else (output_size / input_size)
    k = torch.arange(input_size, dtype=torch.float32, device=device)
    starts = torch.ceil(k * fwd_scale).to(torch.int32).clamp(max=output_size)
    ends = torch.ceil((k + 1) * fwd_scale).to(torch.int32).clamp(max=output_size)
    return starts, ends


def _is_integer_scale(
    output_size: int, input_size: int, scale: Optional[float]
) -> int:
    if scale is not None:
        s = int(scale)
        return s if float(s) == scale else 0
    return output_size // input_size if output_size % input_size == 0 else 0


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest2d_backward"),
    key=["N", "C", "IH", "IW", "OH", "OW"],
)
@triton.jit
def upsample_nearest2d_backward_kernel(
    ptr_go,
    ptr_gi,
    ptr_h_start,
    ptr_h_end,
    ptr_w_start,
    ptr_w_end,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    SCALE_H_INT: tl.constexpr,
    SCALE_W_INT: tl.constexpr,
    MAX_RANGE_H: tl.constexpr,
    MAX_RANGE_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NC_PER_BLOCK: tl.constexpr,
):
    pid_spatial = tl.program_id(axis=0)
    pid_nc_grp = tl.program_id(axis=1)

    idx = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    iw = idx % IW
    ih = idx // IW % IH
    mask = idx < IH * IW

    NC = N * C
    nc_grp_stride = tl.num_programs(axis=1)
    nc_iter = pid_nc_grp * NC_PER_BLOCK

    while nc_iter < NC:
        for nc_off in range(NC_PER_BLOCK):
            nc = nc_iter + nc_off
            if nc < NC:
                grad = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

                if SCALE_H_INT > 0:
                    oh_base = SCALE_H_INT * ih
                    ow_base = SCALE_W_INT * iw
                    for dh in range(SCALE_H_INT):
                        oh = oh_base + dh
                        row_off = nc * OH * OW + oh * OW
                        for dw in range(SCALE_W_INT):
                            ow = ow_base + dw
                            active = mask & (ow < OW)
                            g = tl.load(
                                ptr_go + row_off + ow, mask=active, other=0.0
                            )
                            grad += g.to(tl.float32)
                else:
                    oh_s = tl.load(ptr_h_start + ih, mask=mask, other=0)
                    oh_e = tl.load(ptr_h_end + ih, mask=mask, other=0)
                    ow_s = tl.load(ptr_w_start + iw, mask=mask, other=0)
                    ow_e = tl.load(ptr_w_end + iw, mask=mask, other=0)
                    for dh in range(MAX_RANGE_H):
                        oh = oh_s + dh
                        valid_h = dh < (oh_e - oh_s)
                        for dw in range(MAX_RANGE_W):
                            ow = ow_s + dw
                            valid_w = dw < (ow_e - ow_s)
                            active = valid_h & valid_w & mask
                            g = tl.load(
                                ptr_go + nc * OH * OW + oh * OW + ow,
                                mask=active,
                                other=0.0,
                            )
                            grad += g.to(tl.float32)

                tl.store(ptr_gi + nc * IH * IW + ih * IW + iw, grad, mask=mask)

        nc_iter += nc_grp_stride * NC_PER_BLOCK


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int],
    input_size: Tuple[int, int, int, int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D BACKWARD")
    assert grad_output.device.type == device
    assert grad_output.ndim == 4, "grad_output must be 4-D"

    OH, OW = output_size
    N, C, IH, IW = input_size

    if N * C * IH * IW == 0 or OH * OW == 0:
        return torch.zeros(
            (N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype
        )

    scale_h_int = _is_integer_scale(OH, IH, scales_h)
    scale_w_int = _is_integer_scale(OW, IW, scales_w)

    if scale_h_int > 0 and scale_w_int > 0:
        max_range_h, max_range_w = scale_h_int, scale_w_int
        h_start = h_end = w_start = w_end = grad_output
    else:
        with torch_device_fn.device(grad_output.device):
            h_start, h_end = _build_range_lut(OH, IH, scales_h, grad_output.device)
            w_start, w_end = _build_range_lut(OW, IW, scales_w, grad_output.device)
        max_range_h = max(int((h_end - h_start).max().item()), 1)
        max_range_w = max(int((w_end - w_start).max().item()), 1)

    # allocate output
    grad_input = torch.empty(
        (N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype
    )

    total_elements = IH * IW
    grid = lambda META: (
        triton.cdiv(total_elements, META["BLOCK_SIZE"]),
        triton.cdiv(N * C, META["NC_PER_BLOCK"]),
    )

    with torch_device_fn.device(grad_output.device):
        upsample_nearest2d_backward_kernel[grid](
            grad_output.contiguous(),
            grad_input,
            h_start,
            h_end,
            w_start,
            w_end,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            SCALE_H_INT=scale_h_int,
            SCALE_W_INT=scale_w_int,
            MAX_RANGE_H=max_range_h,
            MAX_RANGE_W=max_range_w,
        )

    return grad_input
