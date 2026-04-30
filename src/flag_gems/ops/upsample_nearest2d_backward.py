import logging
import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@triton.jit
def upsample_nearest2d_backward_x2_channels_last_kernel(
    ptr_grad_input,
    ptr_grad_output,
    N,
    C,
    IH,
    IW,
    grad_output_stride_n,
    grad_output_stride_h,
    grad_output_stride_w,
    grad_input_stride_n,
    grad_input_stride_h,
    grad_input_stride_w,
    BLOCK_SIZE: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if not USE_INT32_IDX:
        idx = idx.to(tl.int64)

    total = N * IH * IW * C
    mask = idx < total
    c = idx % C
    pixel = idx // C
    iw = pixel % IW
    ih = (pixel // IW) % IH
    n = pixel // (IH * IW)

    grad_output_base = (
        n * grad_output_stride_n
        + (ih * 2) * grad_output_stride_h
        + (iw * 2) * grad_output_stride_w
    )
    grad = tl.load(ptr_grad_output + grad_output_base + c, mask=mask, other=0.0).to(
        tl.float32
    )
    grad += tl.load(
        ptr_grad_output + grad_output_base + grad_output_stride_w + c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    grad += tl.load(
        ptr_grad_output + grad_output_base + grad_output_stride_h + c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    grad += tl.load(
        ptr_grad_output
        + grad_output_base
        + grad_output_stride_h
        + grad_output_stride_w
        + c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    grad_input_offset = (
        n * grad_input_stride_n
        + ih * grad_input_stride_h
        + iw * grad_input_stride_w
        + c
    )
    tl.store(ptr_grad_input + grad_input_offset, grad, mask=mask)


@triton.jit
def upsample_nearest2d_backward_x2_contiguous_kernel(
    ptr_grad_input,
    ptr_grad_output,
    N,
    C,
    IH,
    IW,
    BLOCK_SIZE: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if not USE_INT32_IDX:
        idx = idx.to(tl.int64)

    total = N * C * IH * IW
    mask = idx < total
    iw = idx % IW
    ih = (idx // IW) % IH
    nc = idx // (IH * IW)

    OW = IW * 2
    oh = ih * 2
    ow = iw * 2
    grad_output_base = nc * (IH * 2 * OW) + oh * OW + ow
    grad = tl.load(ptr_grad_output + grad_output_base, mask=mask, other=0.0).to(
        tl.float32
    )
    grad += tl.load(ptr_grad_output + grad_output_base + 1, mask=mask, other=0.0).to(
        tl.float32
    )
    grad += tl.load(ptr_grad_output + grad_output_base + OW, mask=mask, other=0.0).to(
        tl.float32
    )
    grad += tl.load(
        ptr_grad_output + grad_output_base + OW + 1, mask=mask, other=0.0
    ).to(tl.float32)
    tl.store(ptr_grad_input + idx, grad, mask=mask)


@triton.jit
def upsample_nearest2d_backward_kernel(
    ptr_grad_input,
    ptr_grad_output,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    scale_h,
    scale_w,
    grad_output_stride_n,
    grad_output_stride_c,
    grad_output_stride_h,
    grad_output_stride_w,
    grad_input_stride_n,
    grad_input_stride_c,
    grad_input_stride_h,
    grad_input_stride_w,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if not USE_INT32_IDX:
        idx = idx.to(tl.int64)

    total = N * C * IH * IW
    mask = idx < total

    iw = idx % IW
    ih = (idx // IW) % IH
    c = (idx // (IH * IW)) % C
    n = idx // (C * IH * IW)

    if SAME_H:
        oh_start = ih
        oh_end = ih + 1
    else:
        oh_start = tl.minimum(tl.ceil(ih.to(tl.float32) * scale_h).to(tl.int32), OH)
        oh_end = tl.minimum(tl.ceil((ih + 1).to(tl.float32) * scale_h).to(tl.int32), OH)
    if SAME_W:
        ow_start = iw
        ow_end = iw + 1
    else:
        ow_start = tl.minimum(tl.ceil(iw.to(tl.float32) * scale_w).to(tl.int32), OW)
        ow_end = tl.minimum(tl.ceil((iw + 1).to(tl.float32) * scale_w).to(tl.int32), OW)

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for h_offset in tl.static_range(0, BLOCK_H):
        oh = oh_start + h_offset
        h_mask = oh < oh_end
        for w_offset in tl.static_range(0, BLOCK_W):
            ow = ow_start + w_offset
            load_mask = mask & h_mask & (ow < ow_end)
            grad_output_offset = (
                n * grad_output_stride_n
                + c * grad_output_stride_c
                + oh * grad_output_stride_h
                + ow * grad_output_stride_w
            )
            grad = tl.load(
                ptr_grad_output + grad_output_offset, mask=load_mask, other=0.0
            ).to(tl.float32)
            acc += grad

    grad_input_offset = (
        n * grad_input_stride_n
        + c * grad_input_stride_c
        + ih * grad_input_stride_h
        + iw * grad_input_stride_w
    )

    tl.store(ptr_grad_input + grad_input_offset, acc, mask=mask)


def _compute_backward_scale(
    input_size: int, output_size: int, scale: Optional[float]
) -> float:
    if scale is not None:
        return float(torch.tensor(scale, dtype=torch.float32).item())
    return float(
        (
            torch.tensor(output_size, dtype=torch.float32)
            / torch.tensor(input_size, dtype=torch.float32)
        ).item()
    )


def _use_channels_last_output(input: torch.Tensor) -> bool:
    return (
        input.is_contiguous(memory_format=torch.channels_last) and input.stride(1) == 1
    )


def _range_block(scale: float, output_size: int) -> int:
    return max(1, min(output_size, math.ceil(scale)))


def _can_use_x2_fast_path(
    IH: int,
    IW: int,
    OH: int,
    OW: int,
    scales_h: Optional[float],
    scales_w: Optional[float],
) -> bool:
    return (
        OH == IH * 2
        and OW == IW * 2
        and (scales_h is None or scales_h == 2.0)
        and (scales_w is None or scales_w == 2.0)
    )


def _identity_copy(input: torch.Tensor) -> torch.Tensor:
    output = torch.empty_strided(
        input.size(), input.stride(), dtype=input.dtype, device=input.device
    )
    torch.ops.aten.copy_.default.redispatch(_FALLBACK_KEYSET, output, input, False)
    return output


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int],
    input_size: Tuple[int, int, int, int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D BACKWARD")
    assert grad_output.device.type == device
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    assert len(input_size) == 4, "The len of input_size must be 4"

    N, C, IH, IW = (int(dim) for dim in input_size)
    OH, OW = (int(dim) for dim in output_size)
    assert tuple(grad_output.shape) == (N, C, OH, OW)

    scale_h = _compute_backward_scale(IH, OH, scales_h)
    scale_w = _compute_backward_scale(IW, OW, scales_w)

    if (OH == IH and OW == IW) and (
        grad_output.is_contiguous() or _use_channels_last_output(grad_output)
    ):
        return _identity_copy(grad_output)

    memory_format = (
        torch.channels_last
        if _use_channels_last_output(grad_output)
        else torch.contiguous_format
    )
    grad_input = torch.empty(
        (N, C, IH, IW),
        device=grad_output.device,
        dtype=grad_output.dtype,
        memory_format=memory_format,
    )

    total_threads = N * C * IH * IW
    if total_threads == 0:
        return grad_input

    use_int32_idx = max(total_threads, grad_output.numel()) <= (2**31 - 1)
    use_x2_fast_path = _can_use_x2_fast_path(IH, IW, OH, OW, scales_h, scales_w)
    if _use_channels_last_output(grad_output) and use_x2_fast_path:

        def grid(meta):
            return (triton.cdiv(N * IH * IW * C, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(grad_output.device):
            upsample_nearest2d_backward_x2_channels_last_kernel[grid](
                grad_input,
                grad_output,
                N,
                C,
                IH,
                IW,
                grad_output.stride(0),
                grad_output.stride(2),
                grad_output.stride(3),
                grad_input.stride(0),
                grad_input.stride(2),
                grad_input.stride(3),
                USE_INT32_IDX=use_int32_idx,
                BLOCK_SIZE=256,
            )
        return grad_input

    if grad_output.is_contiguous() and grad_input.is_contiguous() and use_x2_fast_path:

        def grid(meta):
            return (triton.cdiv(total_threads, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(grad_output.device):
            upsample_nearest2d_backward_x2_contiguous_kernel[grid](
                grad_input,
                grad_output,
                N,
                C,
                IH,
                IW,
                USE_INT32_IDX=use_int32_idx,
                BLOCK_SIZE=256,
            )
        return grad_input

    def grid(meta):
        return (triton.cdiv(total_threads, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(grad_output.device):
        upsample_nearest2d_backward_kernel[grid](
            grad_input,
            grad_output,
            N,
            C,
            IH,
            IW,
            OH,
            OW,
            scale_h,
            scale_w,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            grad_output.stride(3),
            grad_input.stride(0),
            grad_input.stride(1),
            grad_input.stride(2),
            grad_input.stride(3),
            SAME_H=OH == IH,
            SAME_W=OW == IW,
            USE_INT32_IDX=use_int32_idx,
            BLOCK_H=_range_block(scale_h, OH),
            BLOCK_W=_range_block(scale_w, OW),
            BLOCK_SIZE=256,
        )
    return grad_input
