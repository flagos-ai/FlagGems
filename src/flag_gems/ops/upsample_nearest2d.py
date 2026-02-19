import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _to_copy_fallback(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if x.dtype == dtype:
        return x
    return torch.ops.aten._to_copy.default.redispatch(
        _FALLBACK_KEYSET,
        x,
        dtype=dtype,
        layout=None,
        device=x.device,
        pin_memory=None,
        non_blocking=False,
        memory_format=torch.preserve_format,
    )


def _get_reciprocal_scale(
    input_size: int, output_size: int, scale: Optional[float]
) -> float:
    if scale is not None:
        return float(torch.tensor(1.0 / scale, dtype=torch.float32).item())
    return float(
        (
            torch.tensor(input_size, dtype=torch.float32)
            / torch.tensor(output_size, dtype=torch.float32)
        ).item()
    )


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest2d"), key=["N", "C", "OH", "OW"]
)
@triton.heuristics(runtime.get_heuristic_config("upsample_nearest2d"))
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
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)
    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ow = idx % OW
    oh = idx // OW % OH
    if SAME_H:
        ih = oh
    else:
        # tl.floor() cannot be found in 2.3.1, using int trunc
        ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)

    offset_o = (nc_iter * OH + oh) * OW + ow
    offset_i = (nc_iter * IH + ih) * IW + iw
    src_index_stride = nc_stride * IH * IW
    dst_index_stride = nc_stride * OH * OW
    while nc_iter < NC:
        data = tl.load(ptr_i + offset_i)
        tl.store(ptr_o + offset_o, data)
        ptr_i += src_index_stride
        ptr_o += dst_index_stride
        nc_iter += nc_stride


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
    reciprocal_scale_h = _get_reciprocal_scale(IH, OH, scales_h)
    reciprocal_scale_w = _get_reciprocal_scale(IW, OW, scales_w)
    # allocate output
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    total_threads = OH * OW
    grid = lambda META: (
        triton.cdiv(total_threads, META["BLOCK_SIZE"]),
        triton.cdiv(N * C, 4),
    )

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
            reciprocal_scale_h,
            reciprocal_scale_w,
        )
    return output


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int],
    input_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D BACKWARD")
    assert grad_output.device.type == device
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    assert len(input_size) == 4, "The len of input_size must be 4"

    OH, OW = int(output_size[0]), int(output_size[1])
    N, C, IH, IW = [int(dim) for dim in input_size]
    reciprocal_scale_h = _get_reciprocal_scale(IH, OH, scales_h)
    reciprocal_scale_w = _get_reciprocal_scale(IW, OW, scales_w)

    if scales_h is None and OH == IH:
        ih = torch.arange(OH, device=grad_output.device, dtype=torch.int64)
    else:
        oh = torch.arange(OH, device=grad_output.device, dtype=torch.float32)
        ih = torch.clamp((oh * reciprocal_scale_h).to(torch.int64), max=IH - 1)

    if scales_w is None and OW == IW:
        iw = torch.arange(OW, device=grad_output.device, dtype=torch.int64)
    else:
        ow = torch.arange(OW, device=grad_output.device, dtype=torch.float32)
        iw = torch.clamp((ow * reciprocal_scale_w).to(torch.int64), max=IW - 1)

    index = (ih[:, None] * IW + iw[None, :]).reshape(1, 1, -1)
    index = index.expand(N, C, -1)

    accum_dtype = (
        torch.float32
        if grad_output.dtype in (torch.float16, torch.bfloat16)
        else grad_output.dtype
    )
    grad_output_accum = _to_copy_fallback(grad_output, accum_dtype)
    grad_output_flat = grad_output_accum.reshape(N, C, -1)
    grad_input_flat = torch.zeros(
        (N, C, IH * IW), device=grad_output.device, dtype=accum_dtype
    )
    grad_input_flat.scatter_add_(2, index, grad_output_flat)
    grad_input = grad_input_flat.reshape(N, C, IH, IW)
    return _to_copy_fallback(grad_input, grad_output.dtype)
