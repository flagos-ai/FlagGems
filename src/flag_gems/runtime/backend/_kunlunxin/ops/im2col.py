import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _parse_2tuple(value, name):
    if isinstance(value, int):
        return value, value
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return int(value[0]), int(value[1])
    raise ValueError(f"{name} must be an int or a tuple/list of two ints")


@libentry()
@triton.jit
def _im2col_affine_kernel(
    input_ptr,
    output_ptr,
    C,
    H,
    W: tl.constexpr,
    L: tl.constexpr,
    ROWS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    BLOCK: tl.constexpr,
    BND: tl.constexpr,
):
    """Affine fast path for stride==1 and OUT_W == W.

    For a fixed output row ``(batch, channel, kh, kw)`` the input address is
    ``base + pos`` with ``base = (n*C + c)*H*W + (kh*DH - PH)*W + (kw*DW - PW)``
    a *program-level scalar*, so the whole read is unit-stride and contiguous.
    ``pos`` is the flat location ``oh*OUT_W + ow`` in ``[0, L)``.

    ``BND`` is False only when ``PH == PW == 0 and KH == KW == 1``, in which case
    every read is in bounds and the kernel is a pure maskless block copy (this
    is the fastest form this backend supports).  Otherwise the padding lanes
    are masked out and zeroed explicitly, because ``tl.load(other=0.0)`` is not
    honoured here.
    """
    pid_row = ext.program_id(0).to(tl.int32)
    pid_loc = ext.program_id(1).to(tl.int32)
    n = pid_row // ROWS
    row = pid_row % ROWS
    c = row // (KH * KW)
    kp = row % (KH * KW)
    kh = kp // KW
    kw = kp % KW
    loc = pid_loc * BLOCK + tl.arange(0, BLOCK)
    base = (n * C + c) * H * W + (kh * DH - PH) * W + (kw * DW - PW)
    if BND:
        ih = loc // W - PH + kh * DH
        iw = loc % W - PW + kw * DW
        m = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        values = tl.load(input_ptr + base + loc, mask=m, other=0.0)
        values = tl.where(m, values, 0.0)
    else:
        values = tl.load(input_ptr + base + loc)
    tl.store(output_ptr + pid_row * L + loc, values)


@libentry()
@triton.jit
def _im2col_kernel(
    input_ptr,
    output_ptr,
    total,
    C,
    H,
    W,
    KH,
    KW,
    DH,
    DW,
    PH,
    PW,
    SH,
    SW,
    OUT_W,
    ROWS,
    L,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0).to(tl.int32) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < total

    output_pos = offsets % L
    row_batch = offsets // L
    row = row_batch % ROWS
    batch = row_batch // ROWS

    kernel_area = KH * KW
    channel = row // kernel_area
    kernel_pos = row % kernel_area
    kernel_h = kernel_pos // KW
    kernel_w = kernel_pos % KW
    output_h = output_pos // OUT_W
    output_w = output_pos % OUT_W
    input_h = output_h * SH - PH + kernel_h * DH
    input_w = output_w * SW - PW + kernel_w * DW
    in_bounds = (input_h >= 0) & (input_h < H) & (input_w >= 0) & (input_w < W)

    input_offsets = ((batch * C + channel) * H + input_h) * W + input_w
    values = tl.load(input_ptr + input_offsets, mask=valid & in_bounds, other=0.0)
    values = tl.where(in_bounds, values, 0.0)
    tl.store(output_ptr + offsets, values, mask=valid)


def im2col(input, kernel_size, dilation=1, padding=0, stride=1):
    logger.debug("GEMS_KUNLUNXIN IM2COL")
    was_unbatched = input.ndim == 3
    x = input.unsqueeze(0) if was_unbatched else input
    if x.ndim != 4:
        raise ValueError("im2col expects input of shape (N, C, H, W) or (C, H, W)")

    kernel_h, kernel_w = _parse_2tuple(kernel_size, "kernel_size")
    dilation_h, dilation_w = _parse_2tuple(dilation, "dilation")
    padding_h, padding_w = _parse_2tuple(padding, "padding")
    stride_h, stride_w = _parse_2tuple(stride, "stride")

    batch, channels, height, width = x.shape
    output_h = (
        height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)
    ) // stride_h + 1
    output_w = (
        width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)
    ) // stride_w + 1
    rows = channels * kernel_h * kernel_w
    locations = output_h * output_w
    output = torch.empty(
        (batch, rows, locations), dtype=input.dtype, device=input.device
    )
    total = output.numel()
    if total:
        x = x.contiguous()
        bounded = not (
            padding_h == 0 and padding_w == 0 and kernel_h == 1 and kernel_w == 1
        )
        affine = (
            stride_h == 1
            and stride_w == 1
            and output_w == width
            and locations == (1 << (locations.bit_length() - 1))
            and 256 <= locations <= 16384
            and (locations >= 512 or not bounded)
        )
        if affine:
            with torch_device_fn.device(input.device):
                _im2col_affine_kernel[(batch * rows,)](
                    x,
                    output,
                    channels,
                    height,
                    W=width,
                    L=locations,
                    ROWS=rows,
                    KH=kernel_h,
                    KW=kernel_w,
                    DH=dilation_h,
                    DW=dilation_w,
                    PH=padding_h,
                    PW=padding_w,
                    BLOCK=locations,
                    BND=bounded,
                    num_warps=min(16, max(2, locations // 512)),
                )
            return output.squeeze(0) if was_unbatched else output
        block = 2048 if total >= 4096 else 256
        with torch_device_fn.device(input.device):
            _im2col_kernel[(triton.cdiv(total, block),)](
                x,
                output,
                total,
                channels,
                height,
                width,
                kernel_h,
                kernel_w,
                dilation_h,
                dilation_w,
                padding_h,
                padding_w,
                stride_h,
                stride_w,
                output_w,
                rows,
                locations,
                BLOCK=block,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
    return output.squeeze(0) if was_unbatched else output
