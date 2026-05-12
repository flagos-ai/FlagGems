import torch
import triton
import triton.language as tl


@triton.jit
def upsample_nearest3d_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    scale_d,
    scale_h,
    scale_w,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Unravel offsets into (n, c, od, oh, ow) for an output tensor of shape [N, C, OD, OH, OW]
    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    tmp = tmp // OD
    c = tmp % C
    n = tmp // C

    # Compute nearest input indices
    od_f = od.to(tl.float32)
    oh_f = oh.to(tl.float32)
    ow_f = ow.to(tl.float32)

    id_src = tl.minimum((od_f * scale_d).to(tl.int32), ID - 1)
    ih_src = tl.minimum((oh_f * scale_h).to(tl.int32), IH - 1)
    iw_src = tl.minimum((ow_f * scale_w).to(tl.int32), IW - 1)

    # Compute input/output offsets using strides
    in_offset = (
        n * in_stride_n
        + c * in_stride_c
        + id_src * in_stride_d
        + ih_src * in_stride_h
        + iw_src * in_stride_w
    )
    out_offset = (
        n * out_stride_n
        + c * out_stride_c
        + od * out_stride_d
        + oh * out_stride_h
        + ow * out_stride_w
    )

    vals = tl.load(in_ptr + in_offset, mask=mask, other=0)
    tl.store(out_ptr + out_offset, vals, mask=mask)


def _ensure_5d_input(x: torch.Tensor):
    if x.dim() != 5:
        raise ValueError(
            f"Expected 5D input [N, C, D, H, W], but got shape {tuple(x.shape)}"
        )
    return x


def _normalize_output_size(output_size):
    if output_size is None:
        return None
    if isinstance(output_size, torch.Size):
        output_size = tuple(int(s) for s in output_size)
    elif isinstance(output_size, (list, tuple)):
        output_size = tuple(int(s) for s in output_size)
    else:
        raise ValueError("output_size must be a sequence of 3 integers or torch.Size")
    if len(output_size) != 3:
        raise ValueError("output_size must have length 3: (out_d, out_h, out_w)")
    return output_size


def _normalize_scale_factors(scales):
    if scales is None:
        return None
    if isinstance(scales, (list, tuple)):
        if len(scales) != 3:
            raise ValueError(
                "scale_factors must have length 3: (scale_d, scale_h, scale_w)"
            )
        return tuple(float(s) if s is not None else None for s in scales)
    else:
        raise ValueError("scale_factors must be a sequence of 3 floats")


def _compute_out_size_and_kernel_scales(ID, IH, IW, output_size, scales_tuple):
    # Returns (OD, OH, OW, kscale_d, kscale_h, kscale_w)
    # kscale_* is the multiplier used as: src_idx = floor(out_idx * kscale_*)
    if output_size is not None:
        OD, OH, OW = int(output_size[0]), int(output_size[1]), int(output_size[2])
        if OD <= 0 or OH <= 0 or OW <= 0:
            raise ValueError("Output sizes must be positive")
        # When output_size is given, kscale = input_size / output_size
        kscale_d = float(ID) / float(OD)
        kscale_h = float(IH) / float(OH)
        kscale_w = float(IW) / float(OW)
    else:
        sd, sh, sw = scales_tuple
        if sd is None or sh is None or sw is None:
            raise ValueError(
                "All scale factors (scale_d, scale_h, scale_w) must be provided when output_size is None"
            )
        if sd <= 0.0 or sh <= 0.0 or sw <= 0.0:
            raise ValueError("Scale factors must be positive")
        OD = int(torch.floor(torch.tensor(ID * sd)).item())
        OH = int(torch.floor(torch.tensor(IH * sh)).item())
        OW = int(torch.floor(torch.tensor(IW * sw)).item())
        if OD <= 0 or OH <= 0 or OW <= 0:
            raise ValueError("Computed output sizes must be positive")
        # When scale_factors are given, src_idx = floor(out_idx / scale) = floor(out_idx * (1/scale))
        kscale_d = 1.0 / float(sd)
        kscale_h = 1.0 / float(sh)
        kscale_w = 1.0 / float(sw)
    return OD, OH, OW, kscale_d, kscale_h, kscale_w


def _launch_upsample_nearest3d(
    input: torch.Tensor,
    output: torch.Tensor,
    kscale_d: float,
    kscale_h: float,
    kscale_w: float,
):
    N, C, ID, IH, IW = input.shape
    OD, OH, OW = output.shape[2], output.shape[3], output.shape[4]

    in_strides = input.stride()
    out_strides = output.stride()

    total = N * C * OD * OH * OW
    if total == 0:
        return output

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)

    upsample_nearest3d_kernel[grid](
        input,
        output,
        N,
        C,
        ID,
        IH,
        IW,
        OD,
        OH,
        OW,
        in_strides[0],
        in_strides[1],
        in_strides[2],
        in_strides[3],
        in_strides[4],
        out_strides[0],
        out_strides[1],
        out_strides[2],
        out_strides[3],
        out_strides[4],
        float(kscale_d),
        float(kscale_h),
        float(kscale_w),
        total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def upsample_nearest3d(
    input: torch.Tensor, output_size=None, scales_d=None, scales_h=None, scales_w=None
):
    x = _ensure_5d_input(input)
    output_size = _normalize_output_size(output_size)
    scales_tuple = None
    if output_size is None:
        scales_tuple = (
            None if scales_d is None else float(scales_d),
            None if scales_h is None else float(scales_h),
            None if scales_w is None else float(scales_w),
        )
    N, C, ID, IH, IW = x.shape
    OD, OH, OW, ksd, ksh, ksw = _compute_out_size_and_kernel_scales(
        ID, IH, IW, output_size, scales_tuple
    )
    out = torch.empty(
        (N, C, OD, OH, OW), dtype=x.dtype, device=x.device, layout=x.layout
    )
    return _launch_upsample_nearest3d(x, out, ksd, ksh, ksw)


def upsample_nearest3d_vec(input: torch.Tensor, output_size=None, scale_factors=None):
    x = _ensure_5d_input(input)
    output_size = _normalize_output_size(output_size)
    scales_tuple = None
    if output_size is None:
        scales_tuple = _normalize_scale_factors(scale_factors)
    N, C, ID, IH, IW = x.shape
    OD, OH, OW, ksd, ksh, ksw = _compute_out_size_and_kernel_scales(
        ID, IH, IW, output_size, scales_tuple
    )
    out = torch.empty(
        (N, C, OD, OH, OW), dtype=x.dtype, device=x.device, layout=x.layout
    )
    return _launch_upsample_nearest3d(x, out, ksd, ksh, ksw)


def upsample_nearest3d_out(
    input: torch.Tensor,
    output_size=None,
    scales_d=None,
    scales_h=None,
    scales_w=None,
    out: torch.Tensor = None,
):
    x = _ensure_5d_input(input)
    output_size = _normalize_output_size(output_size)
    scales_tuple = None
    if output_size is None:
        scales_tuple = (
            None if scales_d is None else float(scales_d),
            None if scales_h is None else float(scales_h),
            None if scales_w is None else float(scales_w),
        )
    N, C, ID, IH, IW = x.shape
    OD, OH, OW, ksd, ksh, ksw = _compute_out_size_and_kernel_scales(
        ID, IH, IW, output_size, scales_tuple
    )

    if out is None:
        raise ValueError("Argument 'out' must be provided for upsample_nearest3d_out")
    if out.device != x.device or out.dtype != x.dtype:
        raise ValueError(
            "Output tensor 'out' must have the same device and dtype as input"
        )
    expected_shape = (N, C, OD, OH, OW)
    if tuple(out.shape) != expected_shape:
        raise ValueError(
            f"Output tensor 'out' must have shape {expected_shape}, but got {tuple(out.shape)}"
        )

    _launch_upsample_nearest3d(x, out, ksd, ksh, ksw)
    return out
