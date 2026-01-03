import math
import torch
import triton
import triton.language as tl


@triton.jit
def _upsample_nearest_exact1d_kernel(
    in_ptr,
    out_ptr,
    N, C, IW, OW,
    sN_in, sC_in, sW_in,
    sN_out, sC_out, sW_out,
    use_scales: tl.constexpr,
    scale_w,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_nc = tl.program_id(1)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = offs_w < OW

    # Compute (n, c) from flattened plane index
    nc = pid_nc
    n = nc // C
    c = nc - n * C

    base_in = n * sN_in + c * sC_in
    base_out = n * sN_out + c * sC_out

    # Compute source indices iw for each output index ow
    iw = tl.zeros([BLOCK_W], dtype=tl.int32)
    if use_scales:
        ow_f = offs_w.to(tl.float32)
        iw_f = tl.floor(ow_f / scale_w)
        iw = iw_f.to(tl.int32)
    else:
        iw = (offs_w * IW) // OW
    iw = tl.minimum(iw, IW - 1)

    in_ptrs = in_ptr + base_in + iw * sW_in
    x = tl.load(in_ptrs, mask=mask)

    out_ptrs = out_ptr + base_out + offs_w * sW_out
    tl.store(out_ptrs, x, mask=mask)


def _parse_size_1d(val):
    if val is None:
        return None
    if isinstance(val, torch.Size):
        return int(val[-1]) if len(val) > 0 else None
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return None
        return int(val[-1])
    return int(val)


def _parse_scale_1d(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return None
        return float(val[-1])
    return float(val)


def _compute_out_w(iw, output_size, scale):
    if output_size is not None:
        return int(output_size)
    if scale is None:
        raise ValueError("Either output_size or scale must be provided for _upsample_nearest_exact1d.")
    # Follow common convention: OW = floor(IW * scale)
    return int(math.floor(iw * scale))


def _launch_upsample_nearest_exact1d_kernel(input, out, output_size=None, scale=None):
    if input.ndim != 3:
        raise ValueError(f"_upsample_nearest_exact1d expects a 3D tensor (N, C, W); got shape {tuple(input.shape)}")
    if not input.is_cuda or not out.is_cuda:
        # Fallback to the native operator on CPU or non-CUDA devices
        return torch.ops.aten._upsample_nearest_exact1d(input, [out.shape[-1]], [scale] if scale is not None else None)

    N, C, IW = input.shape
    OW = out.shape[-1]

    sN_in, sC_in, sW_in = input.stride()
    sN_out, sC_out, sW_out = out.stride()

    BLOCK_W = 256
    grid = (triton.cdiv(OW, BLOCK_W), N * C)

    use_scales = scale is not None and output_size is None
    scale_w = float(scale) if use_scales else 1.0

    _upsample_nearest_exact1d_kernel[grid](
        input, out,
        N, C, IW, OW,
        sN_in, sC_in, sW_in,
        sN_out, sC_out, sW_out,
        use_scales=use_scales,
        scale_w=scale_w,
        BLOCK_W=BLOCK_W,
    )
    return out


def _extract_io_and_params(args, kwargs, expect_out=False):
    # Extract input tensor
    in_t = kwargs.get("input", None)
    if in_t is None:
        in_t = kwargs.get("self", None)
    if in_t is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
        in_t = args[0]
        args = args[1:]
    if in_t is None or not isinstance(in_t, torch.Tensor):
        raise ValueError("Input tensor not found for _upsample_nearest_exact1d.")

    # Extract output_size / scales from kwargs or remaining args
    output_size = kwargs.get("output_size", kwargs.get("size", kwargs.get("output_size_list", None)))
    scales = kwargs.get("scale_factor", kwargs.get("scales", kwargs.get("scale_factors", kwargs.get("scale", None))))

    # If positional arguments contain size and/or scales
    # Try to interpret next positional as output_size if present and not a tensor
    pos = 0
    if output_size is None and pos < len(args) and not isinstance(args[pos], torch.Tensor):
        output_size = args[pos]
        pos += 1
    if scales is None and pos < len(args) and not isinstance(args[pos], torch.Tensor):
        scales = args[pos]
        pos += 1

    out_t = None
    if expect_out:
        out_t = kwargs.get("out", None)
        if out_t is None:
            # find last tensor among remaining args as out
            for a in reversed(args):
                if isinstance(a, torch.Tensor):
                    out_t = a
                    break
        if out_t is None:
            raise ValueError("Output tensor 'out' not found for _upsample_nearest_exact1d_out.")

    # Normalize single-dim size and scale
    out_w = _parse_size_1d(output_size)
    scale_w = _parse_scale_1d(scales)

    return in_t, out_t, out_w, scale_w


def _prepare_out_tensor(in_t, out_w, scale_w, dtype=None, device=None):
    N, C, IW = in_t.shape
    OW = _compute_out_w(IW, out_w, scale_w)
    if OW < 0:
        raise ValueError("Output width must be non-negative.")
    if dtype is None:
        dtype = in_t.dtype
    if device is None:
        device = in_t.device
    return torch.empty((N, C, OW), dtype=dtype, device=device)


def _upsample_nearest_exact1d(*args, **kwargs):
    in_t, _, out_w, scale_w = _extract_io_and_params(args, kwargs, expect_out=False)
    out_t = _prepare_out_tensor(in_t, out_w, scale_w)
    if out_t.numel() == 0:
        return out_t
    return _launch_upsample_nearest_exact1d_kernel(in_t, out_t, output_size=out_w, scale=scale_w)


def _upsample_nearest_exact1d_out(*args, **kwargs):
    in_t, out_t, out_w, scale_w = _extract_io_and_params(args, kwargs, expect_out=True)
    if out_t.ndim != 3:
        raise ValueError(f"Out tensor must be 3D (N, C, W); got shape {tuple(out_t.shape)}")
    # Validate that out_t has the correct computed width if parameters are provided
    expected_w = _compute_out_w(in_t.shape[-1], out_w, scale_w)
    if out_t.shape[-1] != expected_w:
        raise ValueError(f"Provided out tensor has width {out_t.shape[-1]} but expected {expected_w}.")
    if out_t.numel() == 0:
        return out_t
    return _launch_upsample_nearest_exact1d_kernel(in_t, out_t, output_size=out_w, scale=scale_w)


def _upsample_nearest_exact1d_vec(*args, **kwargs):
    # Treat vec the same as base variant, allowing list-like output_size/scales
    in_t, _, out_w, scale_w = _extract_io_and_params(args, kwargs, expect_out=False)
    out_t = _prepare_out_tensor(in_t, out_w, scale_w)
    if out_t.numel() == 0:
        return out_t
    return _launch_upsample_nearest_exact1d_kernel(in_t, out_t, output_size=out_w, scale=scale_w)