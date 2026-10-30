import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def pool3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    """Compute one spatial dimension of the 3-D max-pool output."""
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1
    return output_size


@libentry()
@triton.jit
def max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    total,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Forward kernel for 3-D max pooling (1-D scan based).

    Grid: (cdiv(N * C * D * H * W, BLOCK),)
    Each lane handles one flattened output position and scans the kd*kh*kw
    window.  The depth dimension is a small runtime loop (``kd``) while the
    two in-plane dimensions are ``tl.static_range``-unrolled: this removes the
    per-tap soft integer division (``k // ncfg`` / ``k % kernel_w``) that the
    previous single-flat-loop form had to do, which on XPU is the dominant
    cost (measured ~2.4x faster).  Unlike a fully unrolled kd*kh*kw peel the
    Kunlunxin compiler still only sees 9 unrolled taps (kh x kw), so it
    compiles in seconds.  Semantics match ATen: ties pick the first window
    position in (D, H, W) iteration order (a strictly-greater update keeps
    the earliest tap).  Out-of-window (padding) taps are clamped to an
    in-bounds address and the loaded value is discarded with ``tl.where``, so
    the loads stay unmasked (the XPU backend treats compound i1-masked loads
    as a slow path).
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    output_mask = offsets < total
    out_dhw = out_d * out_h * out_w
    nc_idx = offsets // out_dhw
    rem = offsets % out_dhw
    od = rem // (out_h * out_w)
    rem2 = rem % (out_h * out_w)
    oh = rem2 // out_w
    ow = rem2 % out_w
    in_hw = in_h * in_w
    d_base = od * stride_d - padding_d
    h_base = oh * stride_h - padding_h
    w_base = ow * stride_w - padding_w
    base_off = nc_idx * (in_d * in_hw)

    max_val = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    max_off = tl.zeros((BLOCK,), dtype=tl.int64)
    for kd in range(0, kernel_d):
        id_ = d_base + kd * dilation_d
        for kh in tl.static_range(0, kernel_h):
            ih_ = h_base + kh * dilation_h
            for kw in tl.static_range(0, kernel_w):
                iw_ = w_base + kw * dilation_w
                active = (
                    output_mask
                    & (id_ >= 0)
                    & (id_ < in_d)
                    & (ih_ >= 0)
                    & (ih_ < in_h)
                    & (iw_ >= 0)
                    & (iw_ < in_w)
                )
                id_s = tl.where(active, id_, 0)
                ih_s = tl.where(active, ih_, 0)
                iw_s = tl.where(active, iw_, 0)
                io = base_off + id_s * in_hw + ih_s * in_w + iw_s
                value = tl.load(input_ptr + io).to(tl.float32)
                value = tl.where(active, value, float("-inf"))
                is_max = value > max_val
                max_val = tl.where(is_max, value, max_val)
                max_off = tl.where(
                    is_max,
                    (id_s * in_hw + ih_s * in_w + iw_s).to(tl.int64),
                    max_off,
                )

    tl.store(output_ptr + offsets, max_val, mask=output_mask)
    tl.store(indices_ptr + offsets, max_off, mask=output_mask)


@triton.jit
def _extract32_wide(raw, sub):
    return (((raw >> (32 * sub)) & 0xFFFFFFFF).to(tl.uint32)).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _extract16_wide(raw, sub, IS_BF16: tl.constexpr):
    if IS_BF16:
        # bf16 == top 16 bits of fp32; reconstruct in the integer domain to
        # avoid the XPU bf16 in-kernel bitcast miscompile.
        return (((raw >> (16 * sub)) & 0xFFFF).to(tl.uint32) << 16).to(
            tl.float32, bitcast=True
        )
    else:
        return (((raw >> (16 * sub)) & 0xFFFF).to(tl.uint16)).to(
            tl.float16, bitcast=True
        ).to(tl.float32)


@libentry()
@triton.jit
def max_pool3d_forward_wide_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    total,
    total_words,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    IS_16BIT: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """W-packed forward kernel for 3-D max pooling.

    Same 1-D scan as ``max_pool3d_forward_kernel``, but the three ``kw`` taps
    are packed two-per-word in the contiguous W dimension (two fp32 per int64,
    or two fp16/bf16 per int32), so the per-tap W loads become a single
    contiguous word load.  Requires ``dilation_w == 1``, ``stride_w % 2 == 0``
    and ``in_w % 2 == 0`` (the packing alignment), enforced by the caller.
    """
    MIS: tl.constexpr = padding_w % 2
    NLOAD: tl.constexpr = (kernel_w + MIS + 1) // 2

    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    output_mask = offsets < total
    out_dhw = out_d * out_h * out_w
    nc_idx = offsets // out_dhw
    rem = offsets % out_dhw
    od = rem // (out_h * out_w)
    rem2 = rem % (out_h * out_w)
    oh = rem2 // out_w
    ow = rem2 % out_w
    in_hw = in_h * in_w
    d_base = od * stride_d - padding_d
    h_base = oh * stride_h - padding_h
    ow_base = ow * stride_w - padding_w
    nc_safe = tl.where(output_mask, nc_idx, 0)

    max_val = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    max_off = tl.zeros((BLOCK,), dtype=tl.int64)
    for kd in range(0, kernel_d):
        id_ = d_base + kd * dilation_d
        d_ok = (id_ >= 0) & (id_ < in_d)
        id_s = tl.where(d_ok, id_, 0)
        for kh in tl.static_range(0, kernel_h):
            ih_ = h_base + kh * dilation_h
            h_ok = d_ok & (ih_ >= 0) & (ih_ < in_h)
            ih_s = tl.where(h_ok, ih_, 0)
            e0 = (nc_safe * in_d + id_s) * in_hw + ih_s * in_w + ow_base
            b0 = e0 >> 1
            for k in tl.static_range(NLOAD):
                bk = tl.minimum(tl.maximum(b0 + k, 0), total_words - 1)
                if IS_16BIT:
                    raw = tl.load(input_ptr.to(tl.pointer_type(tl.int32)) + bk)
                else:
                    raw = tl.load(input_ptr.to(tl.pointer_type(tl.int64)) + bk)
                for s in tl.static_range(kernel_w):
                    if (s + MIS) // 2 != k:
                        pass
                    else:
                        if IS_16BIT:
                            v = _extract16_wide(raw, (s + MIS) % 2, IS_BF16)
                        else:
                            v = _extract32_wide(raw, (s + MIS) % 2)
                        e = ow_base + s
                        valid = output_mask & h_ok & (e >= 0) & (e < in_w)
                        is_new = valid & (v > max_val)
                        max_val = tl.where(is_new, v, max_val)
                        max_off = tl.where(
                            is_new,
                            (id_s * in_hw + ih_s * in_w + e).to(tl.int64),
                            max_off,
                        )

    tl.store(output_ptr + offsets, max_val, mask=output_mask)
    tl.store(indices_ptr + offsets, max_off, mask=output_mask)


@libentry()
@triton.jit
def max_pool3d_backward_scatter_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    out_numel_per_nc,
    in_numel_per_nc,
    BLOCK: tl.constexpr,
):
    """Backward kernel for 3-D max pooling (scatter-based).

    Grid: (N * C, cdiv(out_D * out_H * out_W, BLOCK))
    For each output position, load its pooled index and atomically add the
    gradient to the corresponding input position.  No kernel-window loop is
    needed, so the kernel compiles in seconds.  Indices are int32 flat
    offsets into the (D, H, W) spatial volume of the input.
    """
    nc_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    offsets = block_idx * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < out_numel_per_nc

    base_out = nc_idx * out_numel_per_nc
    grad_val = tl.load(grad_output_ptr + base_out + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    idx_val = tl.load(indices_ptr + base_out + offsets, mask=mask, other=-1).to(
        tl.int32
    )
    valid = mask & (idx_val >= 0)
    safe_idx = tl.where(valid, idx_val, 0).to(tl.int32)

    base_in = nc_idx * in_numel_per_nc
    tl.atomic_add(
        grad_input_ptr + base_in + safe_idx, grad_val, mask=valid, sem="relaxed"
    )


def _parse_pool3d_params(kernel_size, stride, padding, dilation):
    """Parse and validate 3-D pooling parameters.

    Each parameter can be an int (applied to all 3 spatial dims) or a
    3-element tuple/list (D, H, W).
    """

    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return tuple(param)
        raise ValueError(f"Invalid {name}: {param}")

    kd, kh, kw = _parse_param(kernel_size, "kernel_size")
    sd, sh, sw = _parse_param(stride, "stride", default=(kd, kh, kw))
    pd, ph, pw = _parse_param(padding, "padding", default=(0, 0, 0))
    dd, dh, dw = _parse_param(dilation, "dilation", default=(1, 1, 1))

    if sd <= 0 or sh <= 0 or sw <= 0:
        raise ValueError(f"stride must be positive, but got stride=({sd}, {sh}, {sw})")
    if pd < 0 or ph < 0 or pw < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({pd}, {ph}, {pw})"
        )
    if dd <= 0 or dh <= 0 or dw <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dd}, {dh}, {dw})"
        )

    return kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw


def max_pool3d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Compute 3-D max pooling, returning (output, indices).

    Indices are flat offsets into the (D, H, W) spatial volume of the input.
    """
    logger.debug("GEMS_KUNLUNXIN MAX_POOL3D_WITH_INDICES")
    input = input.contiguous()

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params

    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d = pool3d_output_size(in_d, kd, sd, pd, dd, ceil_mode)
    out_h = pool3d_output_size(in_h, kh, sh, ph, dh, ceil_mode)
    out_w = pool3d_output_size(in_w, kw, sw, pw, dw, ceil_mode)

    output = torch.empty(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=torch.int64
    )

    if output.numel() == 0:
        return output, indices

    total = output.numel()
    block = 128

    grid = (triton.cdiv(total, block),)

    # W-packed wide kernel halves the innermost (kw) loads by packing two
    # elements per word; it needs an even input width, an even W stride and
    # unit W dilation for the word alignment to hold.  Otherwise fall back to
    # the plain nested scan kernel.
    use_wide = (
        dw == 1
        and sw % 2 == 0
        and in_w % 2 == 0
        and input.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )

    with torch_device_fn.device(input.device):
        if use_wide:
            is_16bit = input.dtype != torch.float32
            is_bf16 = input.dtype == torch.bfloat16
            max_pool3d_forward_wide_kernel[grid](
                input,
                output,
                indices,
                total,
                input.numel() // 2,
                in_d,
                in_h,
                in_w,
                out_d,
                out_h,
                out_w,
                kd,
                kh,
                kw,
                sd,
                sh,
                sw,
                pd,
                ph,
                pw,
                dd,
                dh,
                is_16bit,
                is_bf16,
                block,
                num_warps=2,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
        else:
            max_pool3d_forward_kernel[grid](
                input,
                output,
                indices,
                total,
                in_d,
                in_h,
                in_w,
                out_d,
                out_h,
                out_w,
                kd,
                kh,
                kw,
                sd,
                sh,
                sw,
                pd,
                ph,
                pw,
                dd,
                dh,
                dw,
                block,
                num_warps=2,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )

    return output, indices


def max_pool3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    """Backward pass for 3-D max pooling (Kunlunxin)."""
    logger.debug("GEMS_KUNLUNXIN MAX_POOL3D_BACKWARD")
    original_dtype = grad_output.dtype
    grad_output = grad_output.to(torch.float32).contiguous()
    indices = indices.to(torch.int32).contiguous()

    # validate parameters (raises on malformed kernel_size/stride/padding/dilation)
    _parse_pool3d_params(kernel_size, stride, padding, dilation)

    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d, out_h, out_w = (
        grad_output.shape[2],
        grad_output.shape[3],
        grad_output.shape[4],
    )

    grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_input.numel() == 0:
        return grad_input.to(original_dtype)

    out_numel_per_nc = out_d * out_h * out_w
    in_numel_per_nc = in_d * in_h * in_w
    block = 1024
    grid = (in_n * in_c, triton.cdiv(out_numel_per_nc, block))

    with torch_device_fn.device(grad_input.device):
        max_pool3d_backward_scatter_kernel[grid](
            grad_output,
            indices,
            grad_input,
            out_numel_per_nc,
            in_numel_per_nc,
            block,
            num_warps=1,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    return grad_input.to(original_dtype)
