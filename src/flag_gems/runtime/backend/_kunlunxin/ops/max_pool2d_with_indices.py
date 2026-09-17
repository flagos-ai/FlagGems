import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


def max_pool2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
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
def max_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int32)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
            h_safe = tl.where(in_mask, h_in, 0)
            w_safe = tl.where(in_mask, w_in, 0)
            input_offset = h_safe * in_stride_h + w_safe * in_stride_w
            current_val = tl.load(
                input_base_ptr + input_offset, mask=in_mask, other=min_val
            )
            current_val = tl.where(in_mask, current_val, min_val)
            current_idx = h_safe * in_w + w_safe

            is_new_max = current_val > max_val_acc
            max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
            max_idx_acc = tl.where(is_new_max & in_mask, current_idx, max_idx_acc)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_h * out_w
    out_h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    output_block_ptr = (
        out_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )
    indices_block_ptr = (
        indices_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )

    out_mask = (out_h_offsets[:, None] < out_h) & (out_w_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc, mask=out_mask)


@libentry()
@triton.jit
def max_pool2d_forward_flat_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    total: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    output_mask = offsets < total
    out_hw: tl.constexpr = out_h * out_w
    nc_idx = offsets // out_hw
    rem = offsets % out_hw
    oh = rem // out_w
    ow = rem % out_w
    nc_safe = tl.where(output_mask, nc_idx, 0)

    max_val = tl.full((BLOCK,), float("-inf"), tl.float32)
    max_idx = tl.full((BLOCK,), -1, tl.int64)
    for kh in tl.static_range(kernel_h):
        for kw in tl.static_range(kernel_w):
            ih = oh * stride_h - padding_h + kh * dilation_h
            iw = ow * stride_w - padding_w + kw * dilation_w
            valid = output_mask & (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)
            ih_safe = tl.where(valid, ih, 0)
            iw_safe = tl.where(valid, iw, 0)
            input_offset = nc_safe * (in_h * in_w) + ih_safe * in_w + iw_safe
            value = tl.load(input_ptr + input_offset)
            value = tl.where(valid, value.to(tl.float32), float("-inf"))
            is_new_max = valid & (value > max_val)
            max_val = tl.where(is_new_max, value, max_val)
            max_idx = tl.where(is_new_max, ih_safe * in_w + iw_safe, max_idx)

    tl.store(output_ptr + offsets, max_val, mask=output_mask)
    tl.store(indices_ptr + offsets, max_idx, mask=output_mask)


@triton.jit
def _extract32_wide(raw, sub):
    return (((raw >> (32 * sub)) & 0xFFFFFFFF).to(tl.uint32)).to(
        tl.float32, bitcast=True
    )


@libentry()
@triton.jit
def max_pool2d_forward_wide_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    total,
    total64,
    in_h,
    in_w,
    out_h,
    out_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    MIS: tl.constexpr = padding_w % 2
    NLOAD: tl.constexpr = (kernel_w + MIS + 1) // 2

    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    output_mask = offsets < total
    out_hw = out_h * out_w
    nc_idx = offsets // out_hw
    rem = offsets % out_hw
    oh = rem // out_w
    ow = rem % out_w
    nc_safe = tl.where(output_mask, nc_idx, 0)
    p64 = input_ptr.to(tl.pointer_type(tl.int64))

    max_val = tl.full((BLOCK,), float("-inf"), tl.float32)
    max_idx = tl.full((BLOCK,), -1, tl.int64)
    ow_base = ow * stride_w - padding_w

    for kh in tl.static_range(kernel_h):
        ih = oh * stride_h - padding_h + kh
        h_ok = (ih >= 0) & (ih < in_h)
        ih_safe = tl.where(h_ok, ih, 0)
        e0 = (nc_safe * in_h + ih_safe) * in_w + ow_base
        b0 = e0 >> 1
        for k in tl.static_range(NLOAD):
            bk = tl.minimum(tl.maximum(b0 + k, 0), total64 - 1)
            raw = tl.load(p64 + bk)
            for s in tl.static_range(kernel_w):
                if (s + MIS) // 2 != k:
                    pass
                else:
                    v = _extract32_wide(raw, (s + MIS) % 2)
                    e = ow_base + s
                    valid = output_mask & h_ok & (e >= 0) & (e < in_w)
                    is_new = valid & (v > max_val)
                    max_val = tl.where(is_new, v, max_val)
                    max_idx = tl.where(is_new, ih_safe * in_w + e, max_idx)

    tl.store(output_ptr + offsets, max_val, mask=output_mask)
    tl.store(indices_ptr + offsets, max_idx, mask=output_mask)


@libentry()
@triton.jit
def max_pool2d_backward_flat_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    total: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    input_mask = offsets < total
    in_hw: tl.constexpr = in_h * in_w
    out_hw: tl.constexpr = out_h * out_w
    nc_idx = offsets // in_hw
    rem = offsets % in_hw
    ih = rem // in_w
    iw = rem % in_w
    nc_safe = tl.where(input_mask, nc_idx, 0)
    input_flat_idx = ih * in_w + iw

    grad_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kh in tl.static_range(kernel_h):
        for kw in tl.static_range(kernel_w):
            h_num = ih + padding_h - kh * dilation_h
            w_num = iw + padding_w - kw * dilation_w
            h_nonnegative = h_num >= 0
            w_nonnegative = w_num >= 0
            h_num_safe = tl.where(h_nonnegative, h_num, 0)
            w_num_safe = tl.where(w_nonnegative, w_num, 0)
            oh = h_num_safe // stride_h
            ow = w_num_safe // stride_w
            h_rem = h_num_safe - oh * stride_h
            w_rem = w_num_safe - ow * stride_w
            valid = (
                input_mask
                & h_nonnegative
                & w_nonnegative
                & (h_rem == 0)
                & (w_rem == 0)
                & (oh < out_h)
                & (ow < out_w)
            )
            oh_safe = tl.where(valid, oh, 0)
            ow_safe = tl.where(valid, ow, 0)
            out_offset = nc_safe * out_hw + oh_safe * out_w + ow_safe
            index_value = tl.load(indices_ptr + out_offset)
            match = valid & (index_value == input_flat_idx)
            grad_value = tl.load(grad_output_ptr + out_offset)
            grad_acc += tl.where(match, grad_value, 0.0)

    tl.store(grad_input_ptr + offsets, grad_acc, mask=input_mask)


@libentry()
@triton.jit
def max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    out_stride_nc,
    out_stride_h,
    out_stride_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_IN_H: tl.constexpr,
    BLOCK_IN_W: tl.constexpr,
):
    nc_idx = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_IN_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    h_in_offsets = h_block_idx * BLOCK_IN_H + tl.arange(0, BLOCK_IN_H)
    w_in_offsets = w_block_idx * BLOCK_IN_W + tl.arange(0, BLOCK_IN_W)

    current_input_flat_idx = h_in_offsets[:, None] * in_w + w_in_offsets[None, :]
    grad_acc = tl.zeros((BLOCK_IN_H, BLOCK_IN_W), dtype=tl.float32)

    indices_base_ptr = indices_ptr + nc_idx * out_stride_nc
    grad_output_base_ptr = grad_output_ptr + nc_idx * out_stride_nc

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            numerator_h = h_in_offsets[:, None] + padding_h - kh * dilation_h
            numerator_w = w_in_offsets[None, :] + padding_w - kw * dilation_w

            valid_map_mask = (numerator_h % stride_h == 0) & (
                numerator_w % stride_w == 0
            )
            h_out = numerator_h // stride_h
            w_out = numerator_w // stride_w
            out_bounds_mask = (
                (h_out >= 0) & (h_out < out_h) & (w_out >= 0) & (w_out < out_w)
            )
            load_mask = valid_map_mask & out_bounds_mask

            safe_h_out = tl.where(load_mask, h_out, 0)
            safe_w_out = tl.where(load_mask, w_out, 0)
            out_offsets = safe_h_out * out_stride_h + safe_w_out

            indices_block = tl.load(
                indices_base_ptr + out_offsets, mask=load_mask, other=-1
            )
            match_mask = indices_block == current_input_flat_idx

            grad_block = tl.load(
                grad_output_base_ptr + out_offsets, mask=match_mask, other=0.0
            )
            grad_acc += grad_block

    grad_input_base_ptr = grad_input_ptr + nc_idx * in_h * in_w
    grad_input_offsets = h_in_offsets[:, None] * in_w + w_in_offsets[None, :]
    store_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)
    tl.store(grad_input_base_ptr + grad_input_offsets, grad_acc, mask=store_mask)


def _parse_pool_params(kernel_size, stride, padding, dilation):
    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param
        if isinstance(param, (list, tuple)) and len(param) == 2:
            return param
        raise ValueError(f"Invalid {name}: {param}")

    kernel_h, kernel_w = _parse_param(kernel_size, "kernel_size")
    stride_h, stride_w = _parse_param(stride, "stride", default=(kernel_h, kernel_w))
    padding_h, padding_w = _parse_param(padding, "padding", default=(0, 0))
    dilation_h, dilation_w = _parse_param(dilation, "dilation", default=(1, 1))

    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(
            f"stride must be positive, but got stride=({stride_h}, {stride_w})"
        )
    if padding_h < 0 or padding_w < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({padding_h}, {padding_w})"
        )
    if dilation_h <= 0 or dilation_w <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dilation_h}, {dilation_w})"
        )

    return (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )


def max_pool2d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS_KUNLUNXIN MAX_POOL2D_WITH_INDICES")
    input = input.contiguous()

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
    (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    ) = params

    in_n, in_c, in_h, in_w = input.shape
    out_h = max_pool2d_output_size(
        in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
    )
    out_w = max_pool2d_output_size(
        in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=torch.int32
    )

    if output.numel() == 0:
        return output, indices

    total = output.numel()
    block = 1024
    grid = (triton.cdiv(total, block),)

    if (
        input.dtype == torch.float32
        and dilation_w == 1
        and stride_w % 2 == 0
        and in_w % 2 == 0
    ):
        with torch_device_fn.device(input.device):
            max_pool2d_forward_wide_kernel[grid](
                input,
                output,
                indices,
                total,
                input.numel() // 2,
                in_h,
                in_w,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                block,
                num_warps=1,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
        return output, indices

    with torch_device_fn.device(input.device):
        max_pool2d_forward_flat_kernel[grid](
            input,
            output,
            indices,
            total,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            block,
            num_warps=1,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    return output, indices


@libentry()
@triton.jit
def max_pool2d_backward_residue_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    n_count,
    in_hw,
    out_hw,
    R_H: tl.constexpr,
    R_W: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    P_H: tl.constexpr,
    P_W: tl.constexpr,
    D_H: tl.constexpr,
    D_W: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Residue-compressed gather backward (Kunlunxin/XPU).

    Grid: (cdiv(n_count, BLOCK), m_count, n_c); one program per (m, nc) row
    of the n_count output positions in this residue class.  Every input
    position belongs to exactly one residue class (R_H, R_W) =
    (ih % S_H, iw % S_W).  For a fixed (R_H, R_W) the kernel offset (kh, kw)
    can only be valid when ``(R_H + P_H - kh * D_H) % S_H == 0`` (and
    symmetrically for w); the condition is constexpr per launch, so the
    impossible taps are pruned at trace time (avg 2.25 of 9 taps for s = 2).
    For the surviving taps the load addresses are affine in the lane index
    (``n + c_w``), which is exactly the block-DMA-friendly pattern of the
    proven forward flat kernel; the generic per-input gather cannot express
    this (data-dependent clamps defeat the vectorizer and unmasked
    non-affine loads raise a device exception on this backend).

    Correctness: the argmax of output (oh, ow) is at input flat index
    ``h * in_w + w`` (identical to the ATen/XDNN convention and to the Gems
    forward), so the input-gradient contribution of ``(oh, ow)`` equals
    ``grad_output[oh, ow]`` when ``indices[oh, ow] == flat`` and 0 otherwise;
    summing over the (at most) K_H x K_W output positions of the residue class
    yields exactly the ATen backward semantics (an input that is the argmax of
    several windows accumulates all of them).  Boundary lanes (n + c_w out of
    [0, OUT_W)) contribute 0 through the value-level ``tl.where``; the loads
    are unconditional on clamped, provably in-bounds offsets (the pattern of
    the proven forward flat kernel -- no i1-masked load slow path).
    """
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = tl.program_id(1)
    nc = tl.program_id(2)
    n_mask = n < n_count
    ih = R_H + m * S_H
    iw = R_W + n * S_W
    input_flat_idx = ih * IN_W + iw

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kh in tl.static_range(K_H):
        if (R_H + P_H - kh * D_H) % S_H == 0:
            c_h = (R_H + P_H - kh * D_H) // S_H
            oh = m + c_h
            oh_c = tl.minimum(tl.maximum(oh, 0), OUT_H - 1)
            h_ok = (oh >= 0) & (oh < OUT_H)
            for kw in tl.static_range(K_W):
                if (R_W + P_W - kw * D_W) % S_W == 0:
                    c_w = (R_W + P_W - kw * D_W) // S_W
                    valid = n_mask & h_ok & (n + c_w >= 0) & (n + c_w < OUT_W)
                    n_safe = tl.where(valid, n, -c_w)
                    out_offset = nc * out_hw + oh_c * OUT_W + n_safe + c_w
                    index_value = tl.load(indices_ptr + out_offset)
                    match = valid & (index_value == input_flat_idx)
                    grad_value = tl.load(grad_output_ptr + out_offset)
                    acc += tl.where(match, grad_value, 0.0)

    tl.store(
        grad_input_ptr + nc * in_hw + ih * IN_W + iw,
        acc,
        mask=n_mask,
    )


def max_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    logger.debug("GEMS_KUNLUNXIN MAX_POOL2D_BACKWARD")
    original_dtype = grad_output.dtype
    grad_output = grad_output.to(torch.float32).contiguous()
    indices = indices.to(torch.int32).contiguous()

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
    (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    ) = params

    in_n, in_c, in_h, in_w = input.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    grad_input = torch.empty_like(input, dtype=torch.float32)

    if grad_input.numel() == 0:
        return grad_input.to(original_dtype)

    n_c = in_n * in_c
    in_hw = in_h * in_w
    out_hw = out_h * out_w

    with torch_device_fn.device(grad_input.device):
        if stride_h == 1 and stride_w == 1:
            total = grad_input.numel()
            block = 1024
            grid = (triton.cdiv(total, block),)
            max_pool2d_backward_flat_kernel[grid](
                grad_output,
                indices,
                grad_input,
                total,
                in_h,
                in_w,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                block,
                num_warps=1,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
        else:
            residue_classes = []
            for r_h in range(stride_h):
                if r_h >= in_h:
                    continue
                m_count = (in_h - r_h + stride_h - 1) // stride_h
                for r_w in range(stride_w):
                    if r_w >= in_w:
                        continue
                    n_count = (in_w - r_w + stride_w - 1) // stride_w
                    if n_count == 0 or m_count == 0:
                        continue
                    if (m_count >= 8 and n_count >= 2 * m_count) or m_count >= 64:
                        residue_classes.append((r_h, r_w, m_count, n_count))
            if residue_classes:
                for r_h, r_w, m_count, n_count in residue_classes:
                    block = min(max(1 << (n_count - 1).bit_length(), 4), 256)
                    max_pool2d_backward_residue_kernel[
                        (triton.cdiv(n_count, block), m_count, n_c)
                    ](
                        grad_output,
                        indices,
                        grad_input,
                        n_count,
                        in_hw,
                        out_hw,
                        R_H=r_h,
                        R_W=r_w,
                        S_H=stride_h,
                        S_W=stride_w,
                        P_H=padding_h,
                        P_W=padding_w,
                        D_H=dilation_h,
                        D_W=dilation_w,
                        K_H=kernel_h,
                        K_W=kernel_w,
                        IN_W=in_w,
                        OUT_H=out_h,
                        OUT_W=out_w,
                        BLOCK=block,
                        num_warps=1,
                        buffer_size_limit=2048,
                        isCloseVectorization=True,
                    )
            else:
                total = grad_input.numel()
                block = 1024
                grid = (triton.cdiv(total, block),)
                max_pool2d_backward_flat_kernel[grid](
                    grad_output,
                    indices,
                    grad_input,
                    total,
                    in_h,
                    in_w,
                    out_h,
                    out_w,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    dilation_h,
                    dilation_w,
                    block,
                    num_warps=1,
                    buffer_size_limit=2048,
                    isCloseVectorization=True,
                )

    return grad_input.to(original_dtype)


def max_pool2d_with_indices_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode: bool,
    indices: torch.Tensor,
):
    """Kunlunxin implementation of aten::max_pool2d_with_indices_backward."""
    logger.debug("GEMS_KUNLUNXIN MAX_POOL2D_WITH_INDICES_BACKWARD")
    return max_pool2d_backward(
        grad_output, self, indices, kernel_size, stride, padding, dilation, ceil_mode
    )
