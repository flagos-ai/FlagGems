import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


def fractional_max_pool2d_output_size(
    in_size: int,
    kernel_size: int,
    output_size: int | None,
) -> int:
    if output_size is None:
        raise ValueError("output_size must be provided")

    target = int(output_size)

    if target < 1:
        raise ValueError("output_size must be >= 1")
    if target > in_size - kernel_size + 1:
        raise ValueError(
            "output_size is too large for the given kernel_size and input size"
        )

    return target


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 4}, num_stages=3, num_warps=1),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 8}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 4}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=4),
    ],
    key=["out_h", "out_w", "kernel_h", "kernel_w"],
)
@triton.jit
def fractional_max_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    random_samples_ptr,
    # Input strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Random sample strides
    rs_stride_n,
    rs_stride_c,
    rs_stride_k,
    # Shapes
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Pooling parameters
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    # Meta
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

    active = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int64)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    rs_base = random_samples_ptr + n_idx * rs_stride_n + c_idx * rs_stride_c
    sample_w = tl.load(rs_base + 0 * rs_stride_k)
    sample_h = tl.load(rs_base + 1 * rs_stride_k)

    sample_dtype = sample_w.dtype

    alpha_h = tl.where(
        out_h > 1,
        tl.full((), in_h - kernel_h, sample_dtype)
        / tl.full((), out_h - 1, sample_dtype),
        tl.full((), 0.0, sample_dtype),
    )
    alpha_w = tl.where(
        out_w > 1,
        tl.full((), in_w - kernel_w, sample_dtype)
        / tl.full((), out_w - 1, sample_dtype),
        tl.full((), 0.0, sample_dtype),
    )

    base_h = tl.floor(sample_h * alpha_h)
    base_w = tl.floor(sample_w * alpha_w)

    h_offsets_f = h_out_offsets.to(sample_dtype)
    w_offsets_f = w_out_offsets.to(sample_dtype)

    row_tmp = tl.floor((h_offsets_f[:, None] + sample_h) * alpha_h) - base_h
    col_tmp = tl.floor((w_offsets_f[None, :] + sample_w) * alpha_w) - base_w

    row_start = tl.where(
        h_out_offsets[:, None] == (out_h - 1), in_h - kernel_h, row_tmp
    )
    col_start = tl.where(
        w_out_offsets[None, :] == (out_w - 1), in_w - kernel_w, col_tmp
    )
    row_start = tl.where(active, row_start, 0).to(tl.int32)
    col_start = tl.where(active, col_start, 0).to(tl.int32)

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = row_start + kh
            w_in = col_start + kw
            in_mask = active & (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
            input_offset = h_in * in_stride_h + w_in * in_stride_w
            current_val = tl.load(
                input_base_ptr + input_offset, mask=in_mask, other=min_val
            )
            current_idx = h_in * in_w + w_in

            if dtype == tl.bfloat16:
                current_val_f32 = current_val.to(tl.float32)
                max_val_acc_f32 = max_val_acc.to(tl.float32)
                current_is_nan = current_val_f32 != current_val_f32
                is_new_max = current_is_nan | (current_val_f32 > max_val_acc_f32)
            else:
                current_is_nan = current_val != current_val
                is_new_max = current_is_nan | (current_val > max_val_acc)

            max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
            max_idx_acc = tl.where(is_new_max & in_mask, current_idx, max_idx_acc)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_h * out_w
    output_block_ptr = (
        out_base_ptr + h_out_offsets[:, None] * out_w + w_out_offsets[None, :]
    )
    indices_block_ptr = (
        indices_base_ptr + h_out_offsets[:, None] * out_w + w_out_offsets[None, :]
    )

    tl.store(output_block_ptr, max_val_acc, mask=active)
    tl.store(indices_block_ptr, max_idx_acc, mask=active)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_OUT_H": 16, "BLOCK_OUT_W": 16}, num_warps=4),
        triton.Config({"BLOCK_OUT_H": 32, "BLOCK_OUT_W": 8}, num_warps=4),
        triton.Config({"BLOCK_OUT_H": 8, "BLOCK_OUT_W": 32}, num_warps=4),
        triton.Config({"BLOCK_OUT_H": 32, "BLOCK_OUT_W": 32}, num_warps=8),
        triton.Config({"BLOCK_OUT_H": 64, "BLOCK_OUT_W": 16}, num_warps=8),
        triton.Config({"BLOCK_OUT_H": 16, "BLOCK_OUT_W": 64}, num_warps=8),
    ],
    key=["out_h", "out_w", "kernel_h", "kernel_w"],
    reset_to_zero=["grad_input_ptr"],
)
@triton.jit
def fractional_max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    # shapes
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # pooling params
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    # tiling
    BLOCK_OUT_H: tl.constexpr,
    BLOCK_OUT_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(out_w, BLOCK_OUT_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_OUT_H + tl.arange(0, BLOCK_OUT_H)
    w_out_offsets = w_block_idx * BLOCK_OUT_W + tl.arange(0, BLOCK_OUT_W)

    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)

    grad_output_block_ptr = (
        grad_output_ptr
        + n_idx * out_stride_n
        + c_idx * out_stride_c
        + h_out_offsets[:, None] * out_stride_h
        + w_out_offsets[None, :] * out_stride_w
    )
    indices_block_ptr = (
        indices_ptr
        + pid_nc * out_h * out_w
        + h_out_offsets[:, None] * out_w
        + w_out_offsets[None, :]
    )

    grad_vals = tl.load(grad_output_block_ptr, mask=out_mask, other=0.0)
    max_indices = tl.load(indices_block_ptr, mask=out_mask, other=-1)

    in_bounds = max_indices >= 0
    target_h = max_indices // in_w
    target_w = max_indices % in_w
    target_mask = in_bounds & (target_h < in_h) & (target_w < in_w)

    grad_input_tile_ptr = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    input_offsets = target_h * in_stride_h + target_w * in_stride_w
    tl.atomic_add(
        grad_input_tile_ptr + input_offsets,
        grad_vals.to(grad_input_ptr.type.element_ty),
        mask=target_mask,
    )


def _parse_fractional_pool_params(kernel_size, output_size):
    def _as_pair(value, name):
        if isinstance(value, int):
            return value, value
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return int(value[0]), int(value[1])
        raise ValueError(f"Invalid {name}: {value}")

    kernel_h, kernel_w = _as_pair(kernel_size, "kernel_size")

    if output_size is not None:
        out_h, out_w = _as_pair(output_size, "output_size")
        if out_h < 1 or out_w < 1:
            raise ValueError("output_size values must be >= 1")
    else:
        raise ValueError("output_size must be provided")

    return {
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "output_h": out_h,
        "output_w": out_w,
    }


def fractional_max_pool2d(
    input: torch.Tensor,
    kernel_size,
    output_size,
    random_samples=None,
):
    logger.debug("GEMS FRACTIONAL_MAX_POOL2D FORWARD")
    input = input.contiguous()

    params = _parse_fractional_pool_params(kernel_size, output_size)
    kernel_h = params["kernel_h"]
    kernel_w = params["kernel_w"]

    in_n, in_c, in_h, in_w = input.shape
    out_h = params["output_h"] or fractional_max_pool2d_output_size(
        in_h, kernel_h, params["output_h"]
    )
    out_w = params["output_w"] or fractional_max_pool2d_output_size(
        in_w, kernel_w, params["output_w"]
    )

    def _normalize_random_samples(samples, dtype):
        if samples is None:
            samples = torch.rand((in_n, in_c, 2), dtype=dtype, device=input.device)
        else:
            samples = samples.to(device=input.device, dtype=dtype)
            expected_shape = (in_n, in_c, 2)
            if samples.shape == (in_n, 2):
                samples = samples[:, None, :].expand(-1, in_c, -1)
            elif samples.shape != expected_shape:
                raise ValueError(
                    f"random_samples must have shape (N, C, 2), but got {samples.shape}"
                )
        return samples

    if out_h == 0 or out_w == 0:
        output = torch.empty(
            (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
        )
        indices = torch.empty(
            (in_n, in_c, out_h, out_w), device=input.device, dtype=torch.int64
        )
        return output, indices

    random_samples = _normalize_random_samples(random_samples, torch.float64)
    if torch.any((random_samples < 0.0) | (random_samples >= 1.0)):
        raise ValueError("random_samples values must be in [0, 1)")

    random_samples = random_samples.contiguous()

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=torch.int64
    )

    if output.numel() == 0:
        return output, indices

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    fractional_max_pool2d_forward_kernel[grid](
        input,
        output,
        indices,
        random_samples,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        random_samples.stride(0),
        random_samples.stride(1),
        random_samples.stride(2),
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
    )

    return output, indices


def fractional_max_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_size,
    output_size,
    indices: torch.Tensor,
):
    logger.debug("GEMS FRACTIONAL_MAX_POOL2D BACKWARD")
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    params = _parse_fractional_pool_params(kernel_size, output_size)
    kernel_h = params["kernel_h"]
    kernel_w = params["kernel_w"]

    in_n, in_c, in_h, in_w = input.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    grad_input = torch.zeros_like(input, dtype=torch.float64)
    if grad_input.numel() == 0:
        return grad_input.to(grad_output.dtype)

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(out_h, meta["BLOCK_OUT_H"])
        * triton.cdiv(out_w, meta["BLOCK_OUT_W"]),
    )

    fractional_max_pool2d_backward_kernel[grid](
        grad_output,
        indices,
        grad_input,
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
    )

    return grad_input.to(grad_output.dtype)
