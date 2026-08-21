import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _gather_kernel(
    grad_output_ptr,
    grad_input_ptr,
    OH,
    OW,
    H,
    W,
    PAD_LEFT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    OH_OW_STRIDE: tl.constexpr,
    H_W_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNKS_PER_ROW: tl.constexpr,
    MAX_H_COUNT: tl.constexpr,
    MAX_W_COUNT: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = pid // CHUNKS_PER_ROW
    chunk = pid % CHUNKS_PER_ROW

    nc = row_id // H
    ih = row_id % H

    base_iw = chunk * BLOCK_SIZE
    iw = base_iw + tl.arange(0, BLOCK_SIZE)
    mask = iw < W

    oh_start = tl.where(
        H == 1,
        0,
        tl.where(ih == 0, 0, tl.where(ih == H - 1, H - 1 + PAD_TOP, ih + PAD_TOP)),
    )
    oh_end = tl.where(
        H == 1,
        OH - 1,
        tl.where(ih == 0, PAD_TOP, tl.where(ih == H - 1, OH - 1, ih + PAD_TOP)),
    )
    oh_count = oh_end - oh_start + 1

    ow_start = tl.where(
        W == 1,
        0,
        tl.where(iw == 0, 0, tl.where(iw == W - 1, W - 1 + PAD_LEFT, iw + PAD_LEFT)),
    )
    ow_end = tl.where(
        W == 1,
        OW - 1,
        tl.where(iw == 0, PAD_LEFT, tl.where(iw == W - 1, OW - 1, iw + PAD_LEFT)),
    )
    ow_count = ow_end - ow_start + 1

    out_base = nc * OH_OW_STRIDE
    in_base = nc * H_W_STRIDE + ih * W

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for dh in tl.static_range(MAX_H_COUNT):
        oh = oh_start + dh
        h_valid = (dh < oh_count) & mask
        for dw in tl.static_range(MAX_W_COUNT):
            ow = ow_start + dw
            w_valid = (dw < ow_count) & h_valid
            grad = tl.load(
                grad_output_ptr + out_base + oh * OW + ow,
                mask=w_valid,
                other=0.0,
            )
            acc += grad

    tl.store(grad_input_ptr + in_base + iw, acc, mask=mask)


def _replication_pad2d_backward_impl(
    grad_output: torch.Tensor, self: torch.Tensor, padding, *, out: torch.Tensor = None
) -> torch.Tensor:
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) != 4:
            raise ValueError(
                "padding must be a sequence of 4 integers: "
                "(pad_left, pad_right, pad_top, pad_bottom)"
            )
        pad_left, pad_right, pad_top, pad_bottom = map(int, padding)
    else:
        raise TypeError(f"Unexpected padding type: {type(padding)}")

    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("Padding values must be non-negative")

    is_3d = self.ndim == 3
    if is_3d:
        C, H, W = self.shape
        N = 1
        grad_output_4d = grad_output.view(
            1, C, grad_output.shape[-2], grad_output.shape[-1]
        )
    elif self.ndim == 4:
        N, C, H, W = self.shape
        grad_output_4d = grad_output.contiguous()
    else:
        raise ValueError(
            "replication_pad2d_backward expects a 3D (C, H, W) or 4D (N, C, H, W) input"
        )

    OH = H + pad_top + pad_bottom
    OW = W + pad_left + pad_right

    expected_grad_shape = (N, C, OH, OW)
    if tuple(grad_output_4d.shape) != expected_grad_shape:
        raise ValueError(
            f"grad_output shape {tuple(grad_output_4d.shape)} does not match "
            f"expected shape {expected_grad_shape}"
        )

    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        result = grad_output_4d.contiguous().to(self.dtype)
        if is_3d:
            result = result.view(C, H, W)
        if out is not None:
            out.copy_(result)
            return out
        return result

    output_dtype = self.dtype
    nc_total = N * C
    total_input_elements = nc_total * H * W

    oh_ow_stride = OH * OW
    h_w_stride = H * W

    MAX_BLOCK = 256

    if W >= MAX_BLOCK:
        block_size = MAX_BLOCK
    else:
        block_size = max(W, 32)

    chunks_per_row = triton.cdiv(W, block_size)
    grid = (nc_total * H * chunks_per_row,)

    if H == 1:
        max_h_count = pad_top + pad_bottom + 1
    else:
        max_h_count = max(pad_top, pad_bottom) + 1

    if W == 1:
        max_w_count = pad_left + pad_right + 1
    else:
        max_w_count = max(pad_left, pad_right) + 1

    if out is not None:
        accum = out
    else:
        accum = torch.empty((N, C, H, W), device=self.device, dtype=output_dtype)

    with torch_device_fn.device(self.device):
        if total_input_elements > 0:
            _gather_kernel[grid](
                grad_output_4d,
                accum,
                OH,
                OW,
                H,
                W,
                pad_left,
                pad_top,
                oh_ow_stride,
                h_w_stride,
                block_size,
                chunks_per_row,
                max_h_count,
                max_w_count,
            )

    if is_3d:
        accum = accum.view(C, H, W)
    return accum


def replication_pad2d_backward(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    logger.debug("GEMS_HYGON REPLICATION_PAD2D_BACKWARD")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=None)


def replication_pad2d_backward_grad_input(
    grad_output: torch.Tensor, self: torch.Tensor, padding, *, grad_input: torch.Tensor
) -> torch.Tensor:
    logger.debug("GEMS_HYGON REPLICATION_PAD2D_BACKWARD_GRAD_INPUT")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=grad_input)
