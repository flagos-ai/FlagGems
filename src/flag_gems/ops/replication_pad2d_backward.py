import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _interior_copy_kernel(
    grad_output_ptr,
    grad_input_ptr,
    OW,
    H,
    W,
    PAD_LEFT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    INTERIOR_ELEMS: tl.constexpr,
    OH_OW_STRIDE: tl.constexpr,
    H_W_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Flat interior kernel for W < BLOCK_SIZE. Each element decodes its
    (nc, ih, iw) coordinate with constexpr stride arithmetic."""
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < INTERIOR_ELEMS

    iw = idx % W
    rest = idx // W
    ih = rest % H
    nc = rest // H

    oh = PAD_TOP + ih
    ow = PAD_LEFT + iw

    out_offset = nc * OH_OW_STRIDE + oh * OW + ow
    in_offset = nc * H_W_STRIDE + ih * W + iw

    grad = tl.load(grad_output_ptr + out_offset, mask=mask, other=0.0)
    tl.store(grad_input_ptr + in_offset, grad, mask=mask)


@triton.jit
def _interior_copy_kernel_row(
    grad_output_ptr,
    grad_input_ptr,
    OW,
    H,
    W,
    PAD_LEFT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    OH_OW_STRIDE: tl.constexpr,
    H_W_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNKS_PER_ROW: tl.constexpr,
):
    """Row-based interior kernel for W >= BLOCK_SIZE. Each block covers
    BLOCK_SIZE consecutive elements within a single row — no per-element division."""
    pid = tl.program_id(0)
    row_id = pid // CHUNKS_PER_ROW
    chunk = pid % CHUNKS_PER_ROW

    nc = row_id // H
    ih = row_id % H

    base_iw = chunk * BLOCK_SIZE
    iw = base_iw + tl.arange(0, BLOCK_SIZE)
    mask = iw < W

    oh = PAD_TOP + ih
    ow = PAD_LEFT + iw

    out_base = nc * OH_OW_STRIDE + oh * OW
    in_base = nc * H_W_STRIDE + ih * W

    grad = tl.load(grad_output_ptr + out_base + ow, mask=mask, other=0.0)
    tl.store(grad_input_ptr + in_base + iw, grad, mask=mask)


@triton.jit
def _edges_atomic_kernel(
    grad_output_ptr,
    grad_input_ptr,
    OH,
    OW,
    H,
    W,
    PAD_LEFT: tl.constexpr,
    PAD_RIGHT: tl.constexpr,
    PAD_TOP: tl.constexpr,
    PAD_BOTTOM: tl.constexpr,
    EDGE_ELEMS: tl.constexpr,
    NC_TOTAL: tl.constexpr,
    OH_OW_STRIDE: tl.constexpr,
    H_W_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Process edge regions (left/right/top/bottom padding) where multiple output
    positions may map to the same input position, requiring atomic_add."""
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < EDGE_ELEMS

    pad_rows = PAD_TOP + PAD_BOTTOM
    left_size = NC_TOTAL * OH * PAD_LEFT
    right_size = NC_TOTAL * OH * PAD_RIGHT

    in_left = idx < left_size
    in_right = (idx >= left_size) & (idx < left_size + right_size)

    safe_w = tl.maximum(1, W)
    safe_oh = tl.maximum(1, OH)
    safe_pad_l = tl.maximum(1, PAD_LEFT)
    safe_pad_r = tl.maximum(1, PAD_RIGHT)
    safe_pad_rows = tl.maximum(1, pad_rows)

    left_idx = idx
    left_ow = left_idx % safe_pad_l
    left_oh = (left_idx // safe_pad_l) % safe_oh
    left_nc = left_idx // (safe_pad_l * safe_oh)

    right_idx = idx - left_size
    right_ow = right_idx % safe_pad_r
    right_oh = (right_idx // safe_pad_r) % safe_oh
    right_nc = right_idx // (safe_pad_r * safe_oh)

    edge_col_idx = idx - left_size - right_size
    ecol = edge_col_idx % safe_w
    erow = (edge_col_idx // safe_w) % safe_pad_rows
    enc = edge_col_idx // (safe_w * safe_pad_rows)

    ow = tl.where(
        in_left, left_ow, tl.where(in_right, OW - PAD_RIGHT + right_ow, PAD_LEFT + ecol)
    )

    oh = tl.where(
        in_left,
        left_oh,
        tl.where(
            in_right,
            right_oh,
            tl.where(erow < PAD_TOP, erow, OH - PAD_BOTTOM + (erow - PAD_TOP)),
        ),
    )

    nc = tl.where(in_left, left_nc, tl.where(in_right, right_nc, enc))

    ih = oh - PAD_TOP
    ih = tl.maximum(0, tl.minimum(H - 1, ih))
    iw = ow - PAD_LEFT
    iw = tl.maximum(0, tl.minimum(W - 1, iw))

    out_offset = nc * OH_OW_STRIDE + oh * OW + ow
    in_offset = nc * H_W_STRIDE + ih * W + iw

    grad = tl.load(grad_output_ptr + out_offset, mask=mask, other=0.0)
    tl.atomic_add(grad_input_ptr + in_offset, grad, mask=mask)


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
    interior_elements = nc_total * H * W
    edge_elements = nc_total * OH * OW - interior_elements

    oh_ow_stride = OH * OW
    h_w_stride = H * W

    BLOCK_SIZE = 256
    BLOCK_SIZE_EDGES = 128

    if out is not None:
        accum = out
    else:
        accum = torch.empty((N, C, H, W), device=self.device, dtype=output_dtype)

    with torch_device_fn.device(self.device):
        if interior_elements > 0:
            if W >= BLOCK_SIZE:
                interior_chunks_per_row = triton.cdiv(W, BLOCK_SIZE)
                interior_grid = (nc_total * H * interior_chunks_per_row,)
                _interior_copy_kernel_row[interior_grid](
                    grad_output_4d,
                    accum,
                    OW,
                    H,
                    W,
                    pad_left,
                    pad_top,
                    oh_ow_stride,
                    h_w_stride,
                    BLOCK_SIZE,
                    interior_chunks_per_row,
                )
            else:
                interior_grid = (triton.cdiv(interior_elements, BLOCK_SIZE),)
                _interior_copy_kernel[interior_grid](
                    grad_output_4d,
                    accum,
                    OW,
                    H,
                    W,
                    pad_left,
                    pad_top,
                    interior_elements,
                    oh_ow_stride,
                    h_w_stride,
                    BLOCK_SIZE,
                )

        if edge_elements > 0:
            edge_grid = (triton.cdiv(edge_elements, BLOCK_SIZE_EDGES),)
            _edges_atomic_kernel[edge_grid](
                grad_output_4d,
                accum,
                OH,
                OW,
                H,
                W,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                edge_elements,
                nc_total,
                oh_ow_stride,
                h_w_stride,
                BLOCK_SIZE_EDGES,
            )

    if is_3d:
        accum = accum.view(C, H, W)
    return accum


def replication_pad2d_backward(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    logger.debug("GEMS REPLICATION_PAD2D_BACKWARD")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=None)


def replication_pad2d_backward_grad_input(
    grad_output: torch.Tensor, self: torch.Tensor, padding, *, grad_input: torch.Tensor
) -> torch.Tensor:
    logger.debug("GEMS REPLICATION_PAD2D_BACKWARD_GRAD_INPUT")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=grad_input)
