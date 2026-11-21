import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_fwd_kernel(
    x_ptr, out_ptr, rows, D, BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    offs_row = pid_row * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_col = pid_col * BLOCK_COL + tl.arange(0, BLOCK_COL)

    row_mask = offs_row < rows
    col_mask = offs_col < D

    a = tl.load(
        x_ptr + offs_row[:, None] * (2 * D) + offs_col[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    b = tl.load(
        x_ptr + offs_row[:, None] * (2 * D) + offs_col[None, :] + D,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    out_val = a * tl.sigmoid(b)
    tl.store(
        out_ptr + offs_row[:, None] * D + offs_col[None, :],
        out_val,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def swiglu_bwd_kernel(
    dout_ptr, x_ptr, dx_ptr, rows, D, BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    offs_row = pid_row * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_col = pid_col * BLOCK_COL + tl.arange(0, BLOCK_COL)

    row_mask = offs_row < rows
    col_mask = offs_col < D

    dout = tl.load(
        dout_ptr + offs_row[:, None] * D + offs_col[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    a = tl.load(
        x_ptr + offs_row[:, None] * (2 * D) + offs_col[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    b = tl.load(
        x_ptr + offs_row[:, None] * (2 * D) + offs_col[None, :] + D,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    sig = tl.sigmoid(b)
    da = dout * sig
    db = dout * a * sig * (1 - sig)

    tl.store(
        dx_ptr + offs_row[:, None] * (2 * D) + offs_col[None, :],
        da,
        mask=row_mask[:, None] & col_mask[None, :],
    )
    tl.store(
        dx_ptr + offs_row[:, None] * (2 * D) + offs_col[None, :] + D,
        db,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def swiglu(x: torch.Tensor):
    assert x.shape[-1] % 2 == 0
    D = x.shape[-1] // 2
    orig_shape = x.shape
    x_2d = x.reshape(-1, 2 * D)
    rows = x_2d.shape[0]
    out_2d = torch.empty((rows, D), device=x.device, dtype=x.dtype)

    BLOCK_ROW = 16
    BLOCK_COL = 128
    grid = (triton.cdiv(rows, BLOCK_ROW), triton.cdiv(D, BLOCK_COL))
    swiglu_fwd_kernel[grid](x_2d, out_2d, rows, D, BLOCK_ROW, BLOCK_COL)
    return out_2d.reshape((*orig_shape[:-1], D))


def dswiglu(dout: torch.Tensor, x: torch.Tensor):
    assert x.shape[-1] % 2 == 0
    D = x.shape[-1] // 2
    orig_shape = x.shape
    x_2d = x.reshape(-1, 2 * D)
    dout_2d = dout.reshape(-1, D)
    rows = x_2d.shape[0]
    dx_2d = torch.empty_like(x_2d)

    BLOCK_ROW = 16
    BLOCK_COL = 128
    grid = (triton.cdiv(rows, BLOCK_ROW), triton.cdiv(D, BLOCK_COL))
    swiglu_bwd_kernel[grid](dout_2d, x_2d, dx_2d, rows, D, BLOCK_ROW, BLOCK_COL)
    return dx_2d.reshape(orig_shape)
