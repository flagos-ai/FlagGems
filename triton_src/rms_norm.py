import triton
import triton.language as tl


@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    out_ptr,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    in_ptr,  # pointer to the input
    w_ptr,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tl.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + cols * x_stride_c, mask, other=0.0).to(cdtype)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms * w).to(cdtype)
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)
    tl.store(INV_RMS + pid, rrms)


@triton.jit(do_not_specialize=["eps"])
def rms_norm_grad_dx_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DX,  # pointer to the output
    W,  # pointer to the weights
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    DX += pid * dx_stride_r
    X += pid * x_stride_r
    DY += pid * x_stride_r
    INV_RMS += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(INV_RMS).to(tl.float32)
    dy = tl.load(DY + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    dy = dy * w

    normalized_buf = x * inv_rms
    row_sum_stats = tl.sum(normalized_buf * dy, axis=0)

    norm_val = normalized_buf / N
    dx = (dy - norm_val * row_sum_stats) * inv_rms

    tl.store(DX + cols * dx_stride_c, dx, mask=mask)


@triton.jit
def rms_norm_grad_dw_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DW,  # pointer to the output
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)

    row_start = row_pid * ROW_BLOCK_SIZE
    col_start = col_pid * COL_BLOCK_SIZE

    offset = row_start * x_stride_r + col_start * x_stride_c
    X += offset
    DY += offset
    INV_RMS += row_start

    rows = tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, COL_BLOCK_SIZE)

    row_mask = (row_start + rows) < M
    col_mask = (col_start + cols) < N

    x = tl.load(
        X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    inv_rms = tl.load(INV_RMS + rows, row_mask, other=0.0).to(tl.float32)
    dy = tl.load(
        DY + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    d_weight = x * dy * inv_rms[:, None]
    partial_dweight_sum = tl.sum(d_weight, axis=0)

    tl.store(
        DW + row_pid * N + col_start + cols,
        partial_dweight_sum,
        mask=col_mask,
    )
