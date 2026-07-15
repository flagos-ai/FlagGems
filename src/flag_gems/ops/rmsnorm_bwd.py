"""
RMSNorm backward implementation aligned with TransformerEngine's rmsnorm_bwd.

TransformerEngine rmsnorm_bwd signature:
    rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)

Returns:
    (dx, dgamma) where:
        - dx: gradient w.r.t. input, shape same as x
        - dgamma: gradient w.r.t. weight, shape same as gamma
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("rmsnorm_bwd_dx"),
    key=["M", "N"],
)
@triton.jit
def rmsnorm_bwd_dx_kernel(
    dx_ptr,
    dz_ptr,
    x_ptr,
    weight_ptr,
    rsigma_ptr,
    M,
    N,
    zero_centered_gamma: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    """
    Compute dx for RMSNorm backward using 2D tiling.
    Each program handles BLOCK_ROW_SIZE rows.
    """
    pid = ext.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = pid < M

    # Determine compute dtype
    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty

    # Setup pointers with row offsets
    dz_ptr = dz_ptr + pid * N
    x_ptr = x_ptr + pid * N
    dx_ptr = dx_ptr + pid * N

    # Load rsigma for each row
    rsigma = tl.load(rsigma_ptr + pid, mask=row_mask).to(cdtype)

    # First pass: compute c1 = mean(x_hat * dz * w) for each row
    c1_acc = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=cdtype)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_ptr + cols, mask=mask, other=0.0).to(cdtype)
        w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(cdtype)

        if zero_centered_gamma:
            w = w + 1.0

        x_hat = x * rsigma
        c1_acc += tl.where(mask, x_hat * dz * w, 0.0)

    c1 = tl.sum(c1_acc, axis=1)[:, None] / N

    # Second pass: compute dx = rsigma * (dz * w - x_hat * c1)
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_ptr + cols, mask=mask, other=0.0).to(cdtype)
        w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(cdtype)

        if zero_centered_gamma:
            w = w + 1.0

        x_hat = x * rsigma
        dx = rsigma * (dz * w - x_hat * c1)

        tl.store(dx_ptr + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("rmsnorm_bwd_dgamma"),
    key=["M", "N"],
)
@triton.jit
def rmsnorm_bwd_dgamma_kernel(
    dgamma_ptr,
    dz_ptr,
    x_ptr,
    rsigma_ptr,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    """
    Compute dgamma for RMSNorm backward using 2D tiling.
    Simple version: each program handles BLOCK_COL_SIZE columns, iterating over all rows.
    """
    pid = ext.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = pid < N

    # Determine compute dtype
    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty

    # Setup pointers with column offsets
    dz_col_ptr = dz_ptr + pid[None, :]
    x_col_ptr = x_ptr + pid[None, :]

    # Accumulate dgamma over all rows
    dgamma_acc = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=cdtype)

    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        row_mask = rows < M
        mask = row_mask & col_mask[None, :]

        x = tl.load(x_col_ptr + rows * N, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_col_ptr + rows * N, mask=mask, other=0.0).to(cdtype)
        rsigma = tl.load(rsigma_ptr + rows, mask=row_mask).to(cdtype)

        x_hat = x * rsigma
        dgamma_acc += tl.where(mask, dz * x_hat, 0.0)

    dgamma = tl.sum(dgamma_acc, axis=0)
    tl.store(dgamma_ptr + pid, dgamma.to(dgamma_ptr.dtype.element_ty), mask=col_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("rmsnorm_bwd_dgamma"),
    key=["M", "N"],
)
@triton.jit
def rmsnorm_bwd_dgamma_parallel_kernel(
    dgamma_partial_ptr,
    dz_ptr,
    x_ptr,
    rsigma_ptr,
    M,
    N,
    stride_partial,  # stride for partial buffer
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    """
    Compute partial dgamma using 2D grid parallelization.
    Grid: (num_col_groups, num_row_groups)
    Each program handles BLOCK_COL_SIZE columns and iterates over its assigned row range.
    """
    col_group_id = ext.program_id(0)
    row_group_id = ext.program_id(1)

    # Column range for this program
    col_start = col_group_id * BLOCK_COL_SIZE
    cols = col_start + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = cols < N

    # Determine compute dtype
    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty

    # Setup pointers with column offsets
    dz_col_ptr = dz_ptr + cols[None, :]
    x_col_ptr = x_ptr + cols[None, :]

    # Row range for this program
    rows_per_group = tl.cdiv(M, tl.num_programs(1))
    row_start = row_group_id * rows_per_group
    row_end = tl.minimum(row_start + rows_per_group, M)

    # Accumulate dgamma over assigned rows
    dgamma_acc = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=cdtype)

    for off in range(row_start, row_end, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        row_mask = rows < row_end
        mask = row_mask & col_mask[None, :]

        x = tl.load(x_col_ptr + rows * N, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_col_ptr + rows * N, mask=mask, other=0.0).to(cdtype)
        rsigma = tl.load(rsigma_ptr + rows, mask=row_mask).to(cdtype)

        x_hat = x * rsigma
        dgamma_acc += tl.where(mask, dz * x_hat, 0.0)

    dgamma_partial = tl.sum(dgamma_acc, axis=0)

    # Store partial dgamma: [row_group_id, col_start:col_start+BLOCK_COL_SIZE]
    tl.store(
        dgamma_partial_ptr + row_group_id * stride_partial + cols,
        dgamma_partial.to(dgamma_partial_ptr.dtype.element_ty),
        mask=col_mask,
    )


@libentry()
@triton.jit
def rmsnorm_bwd_dgamma_reduce_kernel(
    dgamma_ptr,
    dgamma_partial_ptr,
    num_row_groups,
    N,
    stride_partial,
    BLOCK_COL_SIZE: tl.constexpr,
):
    """
    Reduce partial dgamma across all row groups.
    """
    pid = ext.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = pid < N

    dgamma_acc = tl.zeros([BLOCK_COL_SIZE], dtype=tl.float32)

    for row_group in range(num_row_groups):
        partial = tl.load(
            dgamma_partial_ptr + row_group * stride_partial + pid,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        dgamma_acc += partial

    tl.store(
        dgamma_ptr + pid, dgamma_acc.to(dgamma_ptr.dtype.element_ty), mask=col_mask
    )


def rmsnorm_bwd(
    dz: torch.Tensor,
    x: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    """
    RMSNorm backward pass.

    Args:
        dz: gradient of output, shape (*, N)
        x: input tensor from forward pass, shape (*, N)
        rsigma: 1/sqrt(variance + eps) from forward pass, shape (*,)
        gamma: weight tensor, shape (N,)
        sm_margin: SM margin (unused, for API compatibility)
        zero_centered_gamma: if True, gamma is centered around 0

    Returns:
        dx: gradient w.r.t. input, shape (*, N)
        dgamma: gradient w.r.t. weight, shape (N,)
    """
    # Save original shape and flatten to 2D
    original_shape = x.shape
    N = gamma.shape[0]
    x_2d = x.view(-1, N)
    dz_2d = dz.view(-1, N)
    M = x_2d.shape[0]

    # Ensure contiguous
    x_2d = x_2d.contiguous()
    dz_2d = dz_2d.contiguous()
    rsigma = rsigma.contiguous()

    # Allocate outputs
    dx = torch.empty_like(x_2d)
    dgamma = torch.zeros(N, dtype=gamma.dtype, device=gamma.device)

    with torch_device_fn.device(x.device):
        # Compute dx
        grid_dx = lambda meta: (triton.cdiv(M, meta["BLOCK_ROW_SIZE"]),)
        rmsnorm_bwd_dx_kernel[grid_dx](
            dx,
            dz_2d,
            x_2d,
            gamma,
            rsigma,
            M,
            N,
            zero_centered_gamma,
        )

        # Compute dgamma
        # For small M, use simple version (no 2D parallelization overhead)
        # For large M, use 2D parallel version for better performance
        M_THRESHOLD = 2048
        if M <= M_THRESHOLD:
            # Simple version: single kernel iterates over all rows
            grid_dgamma = lambda meta: (triton.cdiv(N, meta["BLOCK_COL_SIZE"]),)
            rmsnorm_bwd_dgamma_kernel[grid_dgamma](
                dgamma,
                dz_2d,
                x_2d,
                rsigma,
                M,
                N,
            )
        else:
            # 2D parallel version for large M
            num_row_groups = min(triton.cdiv(M, 128), 64)
            num_row_groups = max(num_row_groups, 1)

            # Allocate partial dgamma buffer
            dgamma_partial = torch.zeros(
                (num_row_groups, N), dtype=torch.float32, device=x.device
            )
            stride_partial = N

            grid_dgamma = lambda meta: (
                triton.cdiv(N, meta["BLOCK_COL_SIZE"]),
                num_row_groups,
            )
            rmsnorm_bwd_dgamma_parallel_kernel[grid_dgamma](
                dgamma_partial,
                dz_2d,
                x_2d,
                rsigma,
                M,
                N,
                stride_partial,
            )

            # Reduce partial dgamma
            REDUCE_BLOCK_SIZE = 256
            grid_reduce = (triton.cdiv(N, REDUCE_BLOCK_SIZE),)
            rmsnorm_bwd_dgamma_reduce_kernel[grid_reduce](
                dgamma,
                dgamma_partial,
                num_row_groups,
                N,
                stride_partial,
                BLOCK_COL_SIZE=REDUCE_BLOCK_SIZE,
            )

    # Restore original shape for dx
    dx = dx.view(*original_shape)

    return dx, dgamma
