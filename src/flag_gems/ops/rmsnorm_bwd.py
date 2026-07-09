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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["N"])
def rmsnorm_bwd_dx_kernel(
    dx_ptr,
    dz_ptr,
    x_ptr,
    weight_ptr,
    rsigma_ptr,
    N,
    zero_centered_gamma: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute dx for small N (fits in one block)."""
    pid = tl.program_id(0)

    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load inputs
    x = tl.load(x_ptr + pid * N + cols, mask=mask, other=0.0).to(cdtype)
    dz = tl.load(dz_ptr + pid * N + cols, mask=mask, other=0.0).to(cdtype)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(cdtype)
    rsigma = tl.load(rsigma_ptr + pid).to(cdtype)

    if zero_centered_gamma:
        w = w + 1.0

    x_hat = x * rsigma
    dz_w = dz * w

    c1 = tl.sum(x_hat * dz_w, axis=0) / N
    dx = rsigma * (dz_w - x_hat * c1)

    tl.store(dx_ptr + pid * N + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def rmsnorm_bwd_dx_loop_kernel(
    dx_ptr,
    dz_ptr,
    x_ptr,
    weight_ptr,
    rsigma_ptr,
    N,
    zero_centered_gamma: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Compute dx for large N (requires loop)."""
    pid = tl.program_id(0)

    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty

    rsigma = tl.load(rsigma_ptr + pid).to(cdtype)

    # First pass: compute c1 = mean(x_hat * dz * w)
    c1_acc = tl.zeros([TILE_N], dtype=cdtype)
    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        mask = cols < N

        x = tl.load(x_ptr + pid * N + cols, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_ptr + pid * N + cols, mask=mask, other=0.0).to(cdtype)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(cdtype)

        if zero_centered_gamma:
            w = w + 1.0

        x_hat = x * rsigma
        c1_acc += tl.where(mask, x_hat * dz * w, 0.0)

    c1 = tl.sum(c1_acc, axis=0) / N

    # Second pass: compute dx
    for off in range(0, N, TILE_N):
        cols = off + tl.arange(0, TILE_N)
        mask = cols < N

        x = tl.load(x_ptr + pid * N + cols, mask=mask, other=0.0).to(cdtype)
        dz = tl.load(dz_ptr + pid * N + cols, mask=mask, other=0.0).to(cdtype)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(cdtype)

        if zero_centered_gamma:
            w = w + 1.0

        x_hat = x * rsigma
        dx = rsigma * (dz * w - x_hat * c1)

        tl.store(dx_ptr + pid * N + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def rmsnorm_bwd_dgamma_kernel(
    dgamma_ptr,
    dz_ptr,
    x_ptr,
    rsigma_ptr,
    M,
    N,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Compute dgamma for RMSNorm backward."""
    pid = tl.program_id(0)
    col_start = pid * BLOCK_SIZE_N
    cols = col_start + tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < N

    if tl.constexpr(x_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        x_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = x_ptr.dtype.element_ty

    dgamma_acc = tl.zeros([BLOCK_SIZE_N], dtype=cdtype)

    for row in range(M):
        rsigma_val = tl.load(rsigma_ptr + row).to(cdtype)
        x_row = tl.load(x_ptr + row * N + cols, mask=col_mask, other=0.0).to(cdtype)
        dz_row = tl.load(dz_ptr + row * N + cols, mask=col_mask, other=0.0).to(cdtype)

        x_hat = x_row * rsigma_val
        dgamma_acc += dz_row * x_hat

    tl.store(
        dgamma_ptr + cols, dgamma_acc.to(dgamma_ptr.dtype.element_ty), mask=col_mask
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
        MAX_BLOCK_SIZE = 65536
        BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)

        if N <= MAX_BLOCK_SIZE:
            rmsnorm_bwd_dx_kernel[(M,)](
                dx,
                dz_2d,
                x_2d,
                gamma,
                rsigma,
                N,
                zero_centered_gamma,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            # Use loop kernel for large N
            TILE_N = 8192
            rmsnorm_bwd_dx_loop_kernel[(M,)](
                dx,
                dz_2d,
                x_2d,
                gamma,
                rsigma,
                N,
                zero_centered_gamma,
                TILE_N=TILE_N,
            )

        # Compute dgamma
        BLOCK_SIZE_N = min(triton.next_power_of_2(N), 1024)
        num_blocks = triton.cdiv(N, BLOCK_SIZE_N)
        rmsnorm_bwd_dgamma_kernel[(num_blocks,)](
            dgamma,
            dz_2d,
            x_2d,
            rsigma,
            M,
            N,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

    # Restore original shape for dx
    dx = dx.view(*original_shape)

    return dx, dgamma
