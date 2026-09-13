# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _te_rmsnorm_bwd_dx_kernel(
    dx_ptr,
    dz_ptr,
    x_ptr,
    gamma_ptr,
    rsigma_ptr,
    M,
    N,
    zero_centered_gamma: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    rsigma = tl.load(rsigma_ptr + row).to(tl.float32)
    c1 = 0.0

    for col_start in range(0, N, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        dz = tl.load(dz_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if zero_centered_gamma:
            gamma += 1.0
        c1 += tl.sum(tl.where(mask, x * rsigma * dz * gamma, 0.0), axis=0)

    c1 /= N
    for col_start in range(0, N, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        dz = tl.load(dz_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if zero_centered_gamma:
            gamma += 1.0
        x_hat = x * rsigma
        dx = rsigma * (dz * gamma - x_hat * c1)
        tl.store(dx_ptr + row * N + cols, dx, mask=mask)


@triton.jit
def _te_rmsnorm_bwd_dgamma_kernel(
    dgamma_ptr,
    dz_ptr,
    x_ptr,
    rsigma_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    col = tl.program_id(0)
    dgamma = 0.0

    for row in range(0, M):
        x = tl.load(x_ptr + row * N + col).to(tl.float32)
        dz = tl.load(dz_ptr + row * N + col).to(tl.float32)
        rsigma = tl.load(rsigma_ptr + row).to(tl.float32)
        dgamma += dz * x * rsigma

    tl.store(dgamma_ptr + col, dgamma)


def te_rmsnorm_bwd(
    dz: torch.Tensor,
    x: torch.Tensor,
    rsigma: torch.Tensor,
    gamma: torch.Tensor,
    sm_margin: int = 0,
    zero_centered_gamma: bool = False,
):
    del sm_margin
    original_shape = x.shape
    N = gamma.shape[0]
    x_2d = x.reshape(-1, N).contiguous()
    dz_2d = dz.reshape(-1, N).contiguous()
    rsigma = rsigma.contiguous()
    M = x_2d.shape[0]
    dx = torch.empty_like(x_2d)
    dgamma = torch.empty_like(gamma)

    block_size = 128
    with torch_device_fn.device(x.device):
        _te_rmsnorm_bwd_dx_kernel[(M,)](
            dx,
            dz_2d,
            x_2d,
            gamma,
            rsigma,
            M,
            N,
            zero_centered_gamma=zero_centered_gamma,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        _te_rmsnorm_bwd_dgamma_kernel[(N,)](
            dgamma,
            dz_2d,
            x_2d,
            rsigma,
            M,
            N,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )

    return dx.reshape(original_shape), dgamma


__all__ = ["te_rmsnorm_bwd"]
