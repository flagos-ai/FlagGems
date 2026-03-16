import math
import torch
import triton
import triton.language as tl

@triton.jit
def _fft_c2c_kernel(in_ptr, out_ptr, L, LINES, BLOCK_N: tl.constexpr, MAX_ITERS: tl.constexpr):
    line_id = tl.program_id(axis=0)
    k = tl.program_id(axis=1)
    if line_id >= LINES:
        return
    if k >= L:
        return
    acc_r = tl.zeros((), dtype=tl.float32)
    acc_i = tl.zeros((), dtype=tl.float32)
    two_pi = 6.28318530717958647692
    kf = k.to(tl.float32)
    Lf = L.to(tl.float32)
    for it in range(MAX_ITERS):
        n_start = it * BLOCK_N
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < L
        base_idx = line_id * L + n_offsets
        base_f32 = base_idx * 2
        xr = tl.load(in_ptr + base_f32, mask=n_mask, other=0.0)
        xi = tl.load(in_ptr + base_f32 + 1, mask=n_mask, other=0.0)
        nf = n_offsets.to(tl.float32)
        angle = - (two_pi * kf * nf) / Lf
        c = tl.cos(angle)
        s = tl.sin(angle)
        yr = xr * c + xi * s
        yi = xi * c - xr * s
        acc_r += tl.sum(yr, axis=0)
        acc_i += tl.sum(yi, axis=0)
    out_idx = (line_id * L + k) * 2
    tl.store(out_ptr + out_idx, acc_r)
    tl.store(out_ptr + out_idx + 1, acc_i)

def fft_c2c_triton(input_tensor: torch.Tensor) -> torch.Tensor:
    x = input_tensor
    if not torch.is_complex(x):
        x = x.to(torch.complex64)
    orig_dtype = x.dtype
    if x.dtype != torch.complex64:
        x = x.to(torch.complex64)
    if x.numel() == 0:
        return x.clone()
    if not x.is_cuda:
        y = torch.fft.fftn(x)
        if orig_dtype != y.dtype:
            y = y.to(orig_dtype)
        return y
    BLOCK_N = 1024
    MAX_ITERS = 64
    out = x
    for axis in range(out.ndim):
        y = out.movedim(axis, -1).contiguous()
        L = y.shape[-1]
        lines = y.numel() // L
        if L > BLOCK_N * MAX_ITERS:
            y = torch.fft.fft(y, dim=-1)
            out = y.movedim(-1, axis)
            continue
        y_out = torch.empty_like(y)
        y_ptr = y.view(torch.float32)
        y_out_ptr = y_out.view(torch.float32)
        grid = (lines, L)
        _fft_c2c_kernel[grid](
            y_ptr, y_out_ptr,
            L, lines,
            BLOCK_N=BLOCK_N,
            MAX_ITERS=MAX_ITERS
        )
        out = y_out.movedim(-1, axis)
    if orig_dtype != out.dtype:
        out = out.to(orig_dtype)
    return out