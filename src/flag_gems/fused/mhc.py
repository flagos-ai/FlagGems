import triton
import torch
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 16}),
        triton.Config({'BLOCK_H': 16}, num_warps=2),
        triton.Config({'BLOCK_H': 16}, num_warps=4),
        triton.Config({'BLOCK_H': 32}, num_warps=1),
        triton.Config({'BLOCK_H': 32}, num_warps=2),
        triton.Config({'BLOCK_H': 32}, num_warps=4),
        triton.Config({'BLOCK_H': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 64}, num_warps=2),
        triton.Config({'BLOCK_H': 64}, num_warps=4),
        triton.Config({'BLOCK_H': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 64}, num_warps=8),
        triton.Config({'BLOCK_H': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_H': 128}, num_warps=2),
        triton.Config({'BLOCK_H': 128}, num_warps=4),
        triton.Config({'BLOCK_H': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 128}, num_warps=8),
        triton.Config({'BLOCK_H': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_H': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 256}, num_warps=8, num_stages=3),
    ],
    key=['H', 'HC'],
)
@triton.jit
def mhc_kernel(
    x_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_ptr,
    out_ptr,
    n,
    h,
    hc,
    stride_xn,
    stride_rn,
    stride_rhc,
    stride_postn,
    stride_combn,
    stride_outhc,
    BLOCK_H: tl.constexpr,
    HC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h


    x_ptrs = x_ptr + pid_n * stride_xn + offs_h
    x = tl.load(x_ptrs, mask=mask_h, other=0).to(tl.float32)

    post_ptrs = post_mix_ptr + pid_n * stride_postn + tl.arange(0, HC)
    post = tl.load(post_ptrs).to(tl.float32)

    acc = tl.zeros((HC, BLOCK_H), dtype=tl.float32)

    residual_base = residual_ptr + pid_n * stride_rn
    comb_base = comb_ptr + pid_n * stride_combn

    for k in tl.static_range(HC):

        r_ptrs = residual_base + k * stride_rhc + offs_h
        r = tl.load(r_ptrs, mask=mask_h, other=0).to(tl.float32)

        comb_row_ptrs = comb_base + k * HC + tl.arange(0, HC)

        w = tl.load(comb_row_ptrs).to(tl.float32)

        acc += w[:, None] * r[None, :]

    term1 = post[:, None] * x[None, :]

    out = acc + term1

    out_ptrs = (
        out_ptr
        + pid_n * stride_outhc
        + tl.arange(0, HC)[:, None] * h
        + offs_h[None, :]
    )

    tl.store(out_ptrs, out.to(tl.bfloat16), mask=mask_h[None, :])

def mhc(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
):
    n, h = x.shape
    hc = residual.shape[1]

    out = torch.empty((n, hc, h), device=x.device, dtype=torch.bfloat16)

    grid = lambda meta : (
        n,
        triton.cdiv(h, meta['BLOCK_H']),
    )

    mhc_kernel[grid](
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        out,
        n,
        h,
        hc,
        x.stride(0),
        residual.stride(0),
        residual.stride(1),
        post_layer_mix.stride(0),
        comb_res_mix.stride(0),
        out.stride(0),
        HC=hc,
    )

    return out