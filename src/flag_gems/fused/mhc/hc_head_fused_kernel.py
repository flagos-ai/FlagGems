import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
    ],
    key=["H", "HC"],
)
@triton.jit
def _hc_head_apply_pre_mix_kernel(
    hs_ptr,
    pre_mix_ptr,
    out_ptr,
    T,
    H,
    hs_stride_t,
    hs_stride_m,
    hs_stride_h,
    pre_stride_t,
    pre_stride_m,
    out_stride_t,
    out_stride_h,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= T:
        return

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    hs_t_base = pid_t * hs_stride_t
    pre_t_base = pid_t * pre_stride_t

    for i_hc in tl.static_range(HC):
        pre = tl.load(pre_mix_ptr + pre_t_base + i_hc * pre_stride_m).to(tl.float32)
        hs_ptrs = hs_ptr + hs_t_base + i_hc * hs_stride_m + h_off * hs_stride_h
        hs_vals = tl.load(hs_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        acc += pre * hs_vals

    out_ptrs = out_ptr + pid_t * out_stride_t + h_off * out_stride_h
    tl.store(out_ptrs, acc, mask=h_mask)


def hc_head_fused_kernel_ref(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> torch.Tensor:
    if hs_flat.shape[0] == 0:
        return out
    x = hs_flat.reshape(hs_flat.shape[0], hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps
    result = torch.sum(pre_mix.unsqueeze(-1) * hs_flat.to(torch.float32), dim=1).to(
        out.dtype
    )
    out.copy_(result)
    return out


def hc_head_fused_kernel(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> torch.Tensor:
    assert hs_flat.dtype in [torch.float32, torch.float16, torch.bfloat16]
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    num_tokens = hs_flat.shape[0]
    if num_tokens == 0:
        return out

    assert hs_flat.shape == (num_tokens, hc_mult, hidden_size)
    assert fn.shape == (hc_mult, hc_mult * hidden_size)
    assert hc_scale.shape == (1,)
    assert hc_base.shape == (hc_mult,)
    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == hs_flat.dtype

    x = hs_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps

    if hs_flat.device.type != "cuda":
        return hc_head_fused_kernel_ref(
            hs_flat,
            fn,
            hc_scale,
            hc_base,
            out,
            hidden_size,
            rms_eps,
            hc_eps,
            hc_mult,
        )

    hs_flat_c = hs_flat.contiguous()
    pre_mix_c = pre_mix.contiguous()
    out_c = out.contiguous()

    def grid(meta):
        return num_tokens, triton.cdiv(hidden_size, meta["BLOCK_H"])

    _hc_head_apply_pre_mix_kernel[grid](
        hs_flat_c,
        pre_mix_c,
        out_c,
        num_tokens,
        hidden_size,
        hs_flat_c.stride(0),
        hs_flat_c.stride(1),
        hs_flat_c.stride(2),
        pre_mix_c.stride(0),
        pre_mix_c.stride(1),
        out_c.stride(0),
        out_c.stride(1),
        HC=hc_mult,
    )

    if out.data_ptr() != out_c.data_ptr():
        out.copy_(out_c)
    return out
