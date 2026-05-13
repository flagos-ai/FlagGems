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


@triton.jit
def _hc_head_fused_kernel(
    residual_ptr,
    fn_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    total_elems: tl.constexpr,
    hc_mult: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_token = tl.program_id(0)

    sqrsum = tl.zeros([], dtype=tl.float32)
    mix0 = tl.zeros([], dtype=tl.float32)
    mix1 = tl.zeros([], dtype=tl.float32)
    mix2 = tl.zeros([], dtype=tl.float32)
    mix3 = tl.zeros([], dtype=tl.float32)

    res_base = pid_token * total_elems

    for block_start in range(0, total_elems, BLOCK_K):
        offsets = block_start + tl.arange(0, BLOCK_K)
        mask = offsets < total_elems

        res_vals = tl.load(residual_ptr + res_base + offsets, mask=mask, other=0.0).to(tl.float32)
        sqrsum += tl.sum(res_vals * res_vals)

        fn0 = tl.load(fn_ptr + 0 * total_elems + offsets, mask=mask, other=0.0)
        fn1 = tl.load(fn_ptr + 1 * total_elems + offsets, mask=mask, other=0.0)
        fn2 = tl.load(fn_ptr + 2 * total_elems + offsets, mask=mask, other=0.0)
        fn3 = tl.load(fn_ptr + 3 * total_elems + offsets, mask=mask, other=0.0)

        mix0 += tl.sum(res_vals * fn0)
        mix1 += tl.sum(res_vals * fn1)
        mix2 += tl.sum(res_vals * fn2)
        mix3 += tl.sum(res_vals * fn3)

    rsqrt_val = tl.rsqrt(sqrsum / total_elems + rms_eps)

    hc_scale = tl.load(hc_scale_ptr)
    base_0 = tl.load(hc_base_ptr + 0)
    base_1 = tl.load(hc_base_ptr + 1)
    base_2 = tl.load(hc_base_ptr + 2)
    base_3 = tl.load(hc_base_ptr + 3)

    pre_mix_0 = tl.sigmoid(mix0 * rsqrt_val * hc_scale + base_0) + hc_eps
    pre_mix_1 = tl.sigmoid(mix1 * rsqrt_val * hc_scale + base_1) + hc_eps
    pre_mix_2 = tl.sigmoid(mix2 * rsqrt_val * hc_scale + base_2) + hc_eps
    pre_mix_3 = tl.sigmoid(mix3 * rsqrt_val * hc_scale + base_3) + hc_eps

    res_base2 = pid_token * hc_mult * hidden_size
    out_base = pid_token * hidden_size

    for h_start in range(0, hidden_size, BLOCK_H):
        offsets = h_start + tl.arange(0, BLOCK_H)
        mask = offsets < hidden_size

        r0 = tl.load(residual_ptr + res_base2 + 0 * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        r1 = tl.load(residual_ptr + res_base2 + 1 * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        r2 = tl.load(residual_ptr + res_base2 + 2 * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        r3 = tl.load(residual_ptr + res_base2 + 3 * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)

        out_val = pre_mix_0 * r0 + pre_mix_1 * r1 + pre_mix_2 * r2 + pre_mix_3 * r3
        tl.store(out_ptr + out_base + offsets, out_val.to(OUT_DTYPE), mask=mask)


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

    if hc_mult == 4:
        dtype_map = {
            torch.float32: tl.float32,
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
        }
        OUT_DTYPE = dtype_map[hs_flat.dtype]

        residual_c = hs_flat.reshape(num_tokens, hc_mult * hidden_size).contiguous()
        fn_c = fn.contiguous()
        out_c = out.contiguous()

        total_elems = hc_mult * hidden_size
        grid = (num_tokens,)

        _hc_head_fused_kernel[grid](
            residual_c,
            fn_c,
            hc_scale,
            hc_base,
            out_c,
            hidden_size,
            total_elems,
            hc_mult,
            rms_eps,
            hc_eps,
            BLOCK_K=32768,
            BLOCK_H=8192,
            OUT_DTYPE=OUT_DTYPE,
            num_warps=16,
            num_stages=1,
        )

        if out.data_ptr() != out_c.data_ptr():
            out.copy_(out_c)
        return out

    x = hs_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps

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
