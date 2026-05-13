import torch
import triton
import triton.language as tl


@triton.jit
def per_token_group_quant_int8_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_stride,
    N,
    eps,
    int8_min,
    int8_max,
    BLOCK: tl.constexpr,
):
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < N
    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = absmax / int8_max
    y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


# Adapted from vllm v0.17.0
def per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    block = triton.next_power_of_2(group_size)
    num_warps = min(max(block // 256, 1), 8)
    per_token_group_quant_int8_kernel[(x.numel() // group_size,)](
        x,
        x_q,
        x_s,
        group_size,
        group_size,
        eps,
        int8_min=iinfo.min,
        int8_max=iinfo.max,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return x_q, x_s
