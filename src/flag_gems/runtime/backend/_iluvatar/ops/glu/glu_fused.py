# triton_src/glu/glu_fused.py
import torch
import triton
import triton.language as tl


@triton.jit
def _glu_fused_kernel(
    x_ptr,
    y_ptr,
    D,
    stride_xn,
    stride_xd,
    stride_yn,
    stride_yd,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    """
    GLU: y = x[:D//2] * sigmoid(x[D//2:])
    ✅ 一次global load，寄存器切分、sigmoid、mul全融合，无中间张量写回全局内存
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_idx = offs // D
    d_idx = offs % D

    x_ptrs = x_ptr + n_idx * stride_xn + d_idx * stride_xd
    x = tl.load(x_ptrs, mask=offs < D * tl.cdiv(tl.numel(x_ptr), D), other=0.0)
    half = D // 2
    x_val = tl.where(d_idx < half, x, 0.0)
    x_gate = tl.where(d_idx >= half, x, 0.0)
    gate = tl.sigmoid(x_gate)
    y = x_val * gate

    y_ptrs = y_ptr + n_idx * stride_yn + d_idx * stride_yd
    tl.store(y_ptrs, y, mask=offs < D * tl.cdiv(tl.numel(x_ptr), D))


def glu(x: torch.Tensor) -> torch.Tensor:
    """
    GLU activation，输入最后一维必须是偶数维度
    x shape: [*, D], D even
    """
    assert x.is_contiguous()
    assert x.shape[-1] % 2 == 0, "GLU last dim must be even"
    out = torch.empty_like(x)
    D = x.shape[-1]
    total_elem = x.numel()
    grid = lambda META: (triton.cdiv(total_elem, META["BLOCK_SIZE"]),)
    _glu_fused_kernel[grid](
        x,
        out,
        D,
        x.stride(-2),
        x.stride(-1),
        out.stride(-2),
        out.stride(-1),
        BLOCK_SIZE=1024,
    )
    return out
