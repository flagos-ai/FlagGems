# src/flag_gems/ops/triton_src/weight_norm.py
import torch
import triton
import triton.language as tl


@triton.jit
def weight_norm_fwd_kernel(
    v_ptr, g_ptr, w_ptr, norm_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + pid)

    v_sq = v * v
    sum_sq = tl.sum(v_sq, axis=0)
    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, 1e-12)

    w = g * v / norm
    tl.store(w_ptr + offs, w, mask=mask)
    tl.store(norm_ptr + pid, norm)


@triton.jit
def weight_norm_bwd_kernel(
    dw_ptr,
    v_ptr,
    g_ptr,
    norm_ptr,
    dv_ptr,
    dg_ptr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    dw = tl.load(dw_ptr + offs, mask=mask, other=0.0)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + pid)
    norm = tl.load(norm_ptr + pid)
    norm = tl.maximum(norm, 1e-12)

    dot_dw_v = tl.sum(dw * v, axis=0)
    dg = dot_dw_v / norm
    dv = (g / norm) * (dw - v * dot_dw_v / (norm * norm))

    tl.store(dg_ptr + pid, dg)
    tl.store(dv_ptr + offs, dv, mask=mask)


def weight_norm_interface(
    v: torch.Tensor, g: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    WeightNorm forward: w = g * v / ||v||_2
    Performs L2 normalization on each row of v (dim=1).
    Match PyTorch nn.utils.weight_norm for linear weight [out_dim, in_dim].

    Args:
        v: [out_dim, in_dim], contiguous, cuda tensor
        g: [out_dim], contiguous, cuda tensor, per-row scale
    Returns:
        w: same shape as v
        norm: l2 norm of each v row, shape [out_dim]
    Notes:
        Limitation: in_dim <= 8192, BLOCK_SIZE = next_power_of_2(in_dim)
    """
    assert v.is_contiguous(), "v must be contiguous"
    assert g.is_contiguous(), "g must be contiguous"
    assert v.shape[0] == g.shape[0], "row count mismatch between v and g"
    assert v.device.type == "cuda", "only cuda supported"
    assert v.dtype in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ), "dtype support: fp32/bf16/fp16"

    n_rows = v.shape[0]
    n_elem_per_row = v.shape[1]
    w = torch.empty_like(v)
    norm = torch.empty_like(g)

    BLOCK_SIZE = triton.next_power_of_2(n_elem_per_row)
    grid = (n_rows,)

    weight_norm_fwd_kernel[grid](v, g, w, norm, n=n_elem_per_row, BLOCK_SIZE=BLOCK_SIZE)
    return w, norm


def weight_norm_interface_backward(
    dw: torch.Tensor, v: torch.Tensor, g: torch.Tensor, norm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for weight norm.
    Args:
        dw: grad w.r.t w, same shape as v
        v: saved v from forward
        g: saved g from forward
        norm: saved norm from forward
    Returns:
        dv, dg
    """
    assert dw.is_contiguous(), "dw must be contiguous"
    assert v.is_contiguous(), "v must be contiguous"
    assert g.is_contiguous(), "g must be contiguous"
    assert norm.is_contiguous(), "norm must be contiguous"
    assert dw.shape == v.shape
    assert g.shape == norm.shape
    assert v.device.type == "cuda"
    assert v.dtype in (torch.float32, torch.bfloat16, torch.float16)

    n_rows = v.shape[0]
    n_elem_per_row = v.shape[1]
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)

    BLOCK_SIZE = triton.next_power_of_2(n_elem_per_row)
    grid = (n_rows,)

    weight_norm_bwd_kernel[grid](
        dw, v, g, norm, dv, dg, n=n_elem_per_row, BLOCK_SIZE=BLOCK_SIZE
    )
    return dv, dg


class WeightNormTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, g):
        w, norm = weight_norm_interface(v, g)
        ctx.save_for_backward(v, g, norm)
        return w

    @staticmethod
    def backward(ctx, dw):
        v, g, norm = ctx.saved_tensors
        dv, dg = weight_norm_interface_backward(dw, v, g, norm)
        return dv, dg


def weight_norm_triton(v, g):
    """
    Autograd wrapper, direct replacement reference:
        w = g[:, None] * v / torch.norm(v, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    """
    return WeightNormTritonFunc.apply(v, g)
