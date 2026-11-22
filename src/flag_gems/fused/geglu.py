import torch
import triton
import triton.language as tl


@triton.jit
def geglu_kernel(x_ptr, out_ptr, total_elem, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elem

    a = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(x_ptr + offs + total_elem, mask=mask, other=0.0)
    # Iluvatar Triton 仅支持 tl.sigmoid，采用 sigmoid 形式Gelu近似
    # 注意：该近似和PyTorch原生gelu存在数值偏差，适合推理场景，精度敏感任务慎用
    gelu_b = b * tl.sigmoid(1.702 * b)
    res = a * gelu_b
    tl.store(out_ptr + offs, res, mask=mask)


def geglu(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0, "last dim must be even"
    hidden = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (hidden,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    total_elem = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elem, BLOCK_SIZE),)
    geglu_kernel[grid](x, out, total_elem, BLOCK_SIZE=BLOCK_SIZE)
    return out


def dgeglu(x: torch.Tensor):
    raise NotImplementedError("dgeglu backward not implemented yet")
