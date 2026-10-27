import torch
import triton
import triton.language as tl


@triton.jit
def _mse_bwd_kernel(
    x_ptr,
    t_ptr,
    g_ptr,
    out_ptr,
    numel,
    scale,
    BLOCK: tl.constexpr,
    GSCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    t = tl.load(t_ptr + offs, mask=mask, other=0.0)
    if GSCALAR:
        g = tl.load(g_ptr)
    else:
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    diff = x - t
    out = 2.0 * diff * g * scale
    tl.store(out_ptr + offs, out, mask=mask)


def run(grad_output, self, target, reduction=1):
    numel = self.numel()
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    if numel == 0:
        return out

    # reduction semantics: 0=none, 1=mean, 2=sum (PyTorch _Reduction enum)
    if isinstance(reduction, str):
        r = reduction
    else:
        try:
            r = int(reduction)
        except (TypeError, ValueError):
            r = int(reduction.item())
    scale = 1.0 / float(numel) if r in (1, "mean", "MEAN") else 1.0

    g_scalar = grad_output.numel() == 1
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _mse_bwd_kernel[grid](
        self,
        target,
        grad_output,
        out,
        numel,
        scale,
        BLOCK=BLOCK,
        GSCALAR=g_scalar,
        num_warps=4,
    )
    return out


# Alias for FlagGems import convention
mse_loss_backward = run
