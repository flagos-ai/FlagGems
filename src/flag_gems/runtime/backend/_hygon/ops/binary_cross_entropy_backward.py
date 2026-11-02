import torch
import triton
import triton.language as tl


@triton.jit
def _bce_bwd_kernel(
    grad_ptr,
    self_ptr,
    target_ptr,
    weight_ptr,
    out_ptr,
    n_elements,
    denom_val,
    has_weight: tl.constexpr,
    reduction: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    if EVEN:
        g = tl.load(grad_ptr + offs)
        s = tl.load(self_ptr + offs)
        t = tl.load(target_ptr + offs)
        out_dtype = g.dtype
        g = g.to(tl.float32)
        s = s.to(tl.float32)
        t = t.to(tl.float32)
        val = g * (s - t) / ((1.0 - s) * s)
        if has_weight:
            w = tl.load(weight_ptr + offs).to(tl.float32)
            val = val * w
        if reduction == 1:
            val = val / denom_val
        tl.store(out_ptr + offs, val.to(out_dtype))
    else:
        mask = offs < n_elements
        g = tl.load(grad_ptr + offs, mask=mask, other=0.0)
        s = tl.load(self_ptr + offs, mask=mask, other=0.0)
        t = tl.load(target_ptr + offs, mask=mask, other=0.0)
        out_dtype = g.dtype
        g = g.to(tl.float32)
        s = s.to(tl.float32)
        t = t.to(tl.float32)
        val = g * (s - t) / ((1.0 - s) * s)
        if has_weight:
            w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            val = val * w
        if reduction == 1:
            val = val / denom_val
        tl.store(out_ptr + offs, val.to(out_dtype), mask=mask)


def run(grad_output, self, target, weight=None, reduction=1):
    if isinstance(reduction, str):
        reduction = {"none": 0, "mean": 1, "sum": 2}.get(reduction.lower(), 1)
    else:
        reduction = int(reduction)

    grad = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    inp = self if self.is_contiguous() else self.contiguous()
    tgt = target if target.is_contiguous() else target.contiguous()
    w = weight
    if w is not None and not w.is_contiguous():
        w = w.contiguous()

    out = torch.empty_like(inp)
    n = inp.numel()
    if n == 0:
        return out

    grad_flat = grad.view(-1)
    inp_flat = inp.view(-1)
    tgt_flat = tgt.view(-1)
    out_flat = out.view(-1)
    has_weight = w is not None
    w_flat = w.view(-1) if has_weight else inp_flat

    # Unmasked fast path when the size is a multiple of the block; larger
    # sizes use a smaller block (more blocks) for better load balance.
    # At the 1M-element class, fp32 measured faster with BLOCK=1024/1 warp
    # (more blocks, more ILP per thread), fp16/bf16 with BLOCK=2048/4 warps.
    # At the >=4M class, fp16/bf16 measured faster with 2 warps per block.
    if n >= 4_000_000 and n % 1024 == 0:
        if inp.dtype == torch.float32:
            BLOCK, EVEN, NW = 1024, True, 4
        else:
            BLOCK, EVEN, NW = 1024, True, 2
    elif n >= 1_000_000 and n % 2048 == 0:
        if inp.dtype == torch.float32:
            BLOCK, EVEN, NW = 1024, True, 1
        else:
            BLOCK, EVEN, NW = 2048, True, 4
    elif n >= 2048 and n % 2048 == 0:
        BLOCK, EVEN, NW = 2048, True, 4
    else:
        BLOCK, EVEN, NW = 1024, False, 4

    denom_val = float(n) if reduction == 1 else 1.0

    grid = (triton.cdiv(n, BLOCK),)
    _bce_bwd_kernel[grid](
        grad_flat,
        inp_flat,
        tgt_flat,
        w_flat,
        out_flat,
        n,
        denom_val,
        has_weight=has_weight,
        reduction=reduction,
        EVEN=EVEN,
        BLOCK=BLOCK,
        num_warps=NW,
    )
    return out


# Alias for FlagGems import convention
binary_cross_entropy_backward = run
