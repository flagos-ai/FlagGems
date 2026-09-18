import torch
import triton
import triton.language as tl


@triton.jit
def _zero_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + offs, tl.zeros((BLOCK,), dtype=dtype), mask=mask)


@triton.jit
def _fused_plane_kernel(
    grad_ptr,
    idx_ptr,
    out_ptr,
    acc_ptr,
    in_numel_per_nc,
    out_stride_nc,
    FP32_ACC: tl.constexpr,
    BLOCK_Z: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    plane = tl.program_id(0).to(tl.int64)
    dtype = out_ptr.dtype.element_ty
    in_base = plane * in_numel_per_nc
    out_base = plane * out_stride_nc

    if FP32_ACC:
        # zero the fp32 accumulator
        zbase = 0
        for _ in range(0, tl.cdiv(in_numel_per_nc, BLOCK_Z)):
            offs = in_base + zbase + tl.arange(0, BLOCK_Z)
            mask = tl.arange(0, BLOCK_Z) + zbase < in_numel_per_nc
            tl.store(acc_ptr + offs, tl.zeros((BLOCK_Z,), dtype=tl.float32), mask=mask)
            zbase += BLOCK_Z

        tl.debug_barrier()

        # scatter grads into the fp32 accumulator
        sbase = 0
        for _ in range(0, tl.cdiv(out_stride_nc, BLOCK_S)):
            offs = out_base + sbase + tl.arange(0, BLOCK_S)
            mask = tl.arange(0, BLOCK_S) + sbase < out_stride_nc
            grad = tl.load(grad_ptr + offs, mask=mask)
            idx = tl.load(idx_ptr + offs, mask=mask)
            tl.atomic_add(acc_ptr + in_base + idx, grad, mask=mask)
            sbase += BLOCK_S

        tl.debug_barrier()

        # cast fp32 accumulator -> output dtype
        cbase = 0
        for _ in range(0, tl.cdiv(in_numel_per_nc, BLOCK_C)):
            offs = in_base + cbase + tl.arange(0, BLOCK_C)
            mask = tl.arange(0, BLOCK_C) + cbase < in_numel_per_nc
            val = tl.load(acc_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, val.to(dtype), mask=mask)
            cbase += BLOCK_C
    else:
        # direct accumulation in the output dtype
        zbase = 0
        for _ in range(0, tl.cdiv(in_numel_per_nc, BLOCK_Z)):
            offs = in_base + zbase + tl.arange(0, BLOCK_Z)
            mask = tl.arange(0, BLOCK_Z) + zbase < in_numel_per_nc
            tl.store(out_ptr + offs, tl.zeros((BLOCK_Z,), dtype=dtype), mask=mask)
            zbase += BLOCK_Z

        tl.debug_barrier()

        sbase = 0
        for _ in range(0, tl.cdiv(out_stride_nc, BLOCK_S)):
            offs = out_base + sbase + tl.arange(0, BLOCK_S)
            mask = tl.arange(0, BLOCK_S) + sbase < out_stride_nc
            grad = tl.load(grad_ptr + offs, mask=mask)
            idx = tl.load(idx_ptr + offs, mask=mask)
            tl.atomic_add(out_ptr + in_base + idx, grad, mask=mask)
            sbase += BLOCK_S


def run(grad_output, self, indices):
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    out_flat = out.view(-1)
    out_numel = out_flat.numel()
    grad_flat = grad_output.view(-1)
    idx_flat = indices.view(-1)
    n = grad_flat.numel()

    if out_numel == 0 or n == 0:
        if out_numel > 0:
            _zero_kernel[(triton.cdiv(out_numel, 1024),)](
                out_flat, out_numel, BLOCK=1024
            )
        return out

    in_h = self.shape[-2]
    in_w = self.shape[-1]
    out_h = grad_output.shape[-2]
    out_w = grad_output.shape[-1]
    in_numel_per_nc = in_h * in_w
    out_stride_nc = out_h * out_w
    num_planes = n // out_stride_nc

    # With an exact region partition (out_h | in_h and out_w | in_w) every input
    # element belongs to exactly one output window, so forward indices are
    # unique and direct output-dtype atomics are exact and deterministic.
    # Otherwise adaptive windows overlap and the same input can receive
    # several gradients: accumulate in fp32 (matching the upcast reference)
    # and cast once, so the result does not depend on atomic order.
    fp32_acc = (self.dtype != torch.float32) and (
        (in_h % out_h != 0) or (in_w % out_w != 0)
    )
    if fp32_acc:
        acc = torch.empty(self.shape, dtype=torch.float32, device=self.device)
        acc_flat = acc.view(-1)
    else:
        acc_flat = out_flat

    _fused_plane_kernel[(num_planes,)](
        grad_flat,
        idx_flat,
        out_flat,
        acc_flat,
        in_numel_per_nc,
        out_stride_nc,
        FP32_ACC=fp32_acc,
        BLOCK_Z=1024,
        BLOCK_S=1024,
        BLOCK_C=1024,
    )
    return out


# Alias for FlagGems import convention
adaptive_max_pool2d_backward = run
