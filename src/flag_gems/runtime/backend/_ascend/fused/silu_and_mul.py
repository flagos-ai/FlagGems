import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.fused.{__name__.split(".")[-1]}')


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("silu_and_mul"), key=["N"])
@triton.jit
def silu_and_mul_kernel(
    X,
    Y,
    OUT,
    N,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tle.program_id(0)
    base_offset = pid * BLOCK_SIZE

    for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        offsets = base_offset + sub_idx + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask, other=0.0, care_padding=False).to(
            tl.float32
        )
        y = tl.load(Y + offsets, mask=mask, other=0.0, care_padding=False)
        x_silu = x / (1.0 + tl.exp(-x))
        out = x_silu * y
        tl.store(OUT + offsets, out.to(OUT.dtype.element_ty), mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("silu_and_mul_grad"), key=["N"])
@triton.jit
def silu_and_mul_grad_kernel(
    X,
    Y,
    DGRAD,
    DX,
    DY,
    N,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tle.program_id(0)
    base_offset = pid * BLOCK_SIZE

    for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        offsets = base_offset + sub_idx + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask, other=0.0, care_padding=False).to(
            tl.float32
        )
        y = tl.load(Y + offsets, mask=mask, other=0.0, care_padding=False).to(
            tl.float32
        )
        dgrad = tl.load(DGRAD + offsets, mask=mask, other=0.0, care_padding=False).to(
            tl.float32
        )
        sig = 1.0 / (1.0 + tl.exp(-x))
        x_silu = x * sig
        d_x_silu = sig * (1.0 + x * (1.0 - sig))
        dx = d_x_silu * dgrad * y
        dy = dgrad * x_silu
        tl.store(DX + offsets, dx.to(DX.dtype.element_ty), mask=mask)
        tl.store(DY + offsets, dy.to(DY.dtype.element_ty), mask=mask)


class SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        logger.debug("GEMS_ASCEND SILU AND MUL FORWARD")
        ctx.save_for_backward(A, B)
        A = A.contiguous()
        B = B.contiguous()
        out = torch.empty_like(A)
        N = A.numel()

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(A.device):
            silu_and_mul_kernel[grid](A, B, out, N)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_ASCEND SILU AND MUL BACKWARD")
        A, B = ctx.saved_tensors
        A = A.contiguous()
        B = B.contiguous()
        grad_output = grad_output.contiguous()
        grad_A = torch.empty_like(A)
        grad_B = torch.empty_like(B)
        N = A.numel()

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(A.device):
            silu_and_mul_grad_kernel[grid](
                A, B, grad_output, grad_A, grad_B, N
            )
        return grad_A, grad_B


def silu_and_mul(A, B):
    return SiluAndMul.apply(A, B)


def silu_and_mul_out(A, B, out):
    A = A.contiguous()
    B = B.contiguous()
    N = A.numel()

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(A.device):
        silu_and_mul_kernel[grid](A, B, out, N)
    return out
