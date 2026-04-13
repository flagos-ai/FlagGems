import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.fused.{__name__.split(".")[-1]}')

# NOTE: On Ascend NPU, the erf function has limited precision.
# For better accuracy, we use tanh approximation for both "none" and "tanh" modes.
# The tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

# -------------------- forward kernel --------------------


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("gelu_and_mul"), key=["N"])
@triton.jit
def gelu_and_mul_kernel(
    X,
    Y,
    OUT,
    N,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tle.program_id(0)
    base_offset = pid * BLOCK_SIZE
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654

    for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        offsets = base_offset + sub_idx + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask, other=0.0, care_padding=False).to(
            tl.float32
        )
        y = tl.load(Y + offsets, mask=mask, other=0.0, care_padding=False)
        # Use tanh approximation for better accuracy on Ascend NPU
        x_gelu = (
            0.5
            * x
            * (1.0 + tl.math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))
        )
        out = x_gelu * y
        tl.store(OUT + offsets, out.to(OUT.dtype.element_ty), mask=mask)


# -------------------- backward kernel --------------------


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("gelu_and_mul_grad"), key=["N"])
@triton.jit
def gelu_and_mul_grad_kernel(
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
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654

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

        # Forward: use tanh approximation for better accuracy on Ascend NPU
        tanh_arg = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
        tanh_val = tl.math.tanh(tanh_arg)
        x_gelu = 0.5 * x * (1.0 + tanh_val)

        dy = x_gelu * dgrad

        # Backward: compute gradient w.r.t. x
        tanh_sq = tanh_val * tanh_val
        term1 = 0.5 * (1.0 + tanh_val)
        term2 = (
            0.5
            * x
            * (1.0 - tanh_sq)
            * (SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x * x))
        )
        dx = dgrad * y * (term1 + term2)

        tl.store(DX + offsets, dx.to(DX.dtype.element_ty), mask=mask)
        tl.store(DY + offsets, dy.to(DY.dtype.element_ty), mask=mask)


# -------------------- autograd wrapper --------------------


class GeluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, approximate="none"):
        logger.debug("GEMS_ASCEND GELU AND MUL FORWARD")
        ctx.save_for_backward(x, y)
        ctx.approximate = approximate
        x = x.contiguous()
        y = y.contiguous()
        out = torch.empty_like(x)
        N = x.numel()

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(x.device):
            # NOTE: On Ascend NPU, we use tanh approximation for both modes
            # due to limited precision of the erf function on this hardware.
            gelu_and_mul_kernel[grid](x, y, out, N)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_ASCEND GELU AND MUL BACKWARD")
        x, y = ctx.saved_tensors
        x = x.contiguous()
        y = y.contiguous()
        grad_output = grad_output.contiguous()
        grad_x = torch.empty_like(x)
        grad_y = torch.empty_like(y)
        N = x.numel()

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(x.device):
            # NOTE: On Ascend NPU, we use tanh approximation for both modes
            # due to limited precision of the erf function on this hardware.
            gelu_and_mul_grad_kernel[grid](
                x, y, grad_output, grad_x, grad_y, N
            )
        return grad_x, grad_y, None


def gelu_and_mul(x, y, approximate="none"):
    return GeluAndMul.apply(x, y, approximate)
