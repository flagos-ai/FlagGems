# This custom op requires musa device capability >= 31.
# We determine whether to enable this op by distinguish the op registration for different arch.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)
erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow
fast_tanh = tl_extra_shim.fast_tanh
fast_gelu = tl_extra_shim.fast_gelu


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_none(x):
    return fast_gelu(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x):
    output = (
        0.5
        * x
        * (1 + fast_tanh(x * 0.79788456 * (1 + 0.044715 * pow(x.to(tl.float32), 2))))
    )
    return output


# Manual kernel implementation for gelu_backward_none with autotune
@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def gelu_backward_none_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)

    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)

    x_fp32 = x.to(tl.float32)
    scaled_x = scale1 * x_fp32
    dydx = scale2 * x_fp32 * exp(-scaled_x * scaled_x) + 0.5 * erf(scaled_x) + 0.5
    dx = dydx * dy

    tl.store(dx_ptr + offsets, dx, mask=mask)


# Manual kernel implementation for gelu_backward_tanh with autotune
@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def gelu_backward_tanh_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)

    x_fp32 = x.to(tl.float32)
    # 0.79788456 = math.sqrt(2 / math.pi)
    x2 = x_fp32 * x_fp32
    tanh_out = fast_tanh(0.79788456 * x_fp32 * (1.0 + 0.044715 * x2))
    tanh_out_sq = tanh_out * tanh_out
    dydx = 0.5 * x_fp32 * ((1.0 - tanh_out_sq) * (0.79788456 + 0.1070322243 * x2)) + 0.5 * (1.0 + tanh_out)
    dx = dydx * dy

    tl.store(dx_ptr + offsets, dx, mask=mask)


# Keep pointwise_dynamic versions for non-contiguous tensors
@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_none_pointwise(x, dy):
    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)
    x_fp32 = x.to(tl.float32)
    scaled_x = scale1 * x_fp32
    dydx = scale2 * x_fp32 * exp(-scaled_x * scaled_x) + 0.5 * erf(scaled_x) + 0.5
    dx = dydx * dy
    return dx


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_tanh_pointwise(x, dy):
    x_fp32 = x.to(tl.float32)
    # 0.79788456 = math.sqrt(2 / math.pi)
    x2 = x_fp32 * x_fp32
    tanh_out = fast_tanh(0.79788456 * x_fp32 * (1.0 + 0.044715 * x2))
    tanh_out_sq = tanh_out * tanh_out
    dydx = 0.5 * x_fp32 * ((1.0 - tanh_out_sq) * (0.79788456 + 0.1070322243 * x2)) + 0.5 * (1.0 + tanh_out)
    dx = dydx * dy
    return dx


class Gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, approximate):
        logger.debug("GEMS_MTHREADS GELU FORWARD")
        if approximate == "tanh":
            out = gelu_tanh(A)
        else:
            out = gelu_none(A)
        ctx.save_for_backward(A)
        ctx.approximate = approximate
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_MTHREADS GELU BACKWARD")
        (inp,) = ctx.saved_tensors
        approximate = ctx.approximate
        in_grad = _gelu_backward_impl(out_grad, inp, approximate)
        return in_grad, None


def _gelu_backward_impl(grad_output, self, approximate):
    """Internal implementation for gelu_backward with optimized kernels."""
    # Use manual kernel for contiguous tensors
    if self.is_contiguous() and grad_output.is_contiguous():
        dx = torch.empty_like(self)
        n_elements = self.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(self.device):
            if approximate == "tanh":
                gelu_backward_tanh_kernel[grid](self, grad_output, dx, n_elements)
            else:
                gelu_backward_none_kernel[grid](self, grad_output, dx, n_elements)
        return dx
    else:
        # Fallback to pointwise_dynamic for non-contiguous tensors
        if approximate == "tanh":
            return gelu_backward_tanh_pointwise(self, grad_output)
        else:
            return gelu_backward_none_pointwise(self, grad_output)


def gelu(A, *, approximate="none"):
    return Gelu.apply(A, approximate)


def gelu_backward(grad_output, self, *, approximate="none"):
    """Optimized gelu_backward for mthreads backend."""
    logger.debug("GEMS_MTHREADS GELU_BACKWARD")
    return _gelu_backward_impl(grad_output, self, approximate)
