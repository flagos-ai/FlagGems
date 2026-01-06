import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
exp2 = tl_extra_shim.exp2


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_forward(x):
    # log2e: tl.constexpr = math.log2(math.e)
    # triton 3.0.0 disallow calling non-jitted function inside jitted function, even if it is in
    # the rhs of an assignment to a constexpr, so we use numeric literal instead to work around this.
    log2e: tl.constexpr = 1.4426950408889634
    return 1 / (1 + exp2(-x.to(tl.float32) * log2e))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_backward(y, dy):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    return dy_f32 * (1.0 - y_f32) * y_f32


@triton.jit
def sigmoid_backward_custom_kernel(
    x_ptr: tl.tensor,  # *Pointer* to first input vector.
    y_ptr: tl.tensor,  # *Pointer* to second input vector.
    output_ptr: tl.tensor,  # *Pointer* to output vector.
    n_elements: int,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr`` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)

    # No need to add offset and mask, as its stride is 0
    y = tl.load(y_ptr)

    output = y * (1 - x) * x
    # Write output back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def sigmoid_backward_custom(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sigmoid_backward_custom_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("METAX GEMS SIGMOID FORWARD")
        if A.requires_grad is True:
            out = sigmoid_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = sigmoid_forward(A)
            return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("METAX GEMS SIGMOID BACKWARD")
        (out,) = ctx.saved_tensors

        is_grad_stride_0 = True
        for i in range(len(out_grad.stride())):
            if out_grad.stride()[i] != 0:
                is_grad_stride_0 = False
                break

        # temporay plan
        if (is_grad_stride_0) and (out_grad.numel() % 1024 == 0):
            in_grad = sigmoid_backward_custom(out, out_grad)
            return in_grad
        in_grad = sigmoid_backward(out, out_grad)
        return in_grad


def sigmoid(A):
    return Sigmoid.apply(A)
