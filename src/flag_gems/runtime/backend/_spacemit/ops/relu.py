import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("relu_forward"),
    key=["n_elements"],
)
@triton.jit
def relu_forward_kernel(
    X_ptr,
    Out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        x_blk = tl.make_block_ptr(
            base=X_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        x = tl.load(x_blk, boundary_check=(0,))
        out = tl.where(x > 0, x, 0)
        tl.store(out_blk, out, boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("relu_backward"),
    key=["n_elements"],
)
@triton.jit
def relu_backward_kernel(
    X_ptr,
    DY_ptr,
    DX_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        x_blk = tl.make_block_ptr(
            base=X_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        dy_blk = tl.make_block_ptr(
            base=DY_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        dx_blk = tl.make_block_ptr(
            base=DX_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        x = tl.load(x_blk, boundary_check=(0,))
        dy = tl.load(dy_blk, boundary_check=(0,))
        dx = tl.where(x > 0, dy, 0)
        tl.store(dx_blk, dx, boundary_check=(0,))


class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("GEMS_SPACEMIT RELU_FORWARD")
        A = A.contiguous()
        out = torch.empty_like(A)
        n_elements = A.numel()
        with torch_device_fn.device(A.device):
            relu_forward_kernel[(NUM_CTAS,)](A, out, n_elements)
        ctx.save_for_backward(A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_SPACEMIT RELU_BACKWARD")
        (inp,) = ctx.saved_tensors
        inp = inp.contiguous()
        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out_grad)
        n_elements = inp.numel()
        with torch_device_fn.device(inp.device):
            relu_backward_kernel[(NUM_CTAS,)](inp, out_grad, in_grad, n_elements)
        return in_grad


def relu(A):
    return Relu.apply(A)


def relu_(A):
    logger.debug("GEMS_SPACEMIT RELU_")
    A_contig = A.contiguous()
    n_elements = A_contig.numel()
    with torch_device_fn.device(A.device):
        relu_forward_kernel[(NUM_CTAS,)](A_contig, A_contig, n_elements)
    if not A.is_contiguous():
        A.copy_(A_contig)
    return A
