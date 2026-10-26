import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)
exp_func = tl_extra_shim.exp

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sigmoid_forward"),
    key=["n_elements"],
)
@triton.jit
def sigmoid_forward_kernel(
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
        out = 1.0 / (1.0 + exp_func(-x.to(tl.float32)))
        tl.store(out_blk, out.to(x.dtype), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sigmoid_backward"),
    key=["n_elements"],
)
@triton.jit
def sigmoid_backward_kernel(
    DY_ptr,
    Y_ptr,
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

        dy_blk = tl.make_block_ptr(
            base=DY_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        y_blk = tl.make_block_ptr(
            base=Y_ptr,
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

        dy = tl.load(dy_blk, boundary_check=(0,))
        y = tl.load(y_blk, boundary_check=(0,))
        y_f32 = y.to(tl.float32)
        dy_f32 = dy.to(tl.float32)
        dx = dy_f32 * (1.0 - y_f32) * y_f32
        tl.store(dx_blk, dx.to(dy.dtype), boundary_check=(0,))


def sigmoid(self):
    logger.debug("GEMS_SPACEMIT SIGMOID_FORWARD")
    A = self.contiguous()
    out = torch.empty_like(A)
    n_elements = A.numel()
    with torch_device_fn.device(A.device):
        sigmoid_forward_kernel[(NUM_CTAS,)](A, out, n_elements)
    return out


def sigmoid_backward(grad_output, output):
    logger.debug("GEMS_SPACEMIT SIGMOID_BACKWARD")
    grad_output = grad_output.contiguous()
    output = output.contiguous()
    grad_input = torch.empty_like(grad_output)
    n_elements = grad_output.numel()
    with torch_device_fn.device(grad_output.device):
        sigmoid_backward_kernel[(NUM_CTAS,)](grad_output, output, grad_input, n_elements)
    return grad_input


def sigmoid_(A):
    logger.debug("GEMS_SPACEMIT SIGMOID__FORWARD")
    A_contig = A.contiguous()
    n_elements = A_contig.numel()
    with torch_device_fn.device(A.device):
        sigmoid_forward_kernel[(NUM_CTAS,)](A_contig, A_contig, n_elements)
    if not A.is_contiguous():
        A.copy_(A_contig)
    return A
