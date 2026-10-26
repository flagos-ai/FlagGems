import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)
div_rn = tl_extra_shim.div_rn
_silu = tl_extra_shim.silu

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("silu_forward"),
    key=["n_elements"],
)
@triton.jit
def silu_forward_kernel(
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
        x_fp32 = x.to(tl.float32)
        out = _silu(x_fp32)
        tl.store(out_blk, out.to(x.dtype), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("silu_backward"),
    key=["n_elements"],
)
@triton.jit
def silu_backward_kernel(
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
        dy_fp32 = dy.to(tl.float32)
        x_fp32 = x.to(tl.float32)
        sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32))
        dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
        tl.store(dx_blk, dx.to(dy.dtype), boundary_check=(0,))


class Silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS_SPACEMIT SILU_FORWARD")
        A = A.contiguous()
        out = torch.empty_like(A)
        n_elements = A.numel()
        with torch_device_fn.device(A.device):
            silu_forward_kernel[(NUM_CTAS,)](A, out, n_elements)
        ctx.save_for_backward(A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS_SPACEMIT SILU_BACKWARD")
        (inp,) = ctx.saved_tensors
        inp = inp.contiguous()
        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out_grad)
        n_elements = inp.numel()
        with torch_device_fn.device(inp.device):
            silu_backward_kernel[(NUM_CTAS,)](
                inp, out_grad, in_grad, n_elements
            )
        return in_grad


def silu(A):
    return Silu.apply(A)


class InplaceSilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS_SPACEMIT SILU__FORWARD")
        ctx.save_for_backward(A.clone())
        ctx.mark_dirty(A)
        A_contig = A.contiguous()
        n_elements = A_contig.numel()
        with torch_device_fn.device(A.device):
            silu_forward_kernel[(NUM_CTAS,)](A_contig, A_contig, n_elements)
        if not A.is_contiguous():
            A.copy_(A_contig)
        return A

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS_SPACEMIT SILU__BACKWARD")
        (inp,) = ctx.saved_tensors
        inp = inp.contiguous()
        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out_grad)
        n_elements = inp.numel()
        with torch_device_fn.device(inp.device):
            silu_backward_kernel[(NUM_CTAS,)](
                inp, out_grad, in_grad, n_elements
            )
        return in_grad


def silu_(A):
    InplaceSilu.apply(A)
    return A
