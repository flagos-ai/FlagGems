import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)
erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow
tanh = tl_extra_shim.tanh
geluTanh = tl_extra_shim.gelu_tanh
geluNone = tl_extra_shim.gelu_none

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gelu_none_forward"),
    key=["n_elements"],
)
@triton.jit
def gelu_none_kernel(
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
        out = geluNone(x.to(tl.float32))
        tl.store(out_blk, out.to(x.dtype), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gelu_tanh_forward"),
    key=["n_elements"],
)
@triton.jit
def gelu_tanh_kernel(
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
        out = geluTanh(x.to(tl.float32))
        tl.store(out_blk, out.to(x.dtype), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gelu_none_backward"),
    key=["n_elements"],
)
@triton.jit
def gelu_backward_none_kernel(
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
        x_fp32 = x.to(tl.float32)
        scale1: tl.constexpr = 0.7071067811
        scale2: tl.constexpr = 0.3989422803
        dydx = (
            scale2 * x_fp32 * exp(-pow(scale1 * x_fp32, 2))
            + 0.5 * erf(scale1 * x_fp32)
            + 0.5
        )
        dx = dydx * dy
        tl.store(dx_blk, dx.to(dy.dtype), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gelu_tanh_backward"),
    key=["n_elements"],
)
@triton.jit
def gelu_backward_tanh_kernel(
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
        x_fp32 = x.to(tl.float32)
        tanh_out = tanh(0.79788456 * x_fp32 * (1.0 + 0.044715 * pow(x_fp32, 2)))
        dydx = 0.5 * x_fp32 * (
            (1.0 - pow(tanh_out, 2)) * (0.79788456 + 0.1070322243 * pow(x_fp32, 2))
        ) + 0.5 * (1.0 + tanh_out)
        dx = dydx * dy
        tl.store(dx_blk, dx.to(dy.dtype), boundary_check=(0,))


class Gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, approximate):
        logging.debug("GEMS_SPACEMIT GELU_FORWARD")
        A = A.contiguous()
        out = torch.empty_like(A)
        n_elements = A.numel()
        with torch_device_fn.device(A.device):
            if approximate == "tanh":
                gelu_tanh_kernel[(NUM_CTAS,)](A, out, n_elements)
            else:
                gelu_none_kernel[(NUM_CTAS,)](A, out, n_elements)
        ctx.save_for_backward(A)
        ctx.approximate = approximate
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS_SPACEMIT GELU_BACKWARD")
        (inp,) = ctx.saved_tensors
        approximate = ctx.approximate
        inp = inp.contiguous()
        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out_grad)
        n_elements = inp.numel()
        with torch_device_fn.device(inp.device):
            if approximate == "tanh":
                gelu_backward_tanh_kernel[(NUM_CTAS,)](
                    inp, out_grad, in_grad, n_elements
                )
            else:
                gelu_backward_none_kernel[(NUM_CTAS,)](
                    inp, out_grad, in_grad, n_elements
                )
        return in_grad, None


def gelu(A, *, approximate="none"):
    return Gelu.apply(A, approximate)
