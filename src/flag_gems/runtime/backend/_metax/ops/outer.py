import logging

import torch
import triton
import triton.language as tl

# from flag_gems.ops.mul import mul
from flag_gems.ops.mv import mv

logger = logging.getLogger(__name__)


@triton.jit
def mul_outer_kernel(
    inp,
    weight,
    out,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    n_range = tl.arange(0, BLOCK_SIZE_N)
    weight_block_start = pid_y * BLOCK_SIZE_N
    weight_offsets = weight_block_start + n_range[None, :]
    mask_2 = weight_offsets < N
    weight_data = tl.load(weight + weight_offsets, mask=mask_2)
    for i in range(0, BLOCK_SIZE_M):
        inp_offsets = pid_x * BLOCK_SIZE_M + i
        mask_1 = inp_offsets < M
        output_offsets = (pid_x * BLOCK_SIZE_M + i) * N + weight_offsets
        # mask_3 = output_offsets < (M * N)
        inp_data = tl.load(inp + inp_offsets, mask=mask_1)
        inp_bd, weight_bd = tl.broadcast(inp_data, weight_data)
        output = inp_bd * weight_bd
        tl.store(out + output_offsets, output, mask=mask_2)


def mul(inp, weight):
    assert inp.ndim == 2 and weight.ndim == 2, "Invalid input"
    assert inp.shape[1] == 1 and weight.shape[0] == 1, "Invalid input"
    M = inp.shape[0]
    N = weight.shape[1]
    out = torch.empty((M, N), device=inp.device, dtype=inp.dtype)
    num_warps = 1
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 512
    grid = lambda META: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    with torch.cuda.device(inp.device):
        mul_outer_kernel[grid](
            inp,
            weight,
            out,
            M,
            N,
            inp.stride(0),
            weight.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=num_warps,
        )
    return out


class Outer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight):
        logger.debug("METAX GEMS OUTER")
        assert inp.ndim == 1 and weight.ndim == 1, "Invalid input"
        inp1 = inp[:, None]
        weight1 = weight[None, :]
        inp1 = inp1.contiguous()
        weight1 = weight1.contiguous()
        out = mul(inp1, weight1)
        ctx.save_for_backward(inp, weight)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("METAX GEMS OUTER VJP")
        assert out_grad.ndim == 2, "invalide out_grad shape"

        inp, weight = ctx.saved_tensors

        inp_grad = mv(out_grad, weight)
        weight_grad = mv(out_grad.t(), inp)

        return inp_grad, weight_grad


def outer(inp, weight):
    return Outer.apply(inp, weight)
