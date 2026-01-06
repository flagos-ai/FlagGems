import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


def heur_num_warps(args):
    if args["N"] <= 1024:
        return 1
    elif args["N"] <= 2048:
        return 4
    else:
        return 8


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("log_softmax"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "num_warps": heur_num_warps,
    }
)
@triton.jit
def log_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K
    if USE_K:
        offset += pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = tl.log(numerator / denominator)
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, softmax_output, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("log_softmax"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "num_warps": heur_num_warps,
    }
)
@triton.jit
def log_softmax_backward_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_SPLIT: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_split_offset = tl.arange(0, BLOCK_N_SPLIT)
    n_offset = tl.arange(0, BLOCK_N)
    all_offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    out_grad_ptrs_all = out_grad_ptr + all_offsets
    all_mask = m_offset[:, None] < M and n_offset[None, :] < N
    out_grad_all = tl.load(out_grad_ptrs_all, mask=all_mask).to(tl.float32)
    scale = tl.sum(out_grad_all, 1)
    # use for loop to split N dim to reduce register cost
    for n in range(0, tl.cdiv(BLOCK_N, BLOCK_N_SPLIT)):
        offsets = (
            m_offset[:, None] * N * K
            + n_split_offset[None, :] * K
            + n * BLOCK_N_SPLIT * K
            + pid_k
        )
        mask = m_offset[:, None] < M and n_split_offset[None, :] + n * BLOCK_N_SPLIT < N
        out_ptrs = out_ptr + offsets
        out = tl.load(out_ptrs, mask=mask).to(tl.float32)
        exp_out = tl.exp(out.to(tl.float32))
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)

        in_grad = out_grad - exp_out * scale[:, None]
        in_grad_ptrs = in_grad_ptr + offsets
        tl.store(in_grad_ptrs, in_grad, mask=mask)


def log_softmax(self, dim, half_to_float=False):
    logger.debug("METAX GEMS LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N
    USE_K = K != 1

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch_device_fn.device(inp.device):
        log_softmax_kernel[grid](
            out,
            inp,
            M,
            N,
            K,
            USE_K=USE_K,
        )
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("METAX GEMS LOG_SOFTMAX VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    in_grad = torch.empty_like(output, dtype=input_dtype)
    K = output.numel() // M // N

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch_device_fn.device(in_grad.device):
        log_softmax_backward_kernel[grid](
            output,
            grad_output,
            in_grad,
            M,
            N,
            K,
            BLOCK_N_SPLIT=1024,
        )
    return in_grad
